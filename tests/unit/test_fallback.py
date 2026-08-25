from collections.abc import AsyncIterator

import pytest

from app.core.errors import GatewayError
from app.providers.base import ProviderChunk, ProviderError
from app.schemas.chat import ChatRequest, ProviderResult, Usage
from app.services.fallback import FallbackExecutor
from app.services.router import RouteCandidate


class FakeCircuit:
    def __init__(self, blocked=()):
        self.blocked = set(blocked)
        self.successes = []
        self.failures = []

    async def allow(self, provider):
        return provider not in self.blocked

    async def success(self, provider):
        self.successes.append(provider)

    async def failure(self, provider):
        self.failures.append(provider)


class FakeProvider:
    def __init__(self, name, *, result=None, error=None, chunks=None):
        self.name = name
        self.result = result
        self.error = error
        self.chunks = chunks or []
        self.calls = 0

    async def complete(self, request, model):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    async def stream(self, request, model) -> AsyncIterator[ProviderChunk]:
        self.calls += 1
        if self.error:
            raise self.error
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def chat_request():
    return ChatRequest(messages=[{"role": "user", "content": "hello"}])


def result(provider):
    return ProviderResult(
        model=f"{provider}-model",
        provider=provider,
        content="ok",
        usage=Usage(total_tokens=1),
    )


async def test_fallback_skips_retryable_failure_and_counts_attempts(chat_request):
    deepseek = FakeProvider(
        "deepseek",
        error=ProviderError("deepseek", "timeout", None, True),
    )
    openai = FakeProvider("openai", result=result("openai"))
    circuit = FakeCircuit()
    executor = FallbackExecutor({"deepseek": deepseek, "openai": openai}, circuit)

    response, fallback_count = await executor.complete(
        chat_request,
        [RouteCandidate("deepseek", "ds"), RouteCandidate("openai", "gpt")],
    )

    assert response.provider == "openai"
    assert fallback_count == 1
    assert circuit.failures == ["deepseek"]
    assert circuit.successes == ["openai"]


async def test_non_retryable_provider_error_stops_fallback(chat_request):
    deepseek = FakeProvider(
        "deepseek",
        error=ProviderError("deepseek", "bad_request", 400, False),
    )
    openai = FakeProvider("openai", result=result("openai"))
    executor = FallbackExecutor({"deepseek": deepseek, "openai": openai}, FakeCircuit())

    with pytest.raises(ProviderError) as error:
        await executor.complete(
            chat_request,
            [RouteCandidate("deepseek", "ds"), RouteCandidate("openai", "gpt")],
        )

    assert error.value.retryable is False
    assert openai.calls == 0


async def test_all_upstream_rate_limits_return_429(chat_request):
    providers = {
        name: FakeProvider(name, error=ProviderError(name, "rate", 429, True))
        for name in ("deepseek", "openai")
    }
    executor = FallbackExecutor(providers, FakeCircuit())

    with pytest.raises(GatewayError) as error:
        await executor.complete(
            chat_request,
            [RouteCandidate("deepseek", "ds"), RouteCandidate("openai", "gpt")],
        )

    assert error.value.status_code == 429


async def test_stream_falls_back_before_first_content(chat_request):
    deepseek = FakeProvider("deepseek", chunks=[])
    openai = FakeProvider(
        "openai",
        chunks=[ProviderChunk(content="first"), ProviderChunk(content="second")],
    )
    executor = FallbackExecutor(
        {"deepseek": deepseek, "openai": openai},
        FakeCircuit(),
    )

    candidate, chunks, fallback_count = await executor.stream_until_first_chunk(
        chat_request,
        [RouteCandidate("deepseek", "ds"), RouteCandidate("openai", "gpt")],
    )

    assert candidate.provider == "openai"
    assert [chunk.content async for chunk in chunks] == ["first", "second"]
    assert fallback_count == 1
