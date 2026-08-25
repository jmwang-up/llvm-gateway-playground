from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from app.core.auth import ClientIdentity
from app.core.config import Settings
from app.main import create_app
from app.providers.base import ProviderChunk, ProviderError
from app.schemas.chat import ChatRequest, Usage
from app.services.cache import SingleFlightResult
from app.services.gateway import GatewayService
from app.services.rate_limiter import ConcurrencyLease, RateLimitDecision
from app.services.router import RouteCandidate

AUTH = {"X-API-Key": "key-a"}
REQUEST = {
    "model": "auto",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}


class StreamRateLimiter:
    def __init__(self):
        self.decision = RateLimitDecision(True, 59, 0)
        self.released = []

    async def acquire_rate(self, client):
        return self.decision

    async def acquire_concurrency(self, client, request_id):
        return ConcurrencyLease(client, request_id)

    async def release_concurrency(self, lease):
        self.released.append(lease)


class CacheSpy:
    def __init__(self):
        self.calls = 0

    async def get(self, *args):
        self.calls += 1

    async def set(self, *args):
        self.calls += 1

    @asynccontextmanager
    async def single_flight(self, *args):
        self.calls += 1
        yield SingleFlightResult(owner=True)


class StreamRouter:
    def candidates(self, model):
        return [RouteCandidate("deepseek", "deepseek-chat")]


class StreamFallback:
    def __init__(self, chunks=None, *, provider="deepseek", fallback_count=0):
        self.chunks = chunks or []
        self.provider = provider
        self.fallback_count = fallback_count
        self.calls = 0

    async def stream_until_first_chunk(self, request, candidates):
        self.calls += 1

        async def generate() -> AsyncIterator[ProviderChunk]:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        return (
            RouteCandidate(self.provider, f"{self.provider}-model"),
            generate(),
            self.fallback_count,
        )


def build_gateway(fallback):
    settings = Settings(client_api_keys="frontend:key-a", deepseek_api_key="ds")
    rate_limiter = StreamRateLimiter()
    cache = CacheSpy()
    gateway = GatewayService(
        settings=settings,
        rate_limiter=rate_limiter,
        cache=cache,
        router=StreamRouter(),
        fallback=fallback,
    )
    return settings, gateway, rate_limiter, cache


async def post_stream(fallback):
    settings, gateway, rate_limiter, cache = build_gateway(fallback)
    app = create_app(settings)
    app.state.gateway = gateway
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", headers=AUTH, json=REQUEST)
    return response, rate_limiter, cache


async def test_stream_emits_meta_deltas_and_done():
    response, rate_limiter, cache = await post_stream(
        StreamFallback(
            [
                ProviderChunk(content="Hi"),
                ProviderChunk(content=" there"),
                ProviderChunk(
                    usage=Usage(
                        prompt_tokens=1,
                        completion_tokens=2,
                        total_tokens=3,
                    )
                ),
            ]
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.index("event: meta") < response.text.index("event: delta")
    assert response.text.index("event: delta") < response.text.index("event: done")
    assert response.text.count("event: delta") == 2
    assert len(rate_limiter.released) == 1
    assert cache.calls == 0


async def test_stream_reports_actual_fallback_provider():
    response, _, _ = await post_stream(
        StreamFallback(
            [ProviderChunk(content="first")],
            provider="openai",
            fallback_count=1,
        )
    )

    assert '"provider":"openai"' in response.text
    assert '"fallback_count":1' in response.text
    assert '"model":"openai/openai-model"' in response.text


async def test_stream_error_after_delta_ends_without_new_fallback():
    fallback = StreamFallback(
        [
            ProviderChunk(content="first"),
            ProviderError("deepseek", "transport_error", None, True),
        ]
    )

    response, _, _ = await post_stream(fallback)

    assert "event: delta" in response.text
    assert "event: error" in response.text
    assert "event: done" not in response.text
    assert fallback.calls == 1


async def test_stream_generator_close_releases_concurrency_lease():
    fallback = StreamFallback(
        [ProviderChunk(content="first"), ProviderChunk(content="second")]
    )
    _, gateway, rate_limiter, _ = build_gateway(fallback)
    stream = await gateway.stream(
        ClientIdentity("frontend"),
        ChatRequest.model_validate(REQUEST),
        "req-close",
    )

    first = await anext(stream)
    assert first.event == "meta"
    await stream.aclose()

    assert len(rate_limiter.released) == 1


async def test_stream_rate_limit_is_regular_http_error():
    fallback = StreamFallback([ProviderChunk(content="first")])
    settings, gateway, rate_limiter, _ = build_gateway(fallback)
    rate_limiter.decision = RateLimitDecision(False, 0, 3)
    app = create_app(settings)
    app.state.gateway = gateway

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/chat", headers=AUTH, json=REQUEST)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3"
    assert fallback.calls == 0
