from contextlib import asynccontextmanager

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import GatewayError
from app.main import create_app
from app.schemas.chat import ProviderResult, Usage
from app.services.cache import SingleFlightResult
from app.services.gateway import GatewayService
from app.services.rate_limiter import ConcurrencyLease, RateLimitDecision
from app.services.router import RouteCandidate

AUTH = {"X-API-Key": "key-a"}
REQUEST = {
    "model": "auto",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": False,
}


class FakeRateLimiter:
    def __init__(self):
        self.decision = RateLimitDecision(True, 59, 0)
        self.released = []

    async def acquire_rate(self, client):
        return self.decision

    async def acquire_concurrency(self, client, request_id):
        return ConcurrencyLease(client, request_id)

    async def release_concurrency(self, lease):
        self.released.append(lease)


class FakeCache:
    def __init__(self):
        self.values = {}

    def _key(self, client, request):
        return client, request.model_dump_json()

    async def get(self, client, request):
        return self.values.get(self._key(client, request))

    async def set(self, client, request, response):
        self.values[self._key(client, request)] = response
        return True

    @asynccontextmanager
    async def single_flight(self, client, request):
        yield SingleFlightResult(owner=True)


class FakeRouter:
    def candidates(self, model):
        return [RouteCandidate("deepseek", "deepseek-chat")]


class FakeFallback:
    def __init__(self):
        self.calls = 0
        self.error = None
        self.fallback_count = 0

    async def complete(self, request, candidates):
        self.calls += 1
        if self.error:
            raise self.error
        return (
            ProviderResult(
                model="deepseek-chat",
                provider="deepseek",
                content="hi",
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            self.fallback_count,
        )


@pytest.fixture
def components():
    return FakeRateLimiter(), FakeCache(), FakeFallback()


@pytest.fixture
async def client(components):
    rate_limiter, cache, fallback = components
    settings = Settings(client_api_keys="frontend:key-a", deepseek_api_key="ds")
    gateway = GatewayService(
        settings=settings,
        rate_limiter=rate_limiter,
        cache=cache,
        router=FakeRouter(),
        fallback=fallback,
    )
    app = create_app(settings)
    app.state.gateway = gateway
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


async def test_non_streaming_chat_returns_normalized_response(client, components):
    rate_limiter, _, _ = components

    response = await client.post("/chat", headers=AUTH, json=REQUEST)

    assert response.status_code == 200
    assert response.headers["X-Cache"] == "MISS"
    assert response.headers["X-Request-ID"]
    assert response.json()["provider"] == "deepseek"
    assert response.json()["message"] == {"role": "assistant", "content": "hi"}
    assert len(rate_limiter.released) == 1


async def test_cache_hit_does_not_call_provider(client, components):
    _, _, fallback = components

    first = await client.post("/chat", headers=AUTH, json=REQUEST)
    second = await client.post("/chat", headers=AUTH, json=REQUEST)

    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["cached"] is True
    assert fallback.calls == 1


async def test_missing_key_is_rejected(client):
    response = await client.post("/chat", json=REQUEST)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_rate_limit_returns_429_headers(client, components):
    rate_limiter, _, fallback = components
    rate_limiter.decision = RateLimitDecision(False, 0, 2)

    response = await client.post("/chat", headers=AUTH, json=REQUEST)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    assert fallback.calls == 0


async def test_fallback_metadata_and_request_id_are_returned(client, components):
    _, _, fallback = components
    fallback.fallback_count = 1

    response = await client.post(
        "/chat",
        headers={**AUTH, "X-Request-ID": "req-client-1"},
        json=REQUEST,
    )

    assert response.headers["X-Request-ID"] == "req-client-1"
    assert response.json()["fallback_count"] == 1


async def test_concurrency_is_released_when_fallback_fails(client, components):
    rate_limiter, _, fallback = components
    fallback.error = GatewayError("upstream_unavailable", "offline", 503, True)

    response = await client.post("/chat", headers=AUTH, json=REQUEST)

    assert response.status_code == 503
    assert len(rate_limiter.released) == 1
