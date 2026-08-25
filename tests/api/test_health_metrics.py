import httpx
import pytest
from redis.exceptions import RedisError

from app.core.config import Settings
from app.main import create_app


class HealthyRedis:
    async def ping(self):
        return True


class BrokenRedis:
    async def ping(self):
        raise RedisError("offline")


async def make_client(redis_client, provider_count=1):
    app = create_app(Settings(client_api_keys="frontend:key-a"))
    app.state.redis = redis_client
    app.state.provider_count = provider_count
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def test_health_is_live_when_redis_is_down():
    async with await make_client(BrokenRedis()) as client:
        health = await client.get("/health")
        ready = await client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 503


@pytest.mark.parametrize("provider_count", [0])
async def test_ready_requires_one_configured_provider(provider_count):
    async with await make_client(HealthyRedis(), provider_count) as client:
        response = await client.get("/ready")

    assert response.status_code == 503


async def test_ready_accepts_redis_and_provider():
    async with await make_client(HealthyRedis()) as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_metrics_expose_gateway_instruments():
    async with await make_client(HealthyRedis()) as client:
        await client.get("/health")
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_requests_total" in response.text
    assert "gateway_request_duration_seconds" in response.text
    assert "gateway_cache_hits_total" in response.text
    assert "gateway_circuit_state" in response.text
    assert "client" not in response.text.lower()

