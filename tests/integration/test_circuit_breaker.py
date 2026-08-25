import fakeredis.aioredis
import pytest

from app.services.circuit_breaker import RedisCircuitBreaker


@pytest.fixture
def clock():
    value = {"milliseconds": 1_000_000}

    def now() -> int:
        return value["milliseconds"]

    now.advance = lambda milliseconds: value.__setitem__(
        "milliseconds", value["milliseconds"] + milliseconds
    )
    return now


@pytest.fixture
async def redis_client():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def circuit(redis_client, clock):
    return RedisCircuitBreaker(
        redis_client,
        failure_threshold=5,
        open_seconds=30,
        clock_ms=clock,
    )


async def test_circuit_opens_after_threshold_and_allows_one_probe(circuit, clock):
    for _ in range(4):
        await circuit.failure("deepseek")
        assert await circuit.allow("deepseek") is True
    await circuit.failure("deepseek")

    assert await circuit.allow("deepseek") is False
    clock.advance(30_001)
    assert await circuit.allow("deepseek") is True
    assert await circuit.allow("deepseek") is False


async def test_successful_probe_closes_circuit(circuit, clock):
    for _ in range(5):
        await circuit.failure("deepseek")
    clock.advance(30_001)
    assert await circuit.allow("deepseek") is True

    await circuit.success("deepseek")

    assert await circuit.allow("deepseek") is True


async def test_failed_probe_reopens_for_full_window(circuit, clock):
    for _ in range(5):
        await circuit.failure("deepseek")
    clock.advance(30_001)
    assert await circuit.allow("deepseek") is True

    await circuit.failure("deepseek")

    assert await circuit.allow("deepseek") is False

