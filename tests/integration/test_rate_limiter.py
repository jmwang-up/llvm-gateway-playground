import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from app.core.errors import GatewayError
from app.services.rate_limiter import RedisRateLimiter


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
def limiter(redis_client, clock):
    return RedisRateLimiter(
        redis_client,
        capacity=60,
        refill_per_second=1,
        max_concurrent=5,
        lease_seconds=30,
        clock_ms=clock,
    )


async def test_token_bucket_allows_capacity_then_rejects(limiter):
    decisions = [await limiter.acquire_rate("frontend") for _ in range(61)]

    assert all(decision.allowed for decision in decisions[:60])
    assert decisions[59].remaining == 0
    assert decisions[60].allowed is False
    assert decisions[60].retry_after == 1


async def test_token_bucket_refills_and_isolates_clients(limiter, clock):
    for _ in range(60):
        await limiter.acquire_rate("frontend")
    clock.advance(2_000)

    assert (await limiter.acquire_rate("frontend")).allowed is True
    assert (await limiter.acquire_rate("frontend")).allowed is True
    assert (await limiter.acquire_rate("frontend")).allowed is False
    assert (await limiter.acquire_rate("worker")).remaining == 59


async def test_concurrency_is_released_and_isolated(limiter):
    leases = [
        await limiter.acquire_concurrency("frontend", f"req-{index}")
        for index in range(5)
    ]

    with pytest.raises(GatewayError) as error:
        await limiter.acquire_concurrency("frontend", "req-5")
    assert error.value.status_code == 429
    assert await limiter.acquire_concurrency("worker", "req-worker")

    await limiter.release_concurrency(leases[0])
    assert await limiter.acquire_concurrency("frontend", "req-6")


async def test_expired_concurrency_lease_is_reclaimed(limiter, clock):
    for index in range(5):
        await limiter.acquire_concurrency("frontend", f"req-{index}")
    clock.advance(30_001)

    lease = await limiter.acquire_concurrency("frontend", "req-new")

    assert lease.request_id == "req-new"


async def test_redis_failure_is_fail_closed(clock):
    class BrokenRedis:
        async def eval(self, *args, **kwargs):
            raise RedisError("offline")

    limiter = RedisRateLimiter(BrokenRedis(), clock_ms=clock)

    with pytest.raises(GatewayError) as error:
        await limiter.acquire_rate("frontend")

    assert error.value.code == "redis_unavailable"
    assert error.value.status_code == 503
