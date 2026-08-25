import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, Usage
from app.services.cache import RedisChatCache


@pytest.fixture
async def redis_client():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def chat_request():
    return ChatRequest(messages=[{"role": "user", "content": "hello"}])


@pytest.fixture
def chat_response():
    return ChatResponse(
        id="chat_req-1",
        model="deepseek/deepseek-chat",
        provider="deepseek",
        message=ChatMessage(role="assistant", content="hi"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


async def test_cache_round_trip_ttl_and_client_isolation(
    redis_client,
    chat_request,
    chat_response,
):
    cache = RedisChatCache(redis_client, ttl_seconds=300)

    assert await cache.set("frontend", chat_request, chat_response) is True
    assert await cache.get("frontend", chat_request) == chat_response
    assert await cache.get("worker", chat_request) is None
    keys = await redis_client.keys("gateway:cache:v1:*")
    assert await redis_client.ttl(keys[0]) == 300


async def test_single_flight_has_one_owner(redis_client, chat_request):
    cache = RedisChatCache(redis_client, lock_wait_seconds=0)

    async with cache.single_flight("frontend", chat_request) as first:
        async with cache.single_flight("frontend", chat_request) as second:
            assert first.owner is True
            assert second.owner is False
            assert second.cached_response is None


async def test_single_flight_waiter_reads_result(
    redis_client,
    chat_request,
    chat_response,
):
    cache = RedisChatCache(redis_client, lock_wait_seconds=0.1, poll_seconds=0.01)

    async with cache.single_flight("frontend", chat_request) as first:
        assert first.owner is True
        await cache.set("frontend", chat_request, chat_response)
        async with cache.single_flight("frontend", chat_request) as second:
            assert second.owner is False
            assert second.cached_response == chat_response


async def test_cache_redis_failures_are_fail_open(chat_request, chat_response):
    class BrokenRedis:
        async def get(self, *args, **kwargs):
            raise RedisError("offline")

        async def set(self, *args, **kwargs):
            raise RedisError("offline")

    cache = RedisChatCache(BrokenRedis())

    assert await cache.get("frontend", chat_request) is None
    assert await cache.set("frontend", chat_request, chat_response) is False
    async with cache.single_flight("frontend", chat_request) as flight:
        assert flight.owner is True
