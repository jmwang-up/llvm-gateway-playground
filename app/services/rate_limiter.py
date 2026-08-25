import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

from app.core.errors import GatewayError

_TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local refill_per_ms = tonumber(ARGV[3]) / 1000
local requested = 1

local values = redis.call('HMGET', key, 'tokens', 'updated_at')
local tokens = tonumber(values[1])
local updated_at = tonumber(values[2])
if tokens == nil then tokens = capacity end
if updated_at == nil then updated_at = now end

local elapsed = math.max(0, now - updated_at)
tokens = math.min(capacity, tokens + elapsed * refill_per_ms)

local allowed = 0
local retry_ms = 0
if tokens >= requested then
  allowed = 1
  tokens = tokens - requested
else
  retry_ms = math.ceil((requested - tokens) / refill_per_ms)
end

redis.call('HSET', key, 'tokens', tokens, 'updated_at', now)
local ttl_ms = math.ceil((capacity / (refill_per_ms * 1000)) * 2000)
redis.call('PEXPIRE', key, ttl_ms)
return {allowed, tostring(tokens), retry_ms}
"""

_ACQUIRE_CONCURRENCY_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local lease_ms = tonumber(ARGV[2])
local maximum = tonumber(ARGV[3])
local request_id = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - lease_ms)
if redis.call('ZCARD', key) >= maximum then
  return 0
end
redis.call('ZADD', key, now, request_id)
redis.call('PEXPIRE', key, lease_ms * 2)
return 1
"""

_RELEASE_CONCURRENCY_SCRIPT = """
return redis.call('ZREM', KEYS[1], ARGV[1])
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: int


@dataclass(frozen=True)
class ConcurrencyLease:
    client: str
    request_id: str


class RedisRateLimiter:
    def __init__(
        self,
        redis_client: Any,
        *,
        capacity: int = 60,
        refill_per_second: float = 1,
        max_concurrent: int = 5,
        lease_seconds: int = 180,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis_client
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._max_concurrent = max_concurrent
        self._lease_ms = lease_seconds * 1000
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    async def acquire_rate(self, client: str) -> RateLimitDecision:
        try:
            result = await self._redis.eval(
                _TOKEN_BUCKET_SCRIPT,
                1,
                f"gateway:rate:{client}",
                self._clock_ms(),
                self._capacity,
                self._refill_per_second,
            )
        except RedisError as error:
            raise self._redis_unavailable() from error
        retry_ms = int(result[2])
        return RateLimitDecision(
            allowed=bool(result[0]),
            remaining=max(0, math.floor(float(result[1]))),
            retry_after=max(0, math.ceil(retry_ms / 1000)),
        )

    async def acquire_concurrency(
        self,
        client: str,
        request_id: str,
    ) -> ConcurrencyLease:
        try:
            allowed = await self._redis.eval(
                _ACQUIRE_CONCURRENCY_SCRIPT,
                1,
                f"gateway:concurrency:{client}",
                self._clock_ms(),
                self._lease_ms,
                self._max_concurrent,
                request_id,
            )
        except RedisError as error:
            raise self._redis_unavailable() from error
        if not allowed:
            raise GatewayError(
                code="concurrency_limit_exceeded",
                message="Too many concurrent requests",
                status_code=429,
                retryable=True,
            )
        return ConcurrencyLease(client=client, request_id=request_id)

    async def release_concurrency(self, lease: ConcurrencyLease) -> None:
        try:
            await self._redis.eval(
                _RELEASE_CONCURRENCY_SCRIPT,
                1,
                f"gateway:concurrency:{lease.client}",
                lease.request_id,
            )
        except RedisError as error:
            raise self._redis_unavailable() from error

    @staticmethod
    def rate_limit_error(decision: RateLimitDecision) -> GatewayError:
        return GatewayError(
            code="rate_limit_exceeded",
            message="Request rate limit exceeded",
            status_code=429,
            retryable=True,
        )

    @staticmethod
    def _redis_unavailable() -> GatewayError:
        return GatewayError(
            code="redis_unavailable",
            message="Rate limiting is temporarily unavailable",
            status_code=503,
            retryable=True,
        )
