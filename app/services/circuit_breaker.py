import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

_ALLOW_SCRIPT = """
local state_key = KEYS[1]
local probe_key = KEYS[2]
local now = tonumber(ARGV[1])
local open_ms = tonumber(ARGV[2])
local state = redis.call('HGET', state_key, 'state')
if state ~= 'open' then return 1 end
local opened_at = tonumber(redis.call('HGET', state_key, 'opened_at') or '0')
if now - opened_at < open_ms then return 0 end
local acquired = redis.call('SET', probe_key, '1', 'NX', 'PX', open_ms)
if acquired then return 1 end
return 0
"""

_SUCCESS_SCRIPT = """
redis.call('HSET', KEYS[1], 'state', 'closed', 'failures', 0, 'opened_at', 0)
redis.call('DEL', KEYS[2])
return 1
"""

_FAILURE_SCRIPT = """
local failures = redis.call('HINCRBY', KEYS[1], 'failures', 1)
if failures >= tonumber(ARGV[1]) then
  redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', ARGV[2])
end
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[3]) * 4)
redis.call('DEL', KEYS[2])
return failures
"""


@dataclass
class _LocalState:
    failures: int = 0
    opened_at: int = 0
    probe_active: bool = False


class RedisCircuitBreaker:
    def __init__(
        self,
        redis_client: Any,
        *,
        failure_threshold: int = 5,
        open_seconds: int = 30,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis_client
        self._threshold = failure_threshold
        self._open_ms = open_seconds * 1000
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._local: dict[str, _LocalState] = {}
        self._local_lock = asyncio.Lock()

    def _keys(self, provider: str) -> tuple[str, str]:
        return (
            f"gateway:circuit:{provider}",
            f"gateway:circuit-probe:{provider}",
        )

    async def allow(self, provider: str) -> bool:
        state_key, probe_key = self._keys(provider)
        try:
            result = await self._redis.eval(
                _ALLOW_SCRIPT,
                2,
                state_key,
                probe_key,
                self._clock_ms(),
                self._open_ms,
            )
            return bool(result)
        except RedisError:
            return await self._local_allow(provider)

    async def success(self, provider: str) -> None:
        state_key, probe_key = self._keys(provider)
        try:
            await self._redis.eval(_SUCCESS_SCRIPT, 2, state_key, probe_key)
        except RedisError:
            await self._local_success(provider)

    async def failure(self, provider: str) -> None:
        state_key, probe_key = self._keys(provider)
        try:
            await self._redis.eval(
                _FAILURE_SCRIPT,
                2,
                state_key,
                probe_key,
                self._threshold,
                self._clock_ms(),
                self._open_ms,
            )
        except RedisError:
            await self._local_failure(provider)

    async def _local_allow(self, provider: str) -> bool:
        async with self._local_lock:
            state = self._local.setdefault(provider, _LocalState())
            if state.failures < self._threshold:
                return True
            if self._clock_ms() - state.opened_at < self._open_ms:
                return False
            if state.probe_active:
                return False
            state.probe_active = True
            return True

    async def _local_success(self, provider: str) -> None:
        async with self._local_lock:
            self._local[provider] = _LocalState()

    async def _local_failure(self, provider: str) -> None:
        async with self._local_lock:
            state = self._local.setdefault(provider, _LocalState())
            state.failures += 1
            state.probe_active = False
            if state.failures >= self._threshold:
                state.opened_at = self._clock_ms()

