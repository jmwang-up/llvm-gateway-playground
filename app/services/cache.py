import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from redis.exceptions import RedisError

from app.schemas.chat import ChatRequest, ChatResponse

_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def build_cache_key(client: str, request: ChatRequest) -> str:
    normalized = {
        "schema_version": 1,
        "client_identity": client,
        "requested_model": request.model,
        "messages": [message.model_dump() for message in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"gateway:cache:v1:{digest}"


@dataclass(frozen=True)
class SingleFlightResult:
    owner: bool
    cached_response: ChatResponse | None = None


class RedisChatCache:
    def __init__(
        self,
        redis_client: Any,
        *,
        ttl_seconds: int = 300,
        lock_seconds: int = 30,
        lock_wait_seconds: float = 5,
        poll_seconds: float = 0.05,
    ) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._lock_seconds = lock_seconds
        self._lock_wait_seconds = lock_wait_seconds
        self._poll_seconds = poll_seconds

    async def get(self, client: str, request: ChatRequest) -> ChatResponse | None:
        try:
            value = await self._redis.get(build_cache_key(client, request))
        except RedisError:
            return None
        if value is None:
            return None
        try:
            return ChatResponse.model_validate_json(value)
        except (ValidationError, ValueError, TypeError):
            return None

    async def set(
        self,
        client: str,
        request: ChatRequest,
        response: ChatResponse,
    ) -> bool:
        try:
            result = await self._redis.set(
                build_cache_key(client, request),
                response.model_dump_json(),
                ex=self._ttl_seconds,
            )
        except RedisError:
            return False
        return bool(result)

    @asynccontextmanager
    async def single_flight(
        self,
        client: str,
        request: ChatRequest,
    ) -> AsyncIterator[SingleFlightResult]:
        cache_key = build_cache_key(client, request)
        lock_key = cache_key.replace("gateway:cache:v1:", "gateway:cache-lock:v1:")
        token = secrets.token_hex(16)
        try:
            acquired = await self._redis.set(
                lock_key,
                token,
                nx=True,
                px=self._lock_seconds * 1000,
            )
        except RedisError:
            yield SingleFlightResult(owner=True)
            return

        if acquired:
            try:
                yield SingleFlightResult(owner=True)
            finally:
                try:
                    await self._redis.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, token)
                except RedisError:
                    pass
            return

        deadline = time.monotonic() + self._lock_wait_seconds
        while True:
            cached = await self.get(client, request)
            if cached is not None:
                yield SingleFlightResult(owner=False, cached_response=cached)
                return
            if time.monotonic() >= deadline:
                yield SingleFlightResult(owner=False)
                return
            await asyncio.sleep(self._poll_seconds)
