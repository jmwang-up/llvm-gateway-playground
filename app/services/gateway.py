import asyncio
from collections.abc import AsyncIterator
from typing import Any

from app.core.auth import ClientIdentity
from app.core.config import Settings
from app.core.errors import GatewayError
from app.providers.base import ProviderError
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, StreamEvent, Usage
from app.services.rate_limiter import RedisRateLimiter


class GatewayService:
    def __init__(
        self,
        *,
        settings: Settings,
        rate_limiter: Any,
        cache: Any,
        router: Any,
        fallback: Any,
    ) -> None:
        self._settings = settings
        self._rate_limiter = rate_limiter
        self._cache = cache
        self._router = router
        self._fallback = fallback

    async def complete(
        self,
        client: ClientIdentity,
        request: ChatRequest,
        request_id: str,
    ) -> ChatResponse:
        decision = await self._rate_limiter.acquire_rate(client.name)
        if not decision.allowed:
            raise RedisRateLimiter.rate_limit_error(decision)
        lease = await self._rate_limiter.acquire_concurrency(client.name, request_id)
        try:
            cached = await self._cache.get(client.name, request)
            if cached is not None:
                return cached.model_copy(update={"cached": True})
            async with self._cache.single_flight(client.name, request) as flight:
                if not flight.owner and flight.cached_response is not None:
                    return flight.cached_response.model_copy(update={"cached": True})
                try:
                    result, fallback_count = await asyncio.wait_for(
                        self._fallback.complete(
                            request,
                            self._router.candidates(request.model),
                        ),
                        timeout=self._settings.total_timeout_seconds,
                    )
                except TimeoutError as error:
                    raise GatewayError(
                        code="gateway_timeout",
                        message="The model request exceeded its total time budget",
                        status_code=504,
                        retryable=True,
                    ) from error
                response = ChatResponse(
                    id=f"chat_{request_id}",
                    model=result.model,
                    provider=result.provider,
                    message=ChatMessage(role="assistant", content=result.content),
                    usage=result.usage,
                    cached=False,
                    fallback_count=fallback_count,
                )
                await self._cache.set(client.name, request, response)
                return response
        finally:
            await self._rate_limiter.release_concurrency(lease)

    async def stream(
        self,
        client: ClientIdentity,
        request: ChatRequest,
        request_id: str,
    ) -> AsyncIterator[StreamEvent]:
        decision = await self._rate_limiter.acquire_rate(client.name)
        if not decision.allowed:
            raise RedisRateLimiter.rate_limit_error(decision)
        lease = await self._rate_limiter.acquire_concurrency(client.name, request_id)
        try:
            candidate, chunks, fallback_count = (
                await self._fallback.stream_until_first_chunk(
                    request,
                    self._router.candidates(request.model),
                )
            )
        except BaseException:
            await self._rate_limiter.release_concurrency(lease)
            raise

        async def generate() -> AsyncIterator[StreamEvent]:
            usage = Usage()
            try:
                yield StreamEvent(
                    event="meta",
                    data={
                        "id": f"chat_{request_id}",
                        "model": f"{candidate.provider}/{candidate.model}",
                        "provider": candidate.provider,
                        "fallback_count": fallback_count,
                    },
                )
                try:
                    async for chunk in chunks:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.content:
                            yield StreamEvent(
                                event="delta",
                                data={"content": chunk.content},
                            )
                except ProviderError:
                    yield StreamEvent(
                        event="error",
                        data={
                            "code": "upstream_stream_error",
                            "message": "The upstream stream ended unexpectedly",
                            "retryable": True,
                            "request_id": request_id,
                        },
                    )
                    return
                yield StreamEvent(
                    event="done",
                    data={"usage": usage.model_dump(mode="json")},
                )
            finally:
                close = getattr(chunks, "aclose", None)
                if close is not None:
                    await close()
                try:
                    await self._rate_limiter.release_concurrency(lease)
                except GatewayError:
                    pass

        return generate()
