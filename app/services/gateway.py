import asyncio
from typing import Any

from app.core.auth import ClientIdentity
from app.core.config import Settings
from app.core.errors import GatewayError
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
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

