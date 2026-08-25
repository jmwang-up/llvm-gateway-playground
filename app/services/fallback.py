from collections.abc import AsyncIterator
from typing import Protocol

from app.core.errors import GatewayError
from app.providers.base import ProviderAdapter, ProviderChunk, ProviderError
from app.schemas.chat import ChatRequest, ProviderResult
from app.services.router import RouteCandidate


class CircuitBreaker(Protocol):
    async def allow(self, provider: str) -> bool: ...
    async def success(self, provider: str) -> None: ...
    async def failure(self, provider: str) -> None: ...


class FallbackExecutor:
    def __init__(
        self,
        providers: dict[str, ProviderAdapter],
        circuit: CircuitBreaker,
        metrics=None,
    ) -> None:
        self._providers = providers
        self._circuit = circuit
        self._metrics = metrics

    async def complete(
        self,
        request: ChatRequest,
        candidates: list[RouteCandidate],
    ) -> tuple[ProviderResult, int]:
        failures: list[ProviderError] = []
        fallback_count = 0
        for candidate in candidates:
            if not await self._circuit.allow(candidate.provider):
                fallback_count += 1
                continue
            provider = self._providers.get(candidate.provider)
            if provider is None:
                fallback_count += 1
                continue
            try:
                result = await provider.complete(
                    request.model_copy(update={"stream": False}),
                    candidate.model,
                )
            except ProviderError as error:
                self._provider_metric(candidate.provider, "error", error.code)
                if not error.retryable:
                    raise
                failures.append(error)
                fallback_count += 1
                await self._circuit.failure(candidate.provider)
                self._fallback_metric(candidates, candidate)
                continue
            self._provider_metric(candidate.provider, "success")
            await self._circuit.success(candidate.provider)
            return result, fallback_count
        raise self._unavailable(failures)

    async def stream_until_first_chunk(
        self,
        request: ChatRequest,
        candidates: list[RouteCandidate],
    ) -> tuple[RouteCandidate, AsyncIterator[ProviderChunk], int]:
        failures: list[ProviderError] = []
        fallback_count = 0
        for candidate in candidates:
            if not await self._circuit.allow(candidate.provider):
                fallback_count += 1
                continue
            provider = self._providers.get(candidate.provider)
            if provider is None:
                fallback_count += 1
                continue
            iterator = provider.stream(
                request.model_copy(update={"stream": True}),
                candidate.model,
            ).__aiter__()
            buffered: list[ProviderChunk] = []
            try:
                async for chunk in iterator:
                    buffered.append(chunk)
                    if chunk.content:
                        await self._circuit.success(candidate.provider)
                        self._provider_metric(candidate.provider, "success")
                        return (
                            candidate,
                            self._prepend(buffered, iterator),
                            fallback_count,
                        )
                raise ProviderError(candidate.provider, "invalid_response", 200, True)
            except ProviderError as error:
                self._provider_metric(candidate.provider, "error", error.code)
                if not error.retryable:
                    raise
                failures.append(error)
                fallback_count += 1
                await self._circuit.failure(candidate.provider)
                self._fallback_metric(candidates, candidate)
                await iterator.aclose()
        raise self._unavailable(failures)

    @staticmethod
    async def _prepend(
        buffered: list[ProviderChunk],
        iterator: AsyncIterator[ProviderChunk],
    ) -> AsyncIterator[ProviderChunk]:
        for chunk in buffered:
            yield chunk
        async for chunk in iterator:
            yield chunk

    @staticmethod
    def _unavailable(failures: list[ProviderError]) -> GatewayError:
        all_rate_limited = bool(failures) and all(
            failure.status_code == 429 for failure in failures
        )
        if all_rate_limited:
            return GatewayError(
                code="upstream_rate_limited",
                message="All configured model providers are rate limited",
                status_code=429,
                retryable=True,
            )
        return GatewayError(
            code="upstream_unavailable",
            message="No configured model provider is available",
            status_code=503,
            retryable=True,
        )

    def _provider_metric(
        self,
        provider: str,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        if self._metrics is None:
            return
        self._metrics.provider_requests.labels(provider, outcome).inc()
        if error_code is not None:
            self._metrics.provider_errors.labels(provider, error_code).inc()

    def _fallback_metric(
        self,
        candidates: list[RouteCandidate],
        current: RouteCandidate,
    ) -> None:
        if self._metrics is None:
            return
        index = candidates.index(current)
        if index + 1 < len(candidates):
            self._metrics.fallbacks.labels(
                current.provider,
                candidates[index + 1].provider,
            ).inc()
