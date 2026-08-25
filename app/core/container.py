from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderAdapter
from app.providers.deepseek import DeepSeekProvider
from app.providers.openai import OpenAIProvider
from app.services.cache import RedisChatCache
from app.services.circuit_breaker import RedisCircuitBreaker
from app.services.fallback import FallbackExecutor
from app.services.gateway import GatewayService
from app.services.rate_limiter import RedisRateLimiter
from app.services.router import ModelRouter


@dataclass(frozen=True)
class ServiceContainer:
    gateway: GatewayService
    providers: dict[str, ProviderAdapter]

    @property
    def provider_count(self) -> int:
        return len(self.providers)


def build_services(
    settings: Settings,
    *,
    redis_client: Any,
    http_client: httpx.AsyncClient,
    metrics: Any | None = None,
) -> ServiceContainer:
    providers: dict[str, ProviderAdapter] = {}
    if settings.deepseek_api_key:
        providers["deepseek"] = DeepSeekProvider(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            http_client,
        )
    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            settings.openai_api_key,
            settings.openai_base_url,
            http_client,
        )
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            settings.anthropic_api_key,
            settings.anthropic_base_url,
            http_client,
        )

    router = ModelRouter(
        default_models={
            "deepseek": settings.deepseek_default_model if "deepseek" in providers else "",
            "openai": settings.openai_default_model if "openai" in providers else "",
            "anthropic": settings.anthropic_default_model if "anthropic" in providers else "",
        },
        equivalents=settings.model_equivalents,
    )
    rate_limiter = RedisRateLimiter(
        redis_client,
        capacity=settings.rate_limit_capacity,
        refill_per_second=settings.rate_limit_refill_per_second,
        max_concurrent=settings.max_concurrent_requests,
        lease_seconds=settings.concurrency_lease_seconds,
    )
    cache = RedisChatCache(
        redis_client,
        ttl_seconds=settings.cache_ttl_seconds,
        lock_seconds=settings.cache_lock_seconds,
        lock_wait_seconds=settings.cache_lock_wait_seconds,
    )
    circuit = RedisCircuitBreaker(
        redis_client,
        failure_threshold=settings.circuit_failure_threshold,
        open_seconds=settings.circuit_open_seconds,
        metrics=metrics,
    )
    fallback = FallbackExecutor(providers, circuit, metrics=metrics)
    gateway = GatewayService(
        settings=settings,
        rate_limiter=rate_limiter,
        cache=cache,
        router=router,
        fallback=fallback,
        metrics=metrics,
    )
    return ServiceContainer(gateway=gateway, providers=providers)
