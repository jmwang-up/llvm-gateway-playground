import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.auth import APIKeyAuthenticator
from app.core.config import Settings, get_settings
from app.core.container import build_services
from app.core.errors import GatewayError
from app.observability.logging import configure_logging, log_request
from app.observability.metrics import GatewayMetrics


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    development = resolved_settings.environment == "development"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging()
        redis_client = Redis.from_url(
            resolved_settings.redis_url,
            decode_responses=True,
        )
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                resolved_settings.provider_timeout_seconds,
                connect=min(5, resolved_settings.provider_timeout_seconds),
            )
        )
        services = build_services(
            resolved_settings,
            redis_client=redis_client,
            http_client=http_client,
            metrics=application.state.metrics,
        )
        application.state.redis = redis_client
        application.state.http_client = http_client
        application.state.gateway = services.gateway
        application.state.provider_count = services.provider_count
        try:
            yield
        finally:
            await http_client.aclose()
            await redis_client.aclose()

    application = FastAPI(
        title="AI Gateway",
        docs_url="/docs" if development else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.authenticator = APIKeyAuthenticator(resolved_settings.client_keys)
    application.state.metrics = GatewayMetrics()
    application.state.provider_count = 0
    application.include_router(chat_router)
    application.include_router(health_router)

    @application.middleware("http")
    async def observe_requests(request: Request, call_next):
        started = time.perf_counter()
        status = "500"
        application.state.metrics.active_requests.inc()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            duration = time.perf_counter() - started
            application.state.metrics.active_requests.dec()
            application.state.metrics.requests.labels(
                request.method,
                request.url.path,
                status,
            ).inc()
            application.state.metrics.request_duration.labels(
                request.method,
                request.url.path,
            ).observe(duration)
            log_request(
                request_id=getattr(request.state, "request_id", f"req_{uuid4().hex}"),
                client_identity=getattr(request.state, "client_identity", None),
                requested_model=getattr(request.state, "requested_model", None),
                provider=getattr(request.state, "provider", None),
                actual_model=getattr(request.state, "actual_model", None),
                duration_seconds=duration,
                cache_status=getattr(request.state, "cache_status", None),
                fallback_count=getattr(request.state, "fallback_count", None),
                error_code=getattr(request.state, "error_code", None),
            )

    @application.exception_handler(GatewayError)
    async def handle_gateway_error(request: Request, error: GatewayError) -> JSONResponse:
        request.state.error_code = error.code
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex}")
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.retryable,
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id, **error.headers},
        )

    return application


app = create_app()
