import asyncio

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.exceptions import RedisError

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, str]:
    redis_client = getattr(request.app.state, "redis", None)
    provider_count = getattr(request.app.state, "provider_count", 0)
    try:
        redis_ready = bool(
            redis_client
            and await asyncio.wait_for(redis_client.ping(), timeout=0.5)
        )
    except (RedisError, TimeoutError):
        redis_ready = False
    if not redis_ready or provider_count < 1:
        response.status_code = 503
        return {"status": "not_ready"}
    return {"status": "ready"}


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    payload = generate_latest(request.app.state.metrics.registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

