import json
import logging

from pythonjsonlogger.json import JsonFormatter


def configure_logging() -> None:
    root = logging.getLogger()
    if not any(getattr(handler, "_gateway_json", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler._gateway_json = True
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def log_request(
    *,
    request_id: str,
    client_identity: str | None,
    requested_model: str | None,
    provider: str | None,
    actual_model: str | None,
    duration_seconds: float,
    cache_status: str | None,
    fallback_count: int | None,
    error_code: str | None,
) -> None:
    payload = {
        "event": "gateway_request",
        "request_id": request_id,
        "client_identity": client_identity,
        "requested_model": requested_model,
        "provider": provider,
        "actual_model": actual_model,
        "duration_seconds": round(duration_seconds, 6),
        "cache_status": cache_status,
        "fallback_count": fallback_count,
        "error_code": error_code,
    }
    logging.getLogger("gateway.request").info(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )

