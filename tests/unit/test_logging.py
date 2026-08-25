import json
import logging

from app.observability.logging import configure_logging, log_request


def test_request_log_uses_allowlist_and_excludes_secrets(caplog):
    configure_logging()
    caplog.set_level(logging.INFO, logger="gateway.request")

    log_request(
        request_id="req-1",
        client_identity="frontend",
        requested_model="auto",
        provider="deepseek",
        actual_model="deepseek-chat",
        duration_seconds=0.12,
        cache_status="MISS",
        fallback_count=0,
        error_code=None,
    )

    payload = json.loads(caplog.records[-1].getMessage())
    assert payload["request_id"] == "req-1"
    assert payload["client_identity"] == "frontend"
    assert payload["provider"] == "deepseek"
    serialized = json.dumps(payload)
    assert "X-API-Key" not in serialized
    assert "messages" not in serialized
    assert "content" not in serialized
