import pytest

from app.core.config import Settings


def test_client_keys_are_parsed_to_named_identities():
    settings = Settings(
        client_api_keys="frontend:key-a,worker:key-b",
        deepseek_api_key="ds",
    )

    assert settings.client_keys == {"key-a": "frontend", "key-b": "worker"}
    assert settings.provider_order == ("deepseek", "openai", "anthropic")


def test_duplicate_client_keys_are_rejected():
    with pytest.raises(ValueError):
        Settings(client_api_keys="frontend:key-a,worker:key-a")


def test_malformed_client_keys_are_rejected():
    with pytest.raises(ValueError):
        Settings(client_api_keys="missing-separator")

