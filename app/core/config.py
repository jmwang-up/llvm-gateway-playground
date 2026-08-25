import json
from functools import lru_cache
from typing import ClassVar

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider_order: ClassVar[tuple[str, ...]] = (
        "deepseek",
        "openai",
        "anthropic",
    )

    environment: str = "development"
    client_api_keys: str = "demo:change-me"
    redis_url: str = "redis://localhost:6379/0"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_default_model: str = "deepseek-chat"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-5-mini"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_default_model: str = "claude-sonnet-4-20250514"
    model_equivalents_json: str = "{}"

    rate_limit_capacity: int = Field(default=60, gt=0)
    rate_limit_refill_per_second: float = Field(default=1, gt=0)
    max_concurrent_requests: int = Field(default=5, gt=0)
    concurrency_lease_seconds: int = Field(default=180, gt=0)
    cache_ttl_seconds: int = Field(default=300, gt=0)
    cache_lock_seconds: int = Field(default=30, gt=0)
    cache_lock_wait_seconds: float = Field(default=5, ge=0)
    circuit_failure_threshold: int = Field(default=5, gt=0)
    circuit_open_seconds: int = Field(default=30, gt=0)
    provider_timeout_seconds: float = Field(default=30, gt=0)
    total_timeout_seconds: float = Field(default=75, gt=0)

    @model_validator(mode="after")
    def validate_client_api_keys(self) -> "Settings":
        self.client_keys
        return self

    @property
    def client_keys(self) -> dict[str, str]:
        parsed: dict[str, str] = {}
        identities: set[str] = set()
        for entry in self.client_api_keys.split(","):
            if ":" not in entry:
                raise ValueError("CLIENT_API_KEYS entries must use identity:key")
            identity, key = (part.strip() for part in entry.split(":", 1))
            if not identity or not key:
                raise ValueError("CLIENT_API_KEYS identities and keys cannot be blank")
            if identity in identities or key in parsed:
                raise ValueError("CLIENT_API_KEYS identities and keys must be unique")
            identities.add(identity)
            parsed[key] = identity
        return parsed

    @property
    def model_equivalents(self) -> dict[str, list[str]]:
        try:
            parsed = json.loads(self.model_equivalents_json)
        except json.JSONDecodeError as error:
            raise ValueError("MODEL_EQUIVALENTS_JSON must be valid JSON") from error
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str)
            and isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            for key, value in parsed.items()
        ):
            raise ValueError("MODEL_EQUIVALENTS_JSON must map model names to lists")
        return parsed


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.client_keys
    return settings
