import httpx

from app.core.config import Settings
from app.core.container import build_services
from app.providers.anthropic import AnthropicProvider
from app.providers.deepseek import DeepSeekProvider
from app.providers.openai import OpenAIProvider


async def test_container_registers_all_configured_providers():
    settings = Settings(
        client_api_keys="frontend:key-a",
        deepseek_api_key="ds",
        openai_api_key="oa",
        anthropic_api_key="an",
    )
    async with httpx.AsyncClient() as http_client:
        services = build_services(
            settings,
            redis_client=object(),
            http_client=http_client,
        )

    assert isinstance(services.providers["deepseek"], DeepSeekProvider)
    assert isinstance(services.providers["openai"], OpenAIProvider)
    assert isinstance(services.providers["anthropic"], AnthropicProvider)
    assert services.provider_count == 3
    assert services.gateway is not None


async def test_container_omits_unconfigured_providers():
    settings = Settings(
        client_api_keys="frontend:key-a",
        deepseek_api_key="ds",
        openai_api_key="",
        anthropic_api_key="",
    )
    async with httpx.AsyncClient() as http_client:
        services = build_services(
            settings,
            redis_client=object(),
            http_client=http_client,
        )

    assert list(services.providers) == ["deepseek"]
    assert services.provider_count == 1
