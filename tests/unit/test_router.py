import pytest

from app.core.errors import GatewayError
from app.services.router import ModelRouter, RouteCandidate


@pytest.fixture
def router():
    return ModelRouter(
        default_models={
            "deepseek": "deepseek-chat",
            "openai": "gpt-x",
            "anthropic": "claude-x",
        },
        equivalents={
            "openai/gpt-x": ["anthropic/claude-x"],
        },
    )


def test_auto_route_has_strict_provider_order(router):
    assert router.candidates("auto") == [
        RouteCandidate("deepseek", "deepseek-chat"),
        RouteCandidate("openai", "gpt-x"),
        RouteCandidate("anthropic", "claude-x"),
    ]


def test_explicit_model_uses_only_configured_equivalents(router):
    assert router.candidates("openai/gpt-x") == [
        RouteCandidate("openai", "gpt-x"),
        RouteCandidate("anthropic", "claude-x"),
    ]


def test_unknown_model_is_rejected(router):
    with pytest.raises(GatewayError) as error:
        router.candidates("invented-model")

    assert error.value.code == "unknown_model"
    assert error.value.status_code == 400

