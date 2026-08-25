from dataclasses import dataclass

from app.core.errors import GatewayError


@dataclass(frozen=True)
class RouteCandidate:
    provider: str
    model: str


class ModelRouter:
    _provider_order = ("deepseek", "openai", "anthropic")

    def __init__(
        self,
        default_models: dict[str, str],
        equivalents: dict[str, list[str]] | None = None,
    ) -> None:
        self._default_models = default_models
        self._equivalents = equivalents or {}

    def candidates(self, requested_model: str) -> list[RouteCandidate]:
        if requested_model == "auto":
            candidates = [
                RouteCandidate(provider, self._default_models[provider])
                for provider in self._provider_order
                if self._default_models.get(provider)
            ]
            if candidates:
                return candidates
        elif "/" in requested_model:
            configured = [requested_model, *self._equivalents.get(requested_model, [])]
            candidates = [self._parse_explicit(value) for value in configured]
            if all(candidate.provider in self._provider_order for candidate in candidates):
                return candidates
        raise GatewayError(
            code="unknown_model",
            message="The requested model is not configured",
            status_code=400,
            retryable=False,
        )

    @staticmethod
    def _parse_explicit(value: str) -> RouteCandidate:
        provider, model = value.split("/", 1)
        if not provider or not model:
            raise GatewayError(
                code="unknown_model",
                message="The requested model is not configured",
                status_code=400,
                retryable=False,
            )
        return RouteCandidate(provider, model)

