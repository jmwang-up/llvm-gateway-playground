from dataclasses import dataclass
from secrets import compare_digest

from fastapi import Request

from app.core.errors import GatewayError


@dataclass(frozen=True)
class ClientIdentity:
    name: str


class APIKeyAuthenticator:
    def __init__(self, client_keys: dict[str, str]) -> None:
        self._client_keys = tuple(client_keys.items())

    def authenticate(self, raw_key: str | None) -> ClientIdentity:
        candidate = raw_key or ""
        matched_identity: str | None = None
        for configured_key, identity in self._client_keys:
            if compare_digest(candidate, configured_key):
                matched_identity = identity
        if matched_identity is None:
            raise GatewayError(
                code="invalid_api_key",
                message="Missing or invalid API key",
                status_code=401,
                retryable=False,
            )
        return ClientIdentity(name=matched_identity)


async def get_client_identity(request: Request) -> ClientIdentity:
    raw_key = request.headers.get("X-API-Key")
    identity = request.app.state.authenticator.authenticate(raw_key)
    request.state.client_identity = identity.name
    return identity
