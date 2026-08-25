import pytest

from app.core.auth import APIKeyAuthenticator
from app.core.errors import GatewayError


def test_authenticator_returns_named_identity():
    authenticator = APIKeyAuthenticator({"secret-a": "frontend"})

    assert authenticator.authenticate("secret-a").name == "frontend"


@pytest.mark.parametrize("raw_key", [None, "wrong"])
def test_authenticator_rejects_missing_and_invalid_keys(raw_key):
    authenticator = APIKeyAuthenticator({"secret-a": "frontend"})

    with pytest.raises(GatewayError) as error:
        authenticator.authenticate(raw_key)

    assert error.value.code == "invalid_api_key"
    assert error.value.status_code == 401
    assert error.value.retryable is False
