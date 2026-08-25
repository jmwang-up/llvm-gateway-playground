import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest


def test_chat_request_accepts_text_roles():
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])

    assert request.model == "auto"
    assert request.stream is False


def test_chat_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_request_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "tool", "content": "x"}])
