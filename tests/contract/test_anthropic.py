import json

import httpx
import pytest
import respx

from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.schemas.chat import ChatRequest


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.2,
        max_tokens=42,
    )


async def test_anthropic_complete_maps_system_and_response(chat_request):
    async with httpx.AsyncClient() as client:
        provider = AnthropicProvider("secret", "https://provider.test/v1", client)
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://provider.test/v1/messages").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "model": "claude-actual",
                        "content": [{"type": "text", "text": "Hi there"}],
                        "usage": {"input_tokens": 4, "output_tokens": 2},
                    },
                )
            )
            result = await provider.complete(chat_request, "claude-requested")

    payload = json.loads(route.calls[0].request.content)
    assert payload == {
        "model": "claude-requested",
        "system": "Be concise",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 42,
        "stream": False,
    }
    assert route.calls[0].request.headers["x-api-key"] == "secret"
    assert result.provider == "anthropic"
    assert result.model == "claude-actual"
    assert result.content == "Hi there"
    assert result.usage.total_tokens == 6


async def test_anthropic_stream_maps_events_and_usage(chat_request):
    body = "\n".join(
        [
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            "",
            'event: message_delta',
            'data: {"type":"message_delta","usage":{"input_tokens":4,"output_tokens":2}}',
            "",
        ]
    )
    async with httpx.AsyncClient() as client:
        provider = AnthropicProvider("secret", "https://provider.test/v1", client)
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://provider.test/v1/messages").mock(
                return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            )
            chunks = [chunk async for chunk in provider.stream(chat_request, "claude-requested")]

    payload = json.loads(route.calls[0].request.content)
    assert payload["stream"] is True
    assert [chunk.content for chunk in chunks] == ["Hi", ""]
    assert chunks[-1].usage.total_tokens == 6


@pytest.mark.parametrize("status,retryable", [(400, False), (401, False), (429, True), (503, True)])
async def test_anthropic_normalizes_http_errors(status, retryable, chat_request):
    async with httpx.AsyncClient() as client:
        provider = AnthropicProvider("secret", "https://provider.test/v1", client)
        with respx.mock:
            respx.post("https://provider.test/v1/messages").mock(return_value=httpx.Response(status))
            with pytest.raises(ProviderError) as error:
                await provider.complete(chat_request, "claude-requested")

    assert error.value.status_code == status
    assert error.value.retryable is retryable
