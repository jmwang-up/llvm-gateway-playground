import json

import httpx
import pytest
import respx

from app.providers.base import ProviderError
from app.providers.deepseek import DeepSeekProvider
from app.providers.openai import OpenAIProvider
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


@pytest.mark.parametrize(
    ("provider_class", "provider_name"),
    [(OpenAIProvider, "openai"), (DeepSeekProvider, "deepseek")],
)
async def test_openai_compatible_complete_maps_request_and_response(
    provider_class,
    provider_name,
    chat_request,
):
    async with httpx.AsyncClient() as client:
        provider = provider_class("secret", "https://provider.test/v1", client)
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://provider.test/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "model": "actual-model",
                        "choices": [{"message": {"content": "Hi there"}}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    },
                )
            )
            result = await provider.complete(chat_request, "requested-model")

    payload = json.loads(route.calls[0].request.content)
    assert payload == {
        "model": "requested-model",
        "messages": [message.model_dump() for message in chat_request.messages],
        "temperature": 0.2,
        "max_tokens": 42,
        "stream": False,
    }
    assert result.provider == provider_name
    assert result.model == "actual-model"
    assert result.content == "Hi there"
    assert result.usage.total_tokens == 6
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"


async def test_openai_compatible_stream_maps_sse_chunks(chat_request):
    body = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}',
            "data: [DONE]",
            "",
        ]
    )
    async with httpx.AsyncClient() as client:
        provider = OpenAIProvider("secret", "https://provider.test/v1", client)
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://provider.test/v1/chat/completions").mock(
                return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
            )
            chunks = [chunk async for chunk in provider.stream(chat_request, "model-a")]

    payload = json.loads(route.calls[0].request.content)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert [chunk.content for chunk in chunks] == ["Hi", " there"]
    assert chunks[-1].usage.total_tokens == 6


@pytest.mark.parametrize("status,retryable", [(400, False), (401, False), (429, True), (500, True)])
async def test_openai_compatible_normalizes_http_errors(status, retryable, chat_request):
    async with httpx.AsyncClient() as client:
        provider = OpenAIProvider("super-secret", "https://provider.test/v1", client)
        with respx.mock:
            respx.post("https://provider.test/v1/chat/completions").mock(
                return_value=httpx.Response(status, text="sensitive upstream body")
            )
            with pytest.raises(ProviderError) as error:
                await provider.complete(chat_request, "model-a")

    assert error.value.retryable is retryable
    assert error.value.status_code == status
    assert "super-secret" not in str(error.value)
    assert "sensitive" not in str(error.value)


async def test_openai_compatible_rejects_empty_success(chat_request):
    async with httpx.AsyncClient() as client:
        provider = OpenAIProvider("secret", "https://provider.test/v1", client)
        with respx.mock:
            respx.post("https://provider.test/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            )
            with pytest.raises(ProviderError) as error:
                await provider.complete(chat_request, "model-a")

    assert error.value.code == "invalid_response"
    assert error.value.retryable is True
