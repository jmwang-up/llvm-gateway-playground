import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import ProviderChunk, ProviderError
from app.schemas.chat import ChatRequest, ProviderResult, Usage


class OpenAICompatibleProvider:
    name: str

    def __init__(
        self,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient,
    ) -> None:
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._client = client

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _payload(self, request: ChatRequest, model: str, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _http_error(self, status_code: int) -> ProviderError:
        return ProviderError(
            provider=self.name,
            code="upstream_http_error",
            status_code=status_code,
            retryable=status_code == 429 or status_code >= 500,
        )

    def _transport_error(self, code: str = "transport_error") -> ProviderError:
        return ProviderError(self.name, code, None, True)

    async def complete(self, request: ChatRequest, model: str) -> ProviderResult:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json=self._payload(request, model, stream=False),
            )
        except httpx.TimeoutException as error:
            raise self._transport_error("timeout") from error
        except httpx.TransportError as error:
            raise self._transport_error() from error
        if response.status_code >= 400:
            raise self._http_error(response.status_code)
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            actual_model = payload.get("model") or model
            usage_payload = payload.get("usage") or {}
            if not isinstance(content, str) or not content:
                raise ValueError("empty content")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderError(self.name, "invalid_response", response.status_code, True) from error
        usage = Usage(
            prompt_tokens=usage_payload.get("prompt_tokens", 0),
            completion_tokens=usage_payload.get("completion_tokens", 0),
            total_tokens=usage_payload.get("total_tokens", 0),
        )
        return ProviderResult(
            model=actual_model,
            provider=self.name,
            content=content,
            usage=usage,
        )

    async def stream(self, request: ChatRequest, model: str) -> AsyncIterator[ProviderChunk]:
        emitted = False
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=self._payload(request, model, stream=True),
            ) as response:
                if response.status_code >= 400:
                    raise self._http_error(response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                        content = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        usage_payload = payload.get("usage")
                    except (TypeError, ValueError, IndexError) as error:
                        raise ProviderError(self.name, "invalid_response", response.status_code, True) from error
                    usage = None
                    if usage_payload:
                        usage = Usage(
                            prompt_tokens=usage_payload.get("prompt_tokens", 0),
                            completion_tokens=usage_payload.get("completion_tokens", 0),
                            total_tokens=usage_payload.get("total_tokens", 0),
                        )
                    if content:
                        emitted = True
                        yield ProviderChunk(content=content, usage=usage)
                    elif usage is not None:
                        yield ProviderChunk(usage=usage)
        except ProviderError:
            raise
        except httpx.TimeoutException as error:
            raise self._transport_error("timeout") from error
        except httpx.TransportError as error:
            raise self._transport_error() from error
        if not emitted:
            raise ProviderError(self.name, "invalid_response", 200, True)

