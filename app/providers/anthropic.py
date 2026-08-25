import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.providers.base import ProviderChunk, ProviderError
from app.schemas.chat import ChatRequest, ProviderResult, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str, client: httpx.AsyncClient) -> None:
        self._url = f"{base_url.rstrip('/')}/messages"
        self._client = client
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def _payload(self, request: ChatRequest, model: str, stream: bool) -> dict[str, Any]:
        systems = [message.content for message in request.messages if message.role == "system"]
        messages = [
            message.model_dump()
            for message in request.messages
            if message.role != "system"
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if systems:
            payload["system"] = "\n\n".join(systems)
        return payload

    def _http_error(self, status_code: int) -> ProviderError:
        return ProviderError(
            self.name,
            "upstream_http_error",
            status_code,
            status_code == 429 or status_code >= 500,
        )

    async def complete(self, request: ChatRequest, model: str) -> ProviderResult:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json=self._payload(request, model, False),
            )
        except httpx.TimeoutException as error:
            raise ProviderError(self.name, "timeout", None, True) from error
        except httpx.TransportError as error:
            raise ProviderError(self.name, "transport_error", None, True) from error
        if response.status_code >= 400:
            raise self._http_error(response.status_code)
        try:
            payload = response.json()
            content = "".join(
                block["text"]
                for block in payload["content"]
                if block.get("type") == "text"
            )
            if not content:
                raise ValueError("empty content")
            usage_payload = payload.get("usage") or {}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderError(self.name, "invalid_response", response.status_code, True) from error
        input_tokens = usage_payload.get("input_tokens", 0)
        output_tokens = usage_payload.get("output_tokens", 0)
        return ProviderResult(
            model=payload.get("model") or model,
            provider=self.name,
            content=content,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def stream(self, request: ChatRequest, model: str) -> AsyncIterator[ProviderChunk]:
        emitted = False
        input_tokens = 0
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=self._payload(request, model, True),
            ) as response:
                if response.status_code >= 400:
                    raise self._http_error(response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                    except (TypeError, ValueError) as error:
                        raise ProviderError(self.name, "invalid_response", response.status_code, True) from error
                    event_type = payload.get("type")
                    if event_type == "message_start":
                        input_tokens = payload.get("message", {}).get("usage", {}).get("input_tokens", 0)
                    elif event_type == "content_block_delta":
                        text = payload.get("delta", {}).get("text", "")
                        if text:
                            emitted = True
                            yield ProviderChunk(content=text)
                    elif event_type == "message_delta":
                        usage_payload = payload.get("usage", {})
                        input_tokens = usage_payload.get("input_tokens", input_tokens)
                        output_tokens = usage_payload.get("output_tokens", 0)
                        yield ProviderChunk(
                            usage=Usage(
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                            )
                        )
        except ProviderError:
            raise
        except httpx.TimeoutException as error:
            raise ProviderError(self.name, "timeout", None, True) from error
        except httpx.TransportError as error:
            raise ProviderError(self.name, "transport_error", None, True) from error
        if not emitted:
            raise ProviderError(self.name, "invalid_response", 200, True)
