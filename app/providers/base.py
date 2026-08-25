from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from app.schemas.chat import ChatRequest, ProviderResult, Usage


@dataclass(frozen=True)
class ProviderChunk:
    content: str = ""
    usage: Usage | None = None


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        code: str,
        status_code: int | None,
        retryable: bool,
    ) -> None:
        super().__init__(code)
        self.provider = provider
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class ProviderAdapter(Protocol):
    name: str

    async def complete(self, request: ChatRequest, model: str) -> ProviderResult: ...

    def stream(self, request: ChatRequest, model: str) -> AsyncIterator[ProviderChunk]: ...

