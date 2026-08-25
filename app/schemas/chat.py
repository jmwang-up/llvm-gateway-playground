from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default="auto", min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ProviderResult(BaseModel):
    model: str
    provider: str
    content: str
    usage: Usage


class ChatResponse(BaseModel):
    id: str
    model: str
    provider: str
    message: ChatMessage
    usage: Usage
    cached: bool = False
    fallback_count: int = Field(default=0, ge=0)

