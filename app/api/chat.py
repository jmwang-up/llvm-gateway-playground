import json
import re
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.auth import ClientIdentity, get_client_identity
from app.schemas.chat import ChatRequest, StreamEvent

router = APIRouter()
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _request_id(raw_value: str | None) -> str:
    if raw_value and _REQUEST_ID.fullmatch(raw_value):
        return raw_value
    return f"req_{uuid4().hex}"


def encode_sse(event: StreamEvent) -> bytes:
    data = json.dumps(
        event.data,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event.event}\ndata: {data}\n\n".encode("utf-8")


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    client: ClientIdentity = Depends(get_client_identity),
):
    request_id = _request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    request.state.requested_model = payload.model
    if payload.stream:
        stream = await request.app.state.gateway.stream(client, payload, request_id)
        return StreamingResponse(
            (encode_sse(event) async for event in stream),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    response = await request.app.state.gateway.complete(client, payload, request_id)
    request.state.provider = response.provider
    request.state.actual_model = response.model
    request.state.cache_status = "HIT" if response.cached else "MISS"
    request.state.fallback_count = response.fallback_count
    return JSONResponse(
        content=response.model_dump(mode="json"),
        headers={
            "X-Request-ID": request_id,
            "X-Cache": "HIT" if response.cached else "MISS",
        },
    )
