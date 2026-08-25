import re
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.auth import ClientIdentity, get_client_identity
from app.core.errors import GatewayError
from app.schemas.chat import ChatRequest

router = APIRouter()
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _request_id(raw_value: str | None) -> str:
    if raw_value and _REQUEST_ID.fullmatch(raw_value):
        return raw_value
    return f"req_{uuid4().hex}"


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    client: ClientIdentity = Depends(get_client_identity),
):
    request_id = _request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    if payload.stream:
        raise GatewayError(
            code="streaming_not_ready",
            message="Streaming support is not enabled yet",
            status_code=501,
            retryable=False,
        )
    response = await request.app.state.gateway.complete(client, payload, request_id)
    return JSONResponse(
        content=response.model_dump(mode="json"),
        headers={
            "X-Request-ID": request_id,
            "X-Cache": "HIT" if response.cached else "MISS",
        },
    )
