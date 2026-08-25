from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.core.auth import APIKeyAuthenticator
from app.core.config import Settings, get_settings
from app.core.errors import GatewayError


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    development = resolved_settings.environment == "development"
    application = FastAPI(
        title="AI Gateway",
        docs_url="/docs" if development else None,
        redoc_url=None,
    )
    application.state.settings = resolved_settings
    application.state.authenticator = APIKeyAuthenticator(resolved_settings.client_keys)
    application.include_router(chat_router)

    @application.exception_handler(GatewayError)
    async def handle_gateway_error(request: Request, error: GatewayError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex}")
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.retryable,
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id, **error.headers},
        )

    return application


app = create_app()
