import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(payload: dict):
    if payload.get("stream"):
        async def events():
            chunks = [
                {"choices": [{"delta": {"content": "mock "}}]},
                {
                    "choices": [{"delta": {"content": "response"}}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    return {
        "model": payload.get("model", "mock-model"),
        "choices": [{"message": {"content": "mock response"}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }

