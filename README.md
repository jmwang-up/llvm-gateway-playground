# AI Gateway Playground

A runnable FastAPI gateway exposing one `/chat` API for DeepSeek, OpenAI, and Anthropic. It supports JSON and SSE responses, per-client authentication and limits, exact Redis caching, ordered fallback, circuit breaking, health checks, structured logs, and Prometheus metrics.

## Requirements

- Docker with Compose v2, or Python 3.12 plus Redis 7 for local development.
- At least one provider API key for real requests.

## Quick start

```bash
cp .env.example .env
```

Edit `.env`. Client keys use `identity:key` pairs:

```dotenv
CLIENT_API_KEYS=frontend:replace-me,worker:replace-me-too
DEEPSEEK_API_KEY=your-deepseek-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

Only providers with a configured API key are registered. Start the stack:

```bash
docker compose up --build
```

OpenAPI documentation is available at `http://localhost:8000/docs` in the development environment.

## Call `/chat`

Non-streaming:

```bash
curl --fail http://localhost:8000/chat \
  -H 'X-API-Key: replace-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is an AI gateway?"}],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": false
  }'
```

Streaming SSE:

```bash
curl --no-buffer --fail http://localhost:8000/chat \
  -H 'X-API-Key: replace-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain an AI gateway."}],
    "stream": true
  }'
```

`model: auto` uses strict order: DeepSeek, OpenAI, Anthropic. Explicit models use `provider/model`, for example `openai/gpt-5-mini`. Cross-provider equivalents must be explicitly declared in `MODEL_EQUIVALENTS_JSON`.

## Runtime behavior

- Authentication: multiple named keys from `CLIENT_API_KEYS`.
- Rate limit: 60-token capacity and 1 token/second refill per client by default.
- Concurrency: 5 active requests per client; a stream holds its lease until closed.
- Cache: successful non-streaming requests are cached for 300 seconds and isolated per client.
- Fallback: retryable connection, timeout, 429, 5xx, malformed, and empty-response failures move to the next configured provider.
- Circuit breaker: opens for 30 seconds after 5 consecutive retryable failures.
- Streaming fallback: allowed only before the first content delta. Mid-stream failures produce an SSE `error` event.
- Redis failure: `/chat` fails closed with 503 because distributed rate protection is unavailable.

Prompts and model responses for successful non-streaming calls are stored in Redis for five minutes by default. Adjust `CACHE_TTL_SECONDS` according to your privacy policy.

## Operations

```text
GET /health   process liveness
GET /ready    Redis plus at least one configured provider
GET /metrics  Prometheus exposition
GET /docs     development-only OpenAPI UI
```

Logs are JSON and contain request IDs, named client identity, routing metadata, duration, cache status, fallback count, and normalized errors. Raw API keys, prompts, and model output are not logged. Metrics never use client keys, request IDs, or prompts as labels.

## Tests

Install dependencies with `uv`:

```bash
uv venv .venv --python python3
NO_PROXY=files.pythonhosted.org no_proxy=files.pythonhosted.org \
  UV_CACHE_DIR=/tmp/llm-gateway-uv-cache \
  uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -v
```

Run the no-cost Docker smoke test against a local mock Provider:

```bash
bash tests/smoke/run.sh
```

Real-provider manual tests may incur API charges. Never commit `.env` or real keys.
