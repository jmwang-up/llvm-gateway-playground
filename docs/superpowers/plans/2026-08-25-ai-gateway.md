# AI Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable FastAPI AI gateway with one `/chat` API, DeepSeek/OpenAI/Anthropic adapters, per-client limits and cache, ordered fallback, circuit breaking, SSE streaming, observability, and Docker Compose startup.

**Architecture:** HTTP requests are normalized into provider-independent models and orchestrated by `GatewayService`. Redis-backed services implement authentication-adjacent client isolation, token-bucket/concurrency limits, exact caching, and shared circuit state; provider adapters isolate upstream HTTP and streaming protocols.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic Settings, HTTPX, redis-py asyncio, Prometheus Client, pytest, pytest-asyncio, Docker Compose.

## Global Constraints

- Support Python 3.12.
- Expose `POST /chat`, `/health`, `/ready`, `/metrics`, and development-only `/docs`.
- Support non-streaming JSON and normalized `text/event-stream` output.
- Configure multiple client API keys through environment variables; never log raw client or provider keys.
- Route `auto` in strict DeepSeek, OpenAI, Anthropic order.
- Default per-client limits are 60 token-bucket capacity, 1 token/second refill, and 5 concurrent requests.
- Cache only successful non-streaming calls for 300 seconds by default; isolate cache by client identity.
- Permit streaming fallback only before the first `delta` is emitted.
- Open a provider circuit after 5 consecutive retryable failures for 30 seconds.
- Use a 30-second provider timeout and a 75-second non-streaming total budget by default.
- Treat Redis failure as fail-closed for `/chat` rate limiting and fail-open inside the cache abstraction.
- Do not implement a database, admin UI, semantic cache, cost billing, dynamic cost routing, or Kubernetes deployment.

## File Map

```text
pyproject.toml                         packaging, dependencies, pytest config
.env.example                          documented runtime configuration
Dockerfile                            gateway image
docker-compose.yml                    gateway and Redis services
app/main.py                           application factory and lifespan
app/core/config.py                    validated settings
app/core/auth.py                      X-API-Key authentication
app/core/errors.py                    normalized gateway errors
app/schemas/chat.py                   request, response, usage, SSE schemas
app/api/chat.py                       /chat transport and SSE encoding
app/api/health.py                     /health and /ready
app/providers/base.py                 provider protocol and normalized result types
app/providers/openai_compatible.py    shared OpenAI-format adapter
app/providers/deepseek.py             DeepSeek configuration
app/providers/openai.py               OpenAI configuration
app/providers/anthropic.py            Anthropic protocol adapter
app/services/rate_limiter.py          Redis token bucket and concurrency lease
app/services/cache.py                 exact cache and stampede lock
app/services/router.py                candidate construction
app/services/circuit_breaker.py       Redis circuit state machine
app/services/fallback.py              ordered provider attempts
app/services/gateway.py               cache/limit/fallback orchestration
app/observability/logging.py          structured logging setup
app/observability/metrics.py          Prometheus instruments
tests/unit/                            pure unit tests
tests/integration/                     Redis-backed tests
tests/contract/                        mocked provider HTTP/SSE tests
tests/api/                             FastAPI behavior tests
tests/smoke/                           Compose smoke client
```

---

### Task 1: Project Skeleton, Configuration, and Core Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/core/__init__.py`
- Create: `app/core/config.py`
- Create: `app/core/errors.py`
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/chat.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_schemas.py`

**Interfaces:**
- Produces: `Settings`, `get_settings()`, `GatewayError`, `ChatRequest`, `ChatResponse`, `ChatMessage`, `Usage`, `ProviderResult`, and `create_app()`.
- Consumes: no earlier task interfaces.

- [ ] **Step 1: Add packaging and test configuration**

Create `pyproject.toml` with runtime dependencies `fastapi`, `uvicorn[standard]`, `httpx`, `redis`, `pydantic-settings`, `prometheus-client`, and `python-json-logger`; add test dependencies `pytest`, `pytest-asyncio`, `fakeredis`, `respx`, and `asgi-lifespan`. Configure asyncio mode as `auto` and test path as `tests`.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-gateway-playground"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.116,<1",
  "httpx>=0.28,<1",
  "prometheus-client>=0.22,<1",
  "pydantic-settings>=2.10,<3",
  "python-json-logger>=3.3,<4",
  "redis>=6.4,<7",
  "uvicorn[standard]>=0.35,<1",
]

[project.optional-dependencies]
test = [
  "asgi-lifespan>=2.1,<3",
  "fakeredis>=2.31,<3",
  "pytest>=8.4,<9",
  "pytest-asyncio>=1.1,<2",
  "respx>=0.22,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["app"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Install the project and test dependencies**

Run: `python -m pip install -e '.[test]'`

Expected: installation completes with `llm-gateway-playground-0.1.0` installed in editable mode.

- [ ] **Step 3: Write failing settings and schema tests**

```python
# tests/unit/test_config.py
from app.core.config import Settings


def test_client_keys_are_parsed_to_named_identities():
    settings = Settings(
        client_api_keys="frontend:key-a,worker:key-b",
        deepseek_api_key="ds",
    )
    assert settings.client_keys == {"key-a": "frontend", "key-b": "worker"}
    assert settings.provider_order == ("deepseek", "openai", "anthropic")


def test_duplicate_or_malformed_client_keys_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        Settings(client_api_keys="frontend:key-a,worker:key-a")
    with pytest.raises(ValueError):
        Settings(client_api_keys="missing-separator")
```

```python
# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError
from app.schemas.chat import ChatRequest


def test_chat_request_accepts_text_roles():
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    assert request.model == "auto"
    assert request.stream is False


def test_chat_request_rejects_empty_messages_and_unknown_role():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "tool", "content": "x"}])
```

- [ ] **Step 4: Run tests and verify the expected import failure**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_schemas.py -v`

Expected: FAIL during collection because `app.core.config` and `app.schemas.chat` do not exist.

- [ ] **Step 5: Implement settings, schemas, errors, and app factory**

Implement `Settings` with exact defaults from Global Constraints. Parse `CLIENT_API_KEYS` as comma-separated `identity:key` entries, reject blank/duplicate identities or keys, expose a `client_keys: dict[str, str]` property, and use `@lru_cache` in `get_settings()`.

Define these stable interfaces:

```python
# app/schemas/chat.py
from typing import Literal
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    stream: bool = False

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

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
    fallback_count: int = 0
```

```python
# app/core/errors.py
class GatewayError(Exception):
    def __init__(self, code: str, message: str, status_code: int, retryable: bool):
        super().__init__(message)
        self.code, self.message = code, message
        self.status_code, self.retryable = status_code, retryable
```

`create_app(settings: Settings | None = None) -> FastAPI` must create FastAPI with docs enabled only when `settings.environment == "development"` and install an exception handler returning `{"error": {"code", "message", "retryable", "request_id"}}`.

- [ ] **Step 6: Run unit tests**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_schemas.py -v`

Expected: 4 tests PASS.

- [ ] **Step 7: Add `.env.example` and commit**

Document `CLIENT_API_KEYS`, all three provider keys/base URLs/default models, Redis URL, 60/1/5 rate defaults, 300-second cache TTL, 5/30 circuit defaults, and 30/75 timeout defaults. Do not place usable secrets in the file.

```bash
git add pyproject.toml .env.example app tests/unit
git commit -m "feat: scaffold gateway configuration and schemas"
```

---

### Task 2: Client API-Key Authentication

**Files:**
- Create: `app/core/auth.py`
- Create: `tests/unit/test_auth.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `Settings.client_keys`, `GatewayError` from Task 1.
- Produces: `ClientIdentity(name: str)`, `APIKeyAuthenticator.authenticate(raw_key: str | None) -> ClientIdentity`, and FastAPI dependency `get_client_identity(request: Request) -> ClientIdentity`.

- [ ] **Step 1: Write failing constant-time authentication tests**

```python
from app.core.auth import APIKeyAuthenticator
from app.core.errors import GatewayError


def test_authenticator_returns_named_identity():
    auth = APIKeyAuthenticator({"secret-a": "frontend"})
    assert auth.authenticate("secret-a").name == "frontend"


def test_authenticator_rejects_missing_and_invalid_keys():
    import pytest
    auth = APIKeyAuthenticator({"secret-a": "frontend"})
    for value in (None, "wrong"):
        with pytest.raises(GatewayError) as error:
            auth.authenticate(value)
        assert error.value.status_code == 401
```

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/unit/test_auth.py -v`

Expected: FAIL because `app.core.auth` does not exist.

- [ ] **Step 3: Implement authentication**

Use `secrets.compare_digest` to compare the supplied value with every configured key, avoiding a direct dictionary lookup timing distinction. `ClientIdentity` is an immutable dataclass containing only the configured identity name. Raise `GatewayError("invalid_api_key", "Missing or invalid API key", 401, False)` on failure. The FastAPI dependency reads only `X-API-Key`; it obtains the authenticator from `request.app.state.authenticator`.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/test_auth.py -v`

Expected: 2 tests PASS.

```bash
git add app/core/auth.py app/main.py tests/unit/test_auth.py
git commit -m "feat: authenticate named gateway clients"
```

---

### Task 3: Provider Abstraction and Three HTTP Adapters

**Files:**
- Create: `app/providers/__init__.py`
- Create: `app/providers/base.py`
- Create: `app/providers/openai_compatible.py`
- Create: `app/providers/deepseek.py`
- Create: `app/providers/openai.py`
- Create: `app/providers/anthropic.py`
- Create: `tests/contract/test_openai_compatible.py`
- Create: `tests/contract/test_anthropic.py`

**Interfaces:**
- Consumes: `ChatRequest`, `ProviderResult`, `Usage` from Task 1.
- Produces: `ProviderAdapter.complete(request, model)`, `ProviderAdapter.stream(request, model)`, `ProviderChunk`, `ProviderError`, and three concrete adapters.

- [ ] **Step 1: Define contract tests for successful calls and error normalization**

Use `respx` to assert that OpenAI and DeepSeek send `messages`, `temperature`, `max_tokens`, and `stream`; Anthropic must move all system messages into its top-level `system` field and send remaining messages in `messages`.

```python
@pytest.mark.parametrize("status,retryable", [(400, False), (401, False), (429, True), (500, True)])
async def test_openai_compatible_normalizes_http_errors(status, retryable, adapter, request, respx_mock):
    respx_mock.post("https://provider.test/chat/completions").mock(return_value=httpx.Response(status))
    with pytest.raises(ProviderError) as error:
        await adapter.complete(request, "model-a")
    assert error.value.retryable is retryable
    assert error.value.status_code == status
```

Add success assertions for content, actual model, provider, and usage. Add SSE fixtures with two content chunks and `[DONE]` for OpenAI-compatible APIs, and Anthropic fixtures using `content_block_delta` plus `message_delta` usage.

- [ ] **Step 2: Verify contract tests fail**

Run: `python -m pytest tests/contract -v`

Expected: FAIL because provider modules do not exist.

- [ ] **Step 3: Implement stable provider interfaces**

```python
# app/providers/base.py
from dataclasses import dataclass
from typing import AsyncIterator, Protocol
from app.schemas.chat import ChatRequest, ProviderResult, Usage

@dataclass(frozen=True)
class ProviderChunk:
    content: str = ""
    usage: Usage | None = None

class ProviderError(Exception):
    def __init__(self, provider: str, code: str, status_code: int | None, retryable: bool):
        super().__init__(code)
        self.provider, self.code = provider, code
        self.status_code, self.retryable = status_code, retryable

class ProviderAdapter(Protocol):
    name: str
    async def complete(self, request: ChatRequest, model: str) -> ProviderResult: ...
    def stream(self, request: ChatRequest, model: str) -> AsyncIterator[ProviderChunk]: ...
```

Both adapter families must use `httpx.AsyncClient.stream()` for SSE, ignore comment/blank lines, reject invalid/empty successful payloads as retryable `ProviderError(code="invalid_response")`, map timeouts/transport errors as retryable, and never include an API key or upstream body in exception text.

DeepSeek and OpenAI subclass/configure `OpenAICompatibleProvider`; Anthropic owns its request and response mapping. Inject `httpx.AsyncClient` into constructors so tests share deterministic transports.

- [ ] **Step 4: Run provider contract tests**

Run: `python -m pytest tests/contract -v`

Expected: all OpenAI-compatible and Anthropic success, stream, 4xx, 429, 5xx, timeout, malformed-data, and empty-output cases PASS.

- [ ] **Step 5: Commit**

```bash
git add app/providers tests/contract
git commit -m "feat: add three normalized LLM providers"
```

---

### Task 4: Redis Token Bucket and Concurrency Leases

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/rate_limiter.py`
- Create: `docker-compose.yml`
- Create: `tests/integration/test_rate_limiter.py`

**Interfaces:**
- Consumes: Redis asyncio client and `GatewayError`.
- Produces: `RateLimitDecision`, `ConcurrencyLease`, `RedisRateLimiter.acquire_rate(client)`, `acquire_concurrency(client, request_id)`, and `release_concurrency(lease)`.

- [ ] **Step 1: Add a minimal Redis development service**

Create `docker-compose.yml` with a Redis 7 service, a persistent named volume, port `6379`, and a `redis-cli ping` healthcheck. Task 9 will extend this same file with the gateway and mock-provider services.

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 2s
      timeout: 1s
      retries: 20

volumes:
  redis-data:
```

Run: `docker compose up -d redis && docker compose exec redis redis-cli ping`

Expected: output contains `PONG`.

- [ ] **Step 2: Write failing Redis integration tests**

Tests must prove these exact behaviors with a real Redis fixture when `TEST_REDIS_URL` is available and skip otherwise:

```python
async def test_token_bucket_allows_capacity_then_rejects(limiter):
    decisions = [await limiter.acquire_rate("frontend") for _ in range(61)]
    assert all(item.allowed for item in decisions[:60])
    assert decisions[60].allowed is False
    assert decisions[60].retry_after >= 1

async def test_concurrency_is_isolated_and_released(limiter):
    leases = [await limiter.acquire_concurrency("frontend", f"r{i}") for i in range(5)]
    with pytest.raises(GatewayError) as error:
        await limiter.acquire_concurrency("frontend", "r5")
    assert error.value.status_code == 429
    await limiter.release_concurrency(leases[0])
    assert await limiter.acquire_concurrency("frontend", "r6")
```

Also test client isolation, expired lease cleanup, and Redis connection failure returning `GatewayError(..., status_code=503)`.

- [ ] **Step 3: Verify the targeted tests fail**

Run: `TEST_REDIS_URL=redis://localhost:6379/15 python -m pytest tests/integration/test_rate_limiter.py -v`

Expected: FAIL because `RedisRateLimiter` is undefined.

- [ ] **Step 4: Implement atomic Lua operations**

Define:

```python
@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after: int

@dataclass(frozen=True)
class ConcurrencyLease:
    client: str
    request_id: str
```

The rate Lua script stores floating-point `tokens` and millisecond `updated_at`, refills by `(now-updated_at) * refill_per_second / 1000`, caps at capacity, consumes exactly one token, and sets a TTL of at least twice the full-refill duration. The concurrency Lua script uses a sorted set keyed by client, removes scores older than `now - lease_ttl_ms`, rejects when cardinality is 5, then adds `request_id` with the current time. Release uses `ZREM`. Map all `redis.RedisError` failures to retryable HTTP 503 `GatewayError(code="redis_unavailable")`.

- [ ] **Step 5: Run integration tests**

Run: `TEST_REDIS_URL=redis://localhost:6379/15 python -m pytest tests/integration/test_rate_limiter.py -v`

Expected: token bucket, refill, client isolation, concurrency, expiry, release, and failure tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/rate_limiter.py tests/integration/test_rate_limiter.py docker-compose.yml
git commit -m "feat: add distributed client rate limits"
```

---

### Task 5: Client-Isolated Exact Cache and Stampede Protection

**Files:**
- Create: `app/services/cache.py`
- Create: `tests/unit/test_cache_key.py`
- Create: `tests/integration/test_cache.py`

**Interfaces:**
- Consumes: `ChatRequest`, `ChatResponse`, Redis asyncio client.
- Produces: `build_cache_key(client, request) -> str`, `RedisChatCache.get/set`, and async context manager `single_flight(client, request) -> SingleFlightResult`.

- [ ] **Step 1: Write deterministic cache-key tests**

```python
def test_cache_key_is_stable_and_client_isolated():
    request_a = ChatRequest.model_validate({
        "model": "auto", "messages": [{"content": "hi", "role": "user"}],
        "temperature": 0.7, "max_tokens": 20,
    })
    request_b = ChatRequest.model_validate_json(request_a.model_dump_json())
    assert build_cache_key("frontend", request_a) == build_cache_key("frontend", request_b)
    assert build_cache_key("frontend", request_a) != build_cache_key("worker", request_a)

def test_auto_and_explicit_model_do_not_share_cache():
    auto = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    explicit = auto.model_copy(update={"model": "deepseek/deepseek-chat"})
    assert build_cache_key("frontend", auto) != build_cache_key("frontend", explicit)
```

Integration tests must prove 300-second TTL assignment, hit deserialization, `cached=True` mutation, lock exclusivity, bounded lock wait, and Redis-error fail-open behavior.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/unit/test_cache_key.py tests/integration/test_cache.py -v`

Expected: FAIL because `app.services.cache` does not exist.

- [ ] **Step 3: Implement cache normalization and Redis operations**

Serialize this exact dictionary with `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`:

```python
{
    "schema_version": 1,
    "client_identity": client,
    "requested_model": request.model,
    "messages": [message.model_dump() for message in request.messages],
    "temperature": request.temperature,
    "max_tokens": request.max_tokens,
}
```

Prefix the SHA-256 digest with `gateway:cache:v1:`. Store `ChatResponse.model_dump_json()` using `SET EX`. `get()` catches `RedisError`/invalid cached JSON and returns `None`. `set()` catches `RedisError` and returns `False`. Implement lock ownership using a random token and `SET NX PX`; release through a Lua compare-and-delete script. Waiters poll for a cache value for at most the configured lock-wait budget, then proceed without ownership.

- [ ] **Step 4: Run cache tests and commit**

Run: `python -m pytest tests/unit/test_cache_key.py tests/integration/test_cache.py -v`

Expected: all cache-key, isolation, TTL, lock, timeout, and fail-open tests PASS.

```bash
git add app/services/cache.py tests/unit/test_cache_key.py tests/integration/test_cache.py
git commit -m "feat: cache exact chat requests per client"
```

---

### Task 6: Model Router, Shared Circuit Breaker, and Ordered Fallback

**Files:**
- Create: `app/services/router.py`
- Create: `app/services/circuit_breaker.py`
- Create: `app/services/fallback.py`
- Create: `tests/unit/test_router.py`
- Create: `tests/integration/test_circuit_breaker.py`
- Create: `tests/unit/test_fallback.py`

**Interfaces:**
- Consumes: `ProviderAdapter`, `ProviderError`, provider/model configuration, Redis.
- Produces: `RouteCandidate`, `ModelRouter.candidates(model)`, `RedisCircuitBreaker.allow/success/failure`, `FallbackExecutor.complete`, and `FallbackExecutor.stream_until_first_chunk`.

- [ ] **Step 1: Write router and fallback tests**

```python
def test_auto_route_has_strict_provider_order(router):
    assert [c.provider for c in router.candidates("auto")] == ["deepseek", "openai", "anthropic"]

def test_explicit_model_uses_only_configured_equivalents(router):
    assert router.candidates("openai/gpt-x") == [
        RouteCandidate("openai", "gpt-x"),
        RouteCandidate("anthropic", "claude-x"),
    ]

async def test_fallback_skips_retryable_failure_and_counts_attempts(executor, first_fails, second_succeeds):
    result, fallback_count = await executor.complete(request, candidates)
    assert result.provider == "openai"
    assert fallback_count == 1

async def test_non_retryable_provider_error_stops_fallback(executor, first_bad_request):
    with pytest.raises(ProviderError) as error:
        await executor.complete(request, candidates)
    assert error.value.retryable is False
    assert second_provider.calls == 0
```

Circuit tests cover closed -> open after 5 consecutive failures, skip during 30 seconds, one half-open probe, successful close, failed reopen, and Redis failure fallback to process-local state.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/unit/test_router.py tests/unit/test_fallback.py tests/integration/test_circuit_breaker.py -v`

Expected: FAIL because router, circuit breaker, and fallback modules do not exist.

- [ ] **Step 3: Implement route and circuit interfaces**

```python
@dataclass(frozen=True)
class RouteCandidate:
    provider: str
    model: str

class ModelRouter:
    def candidates(self, requested_model: str) -> list[RouteCandidate]: ...

class CircuitBreaker(Protocol):
    async def allow(self, provider: str) -> bool: ...
    async def success(self, provider: str) -> None: ...
    async def failure(self, provider: str) -> None: ...
```

`ModelRouter` builds `auto` candidates from configured defaults in fixed order and explicit candidates only from an exact alias/equivalence configuration. Unknown names raise HTTP 400 `GatewayError(code="unknown_model")`.

The circuit breaker uses a Redis hash per provider for state, failures, and opened timestamp plus a short `SET NX EX` probe key. All transitions must be atomic Lua operations. Keep a lock-protected in-process implementation with the same semantics and invoke it only when Redis raises `RedisError`.

- [ ] **Step 4: Implement fallback executor**

For each allowed candidate, call its adapter with `request.model_copy(update={"stream": False})`. On success, call `circuit.success`. On retryable `ProviderError`, call `circuit.failure` and continue. On non-retryable error, stop immediately. If all candidates are skipped/failed, raise a normalized `GatewayError`: 429 only when every attempted failure was upstream 429; otherwise 503.

For streaming, `stream_until_first_chunk` must open candidates sequentially and obtain the first non-empty `ProviderChunk` before returning `(candidate, iterator_with_first_chunk, fallback_count)`. This boundary lets the API emit `meta` only after the provider is committed.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/unit/test_router.py tests/unit/test_fallback.py tests/integration/test_circuit_breaker.py -v`

Expected: all routing, stopping, ordering, circuit, and pre-first-chunk fallback tests PASS.

```bash
git add app/services/router.py app/services/circuit_breaker.py app/services/fallback.py tests/unit tests/integration/test_circuit_breaker.py
git commit -m "feat: route and degrade across providers"
```

---

### Task 7: Non-Streaming Gateway Orchestration and `/chat`

**Files:**
- Create: `app/services/gateway.py`
- Create: `app/api/__init__.py`
- Create: `app/api/chat.py`
- Create: `tests/api/test_chat.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: authenticator, rate limiter, cache, router/fallback, and schemas from Tasks 1-6.
- Produces: `GatewayService.complete(client, request, request_id) -> ChatResponse` and `POST /chat` non-streaming behavior.

- [ ] **Step 1: Write failing non-streaming API tests**

Use dependency/state-injected fakes and `httpx.AsyncClient(transport=ASGITransport(app=app))`. Cover:

```python
async def test_non_streaming_chat_returns_normalized_response(client):
    response = await client.post("/chat", headers={"X-API-Key": "key-a"}, json={
        "messages": [{"role": "user", "content": "hello"}], "stream": False,
    })
    assert response.status_code == 200
    assert response.headers["X-Cache"] == "MISS"
    assert response.headers["X-Request-ID"]
    assert response.json()["provider"] == "deepseek"

async def test_cache_hit_does_not_call_provider(client, cache, provider):
    first = await client.post("/chat", headers=AUTH, json=REQUEST)
    second = await client.post("/chat", headers=AUTH, json=REQUEST)
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert provider.calls == 1
```

Also assert 401, 400 unknown model, 429 rate/concurrency errors, 503 Redis failure, fallback metadata, lock-wait cache hit, request ID propagation/generation, and concurrency release after both success and exception.

- [ ] **Step 2: Verify API tests fail**

Run: `python -m pytest tests/api/test_chat.py -v`

Expected: FAIL because the `/chat` route and `GatewayService` do not exist.

- [ ] **Step 3: Implement `GatewayService.complete`**

Use this exact order:

```python
await rate_limiter.acquire_rate(client.name)
lease = await rate_limiter.acquire_concurrency(client.name, request_id)
try:
    cached = await cache.get(client.name, request)
    if cached is not None:
        return cached.model_copy(update={"cached": True})
    async with cache.single_flight(client.name, request) as lock:
        if not lock.owner and lock.cached_response is not None:
            return lock.cached_response.model_copy(update={"cached": True})
        result, fallback_count = await asyncio.wait_for(
            fallback.complete(request, router.candidates(request.model)),
            timeout=settings.total_timeout_seconds,
        )
        response = ChatResponse(
            id=f"chat_{request_id}", model=result.model, provider=result.provider,
            message=ChatMessage(role="assistant", content=result.content),
            usage=result.usage, cached=False, fallback_count=fallback_count,
        )
        await cache.set(client.name, request, response)
        return response
finally:
    await rate_limiter.release_concurrency(lease)
```

Convert total `asyncio.TimeoutError` to retryable HTTP 504. Do not access cache when `request.stream` is true.

- [ ] **Step 4: Implement HTTP transport**

`POST /chat` authenticates through `Depends(get_client_identity)`, accepts/creates `X-Request-ID`, validates it to a bounded safe character set or replaces it, invokes `GatewayService`, and returns `JSONResponse` with `X-Request-ID` and `X-Cache`. Store services on `app.state` through the application lifespan so tests can inject fakes.

- [ ] **Step 5: Run API tests and commit**

Run: `python -m pytest tests/api/test_chat.py -v`

Expected: all non-streaming auth, cache, rate, fallback, timeout, headers, and cleanup tests PASS.

```bash
git add app/services/gateway.py app/api app/main.py tests/api/test_chat.py
git commit -m "feat: expose non-streaming chat gateway"
```

---

### Task 8: Normalized SSE Streaming and Disconnect Cleanup

**Files:**
- Modify: `app/services/gateway.py`
- Modify: `app/api/chat.py`
- Create: `tests/api/test_chat_stream.py`

**Interfaces:**
- Consumes: `FallbackExecutor.stream_until_first_chunk`, rate/concurrency services, provider chunks.
- Produces: `GatewayService.stream(...) -> AsyncIterator[StreamEvent]` and normalized SSE events `meta`, `delta`, `done`, `error`.

- [ ] **Step 1: Write failing streaming tests**

Test raw response lines and event order:

```python
async def test_stream_emits_meta_deltas_and_done(client):
    async with client.stream("POST", "/chat", headers=AUTH, json={**REQUEST, "stream": True}) as response:
        body = await response.aread()
    text = body.decode()
    assert text.index("event: meta") < text.index("event: delta") < text.index("event: done")
    assert response.headers["content-type"].startswith("text/event-stream")

async def test_stream_falls_back_before_first_delta(client, deepseek_fails, openai_streams):
    body = (await client.post("/chat", headers=AUTH, json={**REQUEST, "stream": True})).text
    assert '"provider":"openai"' in body
    assert '"fallback_count":1' in body
    assert "deepseek" not in body

async def test_stream_error_after_delta_never_calls_next_provider(client, breaks_midstream):
    body = (await client.post("/chat", headers=AUTH, json={**REQUEST, "stream": True})).text
    assert "event: delta" in body and "event: error" in body
    assert openai.calls == 0
```

Also test no cache calls, lease held until stream completion, lease release on cancellation/disconnect, JSON compact encoding, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.

- [ ] **Step 2: Verify streaming tests fail**

Run: `python -m pytest tests/api/test_chat_stream.py -v`

Expected: FAIL because streaming orchestration is absent.

- [ ] **Step 3: Implement normalized stream events and encoder**

Define discriminated Pydantic models `MetaEvent`, `DeltaEvent`, `DoneEvent`, and `ErrorEvent`, or an equivalent immutable dataclass hierarchy. Encode each event exactly as:

```python
def encode_sse(event: str, payload: BaseModel | dict) -> bytes:
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()
```

`GatewayService.stream` acquires rate and concurrency limits, calls `stream_until_first_chunk`, yields one `meta` with actual provider/model/fallback count, yields first and subsequent non-empty deltas, and yields `done` with final usage. Catch errors only after `meta`/`delta` has begun and yield a sanitized retryable `error`; never start another adapter after commitment. Release the concurrency lease in the generator's `finally` block.

- [ ] **Step 4: Return `StreamingResponse` safely**

The API route returns `StreamingResponse(generator, media_type="text/event-stream")` with `X-Request-ID`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`. A client disconnect/cancellation must close the provider iterator and run generator cleanup. Pre-first-chunk failures must be resolved before returning the first bytes; if every candidate fails, the generator emits one SSE `error` because HTTP headers may already have been committed by `StreamingResponse`.

- [ ] **Step 5: Run streaming and regression tests, then commit**

Run: `python -m pytest tests/api/test_chat_stream.py tests/api/test_chat.py tests/contract -v`

Expected: all streaming, non-streaming, and provider contract tests PASS.

```bash
git add app/services/gateway.py app/api/chat.py app/schemas/chat.py tests/api/test_chat_stream.py
git commit -m "feat: stream normalized chat events"
```

---

### Task 9: Observability, Health, Docker Compose, and End-to-End Verification

**Files:**
- Create: `app/observability/__init__.py`
- Create: `app/observability/logging.py`
- Create: `app/observability/metrics.py`
- Create: `app/api/health.py`
- Modify: `app/main.py`
- Create: `tests/api/test_health_metrics.py`
- Create: `tests/unit/test_logging.py`
- Create: `Dockerfile`
- Modify: `docker-compose.yml`
- Create: `.dockerignore`
- Create: `tests/smoke/mock_provider.py`
- Create: `tests/smoke/run.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: all application services and settings.
- Produces: JSON request logs, Prometheus `/metrics`, `/health`, `/ready`, container startup, mock-provider smoke test, and operator documentation.

- [ ] **Step 1: Write failing health, metrics, and redaction tests**

```python
async def test_health_is_live_when_redis_is_down(client_with_broken_redis):
    assert (await client_with_broken_redis.get("/health")).status_code == 200
    assert (await client_with_broken_redis.get("/ready")).status_code == 503

async def test_ready_requires_redis_and_one_configured_provider(ready_client):
    response = await ready_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

async def test_metrics_include_gateway_counters(client):
    await client.post("/chat", headers=AUTH, json=REQUEST)
    metrics = (await client.get("/metrics")).text
    assert "gateway_requests_total" in metrics
    assert "gateway_request_duration_seconds" in metrics
```

The logging test captures one JSON record and asserts it contains request ID, client identity/hash, requested model, actual provider/model, duration, cache status, fallback count, and normalized error while excluding raw keys, message content, and output content.

- [ ] **Step 2: Verify observability tests fail**

Run: `python -m pytest tests/api/test_health_metrics.py tests/unit/test_logging.py -v`

Expected: FAIL because health and observability modules do not exist.

- [ ] **Step 3: Implement health, metrics, and structured logging**

`/health` always returns `{"status":"ok"}` when the process can serve HTTP. `/ready` performs `redis.ping()` with a short timeout and verifies at least one provider API key/default model pair; return 503 otherwise.

Create Prometheus counters/histograms/gauges with the exact names from the design. Labels may include provider, model alias, outcome, cache status, and error code; never label by client key, request ID, or prompt to avoid cardinality and data leakage. Mount `/metrics` using an ASGI endpoint or return `generate_latest()` with the Prometheus content type.

Configure `python-json-logger` once during app startup. Emit one completion record per request plus provider-attempt records. Apply an explicit allowlist of log fields rather than redacting arbitrary dictionaries.

- [ ] **Step 4: Run observability tests**

Run: `python -m pytest tests/api/test_health_metrics.py tests/unit/test_logging.py -v`

Expected: health readiness, metric presence, label safety, and log redaction tests PASS.

- [ ] **Step 5: Add container and smoke-test files**

Use this container shape:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .
USER 65532:65532
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Compose defines `redis` with Redis 7 healthcheck and `gateway` depending on healthy Redis, reading `.env`, exposing port 8000, and using `redis://redis:6379/0`. Add an optional `mock-provider` profile used by the smoke script. The mock provider must implement OpenAI-compatible non-streaming and SSE responses without external API calls.

`tests/smoke/run.sh` must use `set -euo pipefail`, start the Compose smoke profile, wait on `/ready` with a bounded retry loop, call non-streaming `/chat`, assert provider/content with a small Python JSON expression, call streaming `/chat`, assert `event: done`, and always run `docker compose down -v` in a trap.

- [ ] **Step 6: Document operation**

`README.md` must include prerequisites, copying `.env.example`, configuring named client keys and provider keys, `docker compose up --build`, curl examples for JSON and SSE, `/health`/`ready`/`metrics`, default rate/cache/fallback behavior, tests, and explicit warnings that prompts are cached in Redis for five minutes and real-provider manual tests may incur cost.

- [ ] **Step 7: Run the complete verification suite**

Run: `python -m pytest -v`

Expected: all unit, contract, integration (or clearly marked skip without `TEST_REDIS_URL`), and API tests PASS.

Run: `docker compose config`

Expected: exits 0 with valid `gateway` and `redis` services.

Run: `bash tests/smoke/run.sh`

Expected: prints successful non-streaming and streaming checks and exits 0 after cleanup.

- [ ] **Step 8: Commit the completed prototype**

```bash
git add app tests Dockerfile docker-compose.yml .dockerignore README.md .env.example
git commit -m "feat: ship observable compose gateway prototype"
```

---

## Final Acceptance

- [ ] Run `python -m pytest -v` and confirm no failures.
- [ ] Run `docker compose config` and confirm valid configuration.
- [ ] Run `bash tests/smoke/run.sh` and confirm both response modes.
- [ ] Run a manual request with each configured real Provider, without recording secrets or response bodies in Git.
- [ ] Confirm `auto` fallback order through mock-provider logs.
- [ ] Confirm a repeated non-streaming request produces `X-Cache: MISS` then `X-Cache: HIT` for one client and remains `MISS` for another.
- [ ] Confirm the 61st immediate request is rejected and a sixth concurrent stream receives HTTP 429.
- [ ] Confirm mid-stream failure produces `event: error` and does not invoke another Provider.
- [ ] Review `git diff --check` and `git status --short` before final handoff.
