# AI Gateway 工作原型设计

## 1. 目标

构建一个可运行且结构可演进的 AI 网关，对外提供统一的 `POST /chat` 接口，内部接入 DeepSeek、OpenAI 和 Anthropic 三家 LLM API。

第一版必须支持：

- 非流式 JSON 响应和基于 SSE 的流式响应。
- 多客户端 API Key 鉴权。
- 按客户端隔离的请求限流、并发限制和精确响应缓存。
- 固定优先级路由、故障降级和 Provider 熔断。
- Docker Compose 一键启动网关与 Redis。
- 结构化日志、Prometheus 指标、健康检查及自动化测试。

第一版不包含管理后台、数据库、语义缓存、动态成本路由、计费系统和 Kubernetes 部署。

## 2. 技术栈

- Python 3.12
- FastAPI + Uvicorn
- Pydantic Settings
- HTTPX 异步客户端
- Redis
- Prometheus Client
- Pytest + pytest-asyncio
- Docker Compose

网关直接通过 HTTPX 调用 Provider API，不绑定三家官方 SDK，以统一控制连接池、超时、错误映射和流式解析。

## 3. 系统架构

请求处理链路：

```text
Client
  -> API Key authentication
  -> per-client rate and concurrency limits
  -> non-streaming cache lookup
  -> model routing
  -> DeepSeek Provider
       failure -> OpenAI Provider
       failure -> Anthropic Provider
  -> normalized JSON response or SSE events
  -> cache successful non-streaming response
```

建议的代码结构：

```text
app/
├── main.py
├── api/
│   ├── chat.py
│   └── health.py
├── core/
│   ├── config.py
│   ├── auth.py
│   └── errors.py
├── schemas/
│   └── chat.py
├── services/
│   ├── gateway.py
│   ├── router.py
│   ├── rate_limiter.py
│   ├── cache.py
│   └── fallback.py
├── providers/
│   ├── base.py
│   ├── deepseek.py
│   ├── openai.py
│   └── anthropic.py
└── observability/
    ├── logging.py
    └── metrics.py
```

模块职责：

- API 层负责 HTTP/SSE 协议、请求校验和响应头。
- `GatewayService` 编排鉴权后的完整请求流程，但不处理具体 Provider 协议。
- Router 把 `auto` 或显式模型解析为有序候选 Provider 列表。
- Provider 适配器负责请求转换、响应转换、SSE 解析和错误归一化。
- 限流、缓存、降级与熔断服务通过明确接口独立实现和测试。

## 4. 配置与鉴权

三家上游 API Key、客户端 Key、默认模型、超时、限流阈值和缓存 TTL 均由环境变量配置。仓库提供 `.env.example`，不提交真实密钥。

客户端请求必须携带：

```http
X-API-Key: <client-key>
```

第一版通过环境变量声明多个命名客户端 Key。Key 名称作为内部客户端身份，用于限流、缓存隔离和指标；日志只记录身份或 Key 的不可逆摘要，不记录真实 Key。第一版所有客户端共享相同限额，不提供运行时 Key 管理和差异化配额。

## 5. `/chat` API

### 5.1 请求

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is an AI gateway?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

第一版约束：

- `messages` 支持 `system`、`user`、`assistant` 三种角色和纯文本内容。
- `model` 支持 `auto`、配置声明的模型别名和 `provider/model-name` 格式。
- 无法识别或没有候选 Provider 的模型返回 HTTP 400。
- `temperature`、`max_tokens` 使用网关统一校验范围，再由适配器映射至上游参数。

### 5.2 非流式响应

```json
{
  "id": "chat_xxx",
  "model": "deepseek/deepseek-chat",
  "provider": "deepseek",
  "message": {
    "role": "assistant",
    "content": "An AI gateway is..."
  },
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 80,
    "total_tokens": 100
  },
  "cached": false,
  "fallback_count": 0
}
```

响应携带 `X-Request-ID` 和 `X-Cache: HIT|MISS`。

### 5.3 流式响应

流式请求返回 `text/event-stream`：

```text
event: meta
data: {"id":"chat_xxx","model":"deepseek/deepseek-chat","provider":"deepseek"}

event: delta
data: {"content":"An AI"}

event: done
data: {"usage":{"prompt_tokens":20,"completion_tokens":80,"total_tokens":100}}
```

统一事件类型为 `meta`、`delta`、`done` 和 `error`。流式响应携带 `X-Request-ID`，不使用响应缓存。

若首个 `delta` 发出前 Provider 失败，网关可以切换至下一候选 Provider；只有最终选定 Provider 的 `meta` 事件会发给客户端。一旦发出首个 `delta`，禁止拼接另一 Provider 的内容；中途失败时发送 `error` 事件并关闭连接。

## 6. 模型路由

`model: auto` 的默认 Provider 顺序固定为：

```text
DeepSeek -> OpenAI -> Anthropic
```

每个 Provider 的默认模型由配置指定。显式模型通过配置中的等价模型映射生成降级候选链。只有明确配置为等价能力的模型才能互相降级，网关不会凭名称推测替代模型。

响应中的 `model` 和 `provider` 始终反映实际完成请求的上游，而非原始的 `auto` 值。

## 7. 限流和并发控制

每个客户端 Key 默认使用：

- 令牌桶容量 60。
- 每秒恢复 1 个令牌，即稳定速率 60 RPM。
- 最大并发请求数 5。

Redis Lua 脚本原子完成令牌检查、补充与扣减，确保多实例部署无竞态。并发控制使用带租约过期时间的 Redis 记录；请求正常结束或客户端断开时主动释放，异常退出时依靠租约过期自动回收。流式请求在连接完整生命周期内占用并发名额。

超限返回 HTTP 429，并提供 `Retry-After` 和 `X-RateLimit-*` 响应头。Redis 不可用时采用 fail-closed 策略：需要限流状态的新 `/chat` 请求返回 HTTP 503，避免在失去保护时无界访问付费上游。`/health` 仍可访问，`/ready` 返回未就绪。

## 8. 精确请求缓存

缓存仅适用于 `stream: false` 的成功响应。缓存键为以下规范化内容的 SHA-256：

```text
schema_version
+ client_identity
+ requested_model
+ normalized_messages
+ temperature
+ max_tokens
```

其中 `requested_model` 保留原始语义；`auto` 与显式模型不共享缓存。JSON 使用稳定字段排序和确定性序列化。默认 TTL 为 300 秒，可通过环境变量调整。

缓存规则：

- 客户端之间严格隔离缓存。
- 不缓存错误、空响应、取消请求或流式响应。
- 缓存值保存完整统一响应；命中时将 `cached` 改为 `true`。
- 使用有界等待的短期 Redis 锁抑制相同请求造成的缓存击穿。
- 等待锁超时后允许请求继续调用上游，避免锁问题阻塞服务。
- Redis 故障时无法安全执行限流，因此请求会在限流阶段返回 503；缓存模块本身按 fail-open 设计，便于未来把限流与缓存 Redis 分离。

第一版不做语义缓存。

## 9. 降级与熔断

以下错误可触发降级：

- 连接失败或读取超时。
- 上游 HTTP 429。
- 上游 HTTP 5xx。
- 上游返回无法解析、无效或空响应。

参数错误、鉴权错误和其他确定性的 4xx 不触发降级。Provider 内部默认不额外重试，避免一次网关请求产生不可控的调用放大；失败后由降级控制器尝试下一个候选。

每个 Provider 使用共享熔断状态：

- 连续失败 5 次后打开熔断。
- 熔断 30 秒内跳过该 Provider。
- 到期后通过 Redis 锁只允许一个半开探测请求。
- 探测成功则关闭熔断并清零失败计数。
- 探测失败则重新打开熔断。

熔断状态正常保存在 Redis。由于限流要求 Redis 可用，Redis 故障时新的聊天请求不会进入 Provider 调用链；进程内熔断状态仅用于已在途请求和未来拆分 Redis 依赖时的防御性退化。

默认单 Provider 超时 30 秒，非流式完整调用链总预算 75 秒。流式请求的连接及首个内容等待预算为每 Provider 30 秒；首个 `delta` 发出后不再进行跨 Provider 降级。所有阈值均可配置。

## 10. 错误模型

非流式错误统一为：

```json
{
  "error": {
    "code": "upstream_unavailable",
    "message": "No model provider is currently available.",
    "retryable": true,
    "request_id": "req_xxx"
  }
}
```

主要状态码：

- 400：请求或模型参数无效。
- 401：客户端 API Key 缺失或无效。
- 429：网关限流或所有上游均限流。
- 502：上游响应无效。
- 503：Redis 不可用、全部 Provider 不可用或全部熔断。
- 504：完整调用超时。

流式响应在尚未输出事件时可使用普通 HTTP 错误；已经开始输出后使用 `event: error` 并结束连接。

## 11. 可观测性和健康检查

结构化 JSON 日志记录：

- `request_id`
- 客户端身份或 Key 摘要
- 请求模型和实际 Provider/模型
- 总耗时和各 Provider 尝试耗时
- 缓存命中状态
- 降级次数
- 标准化错误类型

日志不记录真实 Key、完整提示词或完整模型输出。

Prometheus `/metrics` 至少暴露：

```text
gateway_requests_total
gateway_request_duration_seconds
gateway_cache_hits_total
gateway_rate_limit_rejections_total
gateway_provider_requests_total
gateway_provider_errors_total
gateway_fallbacks_total
gateway_circuit_state
gateway_active_requests
```

健康端点：

- `/health` 仅报告进程存活。
- `/ready` 验证 Redis 可访问且至少配置一个 Provider。
- `/docs` 在开发环境启用。

## 12. 测试策略

- 单元测试：请求规范化、缓存键、模型路由、错误映射和熔断状态机。
- Redis 集成测试：令牌桶原子性、并发租约、TTL、缓存隔离和击穿保护。
- Provider 契约测试：通过 Mock HTTP 服务覆盖三家普通响应、流式响应、超时、429、5xx 和畸形数据。
- API 测试：鉴权、非流式、流式、缓存命中、限流和完整降级链。
- Docker Compose 冒烟测试：启动网关与 Redis，并通过模拟 Provider 完成一次 `/chat`。
- 真实 Provider API 不进入自动测试；项目提供手动验证脚本，避免自动产生费用和外部不稳定性。

## 13. 交付与验收标准

第一版完成时应满足：

1. `docker compose up --build` 能启动网关和 Redis。
2. 使用任一有效客户端 Key 可调用 `/chat` 的非流式和流式模式。
3. 三个 Provider 适配器均可通过契约测试，并可通过配置接入真实 API。
4. `auto` 严格按照 DeepSeek、OpenAI、Anthropic 的顺序选择和降级。
5. 每客户端 60 RPM、5 并发限制在多协程测试中保持原子性。
6. 非流式缓存按客户端隔离，TTL、命中标记和击穿保护正确。
7. 降级、熔断、超时与流式首片段边界符合本设计。
8. 自动化测试通过，并能查看结构化日志、健康状态和 Prometheus 指标。

