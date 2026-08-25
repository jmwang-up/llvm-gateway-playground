from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total",
            "Gateway HTTP requests",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "gateway_request_duration_seconds",
            "Gateway HTTP request duration",
            ("method", "path"),
            registry=self.registry,
        )
        self.cache_hits = Counter(
            "gateway_cache_hits_total",
            "Successful chat cache hits",
            registry=self.registry,
        )
        self.rate_limit_rejections = Counter(
            "gateway_rate_limit_rejections_total",
            "Gateway rate-limit rejections",
            registry=self.registry,
        )
        self.provider_requests = Counter(
            "gateway_provider_requests_total",
            "Provider requests",
            ("provider", "outcome"),
            registry=self.registry,
        )
        self.provider_errors = Counter(
            "gateway_provider_errors_total",
            "Provider request errors",
            ("provider", "error_code"),
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "gateway_fallbacks_total",
            "Provider fallback attempts",
            ("from_provider", "to_provider"),
            registry=self.registry,
        )
        self.circuit_state = Gauge(
            "gateway_circuit_state",
            "Provider circuit state where 1 means open",
            ("provider",),
            registry=self.registry,
        )
        self.active_requests = Gauge(
            "gateway_active_requests",
            "Currently active gateway HTTP requests",
            registry=self.registry,
        )

