# Production Readiness Plan

Future development priorities for making Yasha production-ready, organized by severity and area.

## Critical (Must Have Before Production)

### Security

- [x] **API authentication layer** — API key auth at the gateway level via `YASHA_API_KEYS` env var; OpenAI-compatible `Authorization: Bearer <key>` header
- [ ] **Rate limiting** — per-user/IP/model throttling to prevent GPU resource monopolization
- [ ] **Input size limits** — max prompt length and max_tokens enforcement at the gateway to prevent GPU OOM
- [ ] **Lock down CORS** — replace wildcard `*` origins with environment-specific allowed origins
- [ ] **Plugin sandboxing** — plugins run with full server privileges; add signature verification or sandboxing

### Health & Readiness

- [ ] **Detailed readiness probe** — `/health` must check model health, GPU status, and Ray cluster connectivity (currently returns `ok` unconditionally)
- [ ] **Model-specific health checks** — per-model liveness status (vLLM engine, Ray actor state)
- [ ] **GPU memory checks** — detect and report memory pressure before OOM

### Testing

- [ ] **API endpoint tests** — HTTP-level tests for all `/v1/` endpoints
- [ ] **Integration tests** — actual model loading and inference (at least with a tiny model)
- [ ] **Streaming tests** — SSE streaming correctness and error handling
- [ ] **Plugin loading tests** — verify plugin lifecycle
- [ ] **Error recovery tests** — simulate failures and verify behavior

---

## High Priority (Should Have)

### Deployment & Infrastructure

- [ ] **Kubernetes manifests** — Deployment, Service, ConfigMap, PVC, Ingress with proper resource requests/limits, GPU scheduling, node affinity, tolerations
- [ ] **Helm chart** — parameterized deployment for different environments
- [ ] **Docker Compose** — for simpler non-K8s deployments
- [ ] **Liveness/readiness probes in container spec** — wire the improved `/health` endpoint into K8s probes

### Alerting & Observability

- [ ] **Prometheus alerting rules** — error rate thresholds, latency P99 breaches, model load failures, GPU memory pressure, Ray actor crashes
- [ ] **SLO/SLI definitions** — define target availability and latency for each endpoint type
- [ ] **Structured logging (JSON)** — for log aggregation (ELK/Loki/Splunk)
- [ ] **Request-ID correlation** — trace a request from gateway through Ray actor boundaries
- [ ] **Log level configuration** — via environment variable

### Resilience

- [ ] **Ray actor restart policies** — auto-restart crashed model actors
- [ ] **Circuit breaker** — stop routing to a failing model after N consecutive errors
- [ ] **Backpressure / queue depth limits** — reject requests when queue is saturated instead of unbounded queuing
- [ ] **Graceful shutdown timeout** — add timeout wrapper around `serve.shutdown()` to prevent hanging
- [ ] **GPU OOM recovery** — detect and recover from GPU memory exhaustion

### Update Strategy

- [ ] **Rolling update support** — configure Ray Serve's built-in rolling updates for zero-downtime deploys
- [ ] **Model hot-reload** — allow `models.yaml` changes without full server restart
- [ ] **Changelog** — track breaking changes between versions
- [ ] **Migration guide** — document config format changes between versions

---

## Medium Priority (Nice to Have)

### CI/CD Hardening

- [ ] **Security scanning** — Trivy for Docker images, dependency vulnerability checks
- [ ] **SBOM generation** — Software Bill of Materials for supply chain visibility
- [ ] **Multi-arch builds** — ARM64 support alongside AMD64
- [ ] **Performance benchmarks in CI** — detect throughput/latency regressions

### Operations

- [ ] **Secrets management integration** — document Vault / K8s Secrets / sealed-secrets usage for `HF_TOKEN` and future API keys
- [ ] **Troubleshooting runbook** — common failure modes and resolution steps for on-call
- [ ] **Capacity planning guide** — estimate concurrent users per GPU setup per model mix
- [ ] **GPU memory budgeting guide** — model co-location recommendations to avoid fragmentation
- [ ] **Multi-node Ray cluster setup docs** — head + worker topology, networking, failure handling
- [ ] **Request audit trail** — persistent log of requests for compliance/debugging

### Documentation

- [ ] **OpenAPI/Swagger spec** — formal API reference for consumers
- [ ] **Performance tuning guide** — vLLM engine kwargs, batch sizes, KV cache sizing
- [ ] **Blue-green / canary deployment patterns** — documented strategies for safe rollouts
- [ ] **Model pre-warming** — mechanism to pre-download and cache models before deploy, reducing cold start from minutes to seconds

---

## Current Scorecard

| Area                         | Current | Target |
|------------------------------|---------|--------|
| Architecture & Design        | 8/10    | 9/10   |
| Monitoring (metrics)         | 9/10    | 9/10   |
| Monitoring (alerting + logs) | 4/10    | 8/10   |
| Security                     | 3/10    | 8/10   |
| Resilience                   | 5/10    | 8/10   |
| Testing                      | 3/10    | 7/10   |
| DevOps Experience            | 5/10    | 8/10   |
| Update/Deploy Strategy       | 3/10    | 7/10   |
