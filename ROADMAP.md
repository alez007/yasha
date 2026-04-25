# Roadmap

High-level view of where Modelship is headed. If something here interests you, open an issue or jump into a discussion — contributions are welcome.

## Recently Shipped

- **Dynamic wheel-based plugins** (v0.1.32-dev) — plugin packages build into standalone wheels and are injected into Ray workers at deployment via `runtime_env`. `MSHIP_PLUGINS` is deprecated; plugins resolve automatically from `models.yaml`.
- **Unified Dockerfile** (v0.1.32-dev) — single `Dockerfile` with `--build-arg MSHIP_VARIANT=cpu|gpu`; `Dockerfile.cpu` removed.
- **Auto-detected Ray resources** (v0.1.32-dev) — `RAY_HEAD_CPU_NUM` / `RAY_HEAD_GPU_NUM` are now optional overrides; Ray auto-detects CPUs and GPUs from the host or container cgroups.
- **CPU inference backends** — `transformers` (v0.1.22) and `llama_cpp` (v0.1.24) loaders cover chat, embeddings, STT, and TTS without a GPU.
- **Cluster-wide deploy coordinator** (v0.1.30) — retry-pass deploy loop and `/status` readiness endpoint with per-model load timings.
- **Observability** — Prometheus metrics, Grafana dashboard, structured JSON logging, syslog and OpenTelemetry export, alerting rules.

## Up Next

### Testing
- API endpoint tests for all `/v1/` routes
- Integration tests with real model loading (small models)
- Streaming (SSE) correctness tests
- Plugin lifecycle tests (wheel build, runtime_env injection, fallback paths)

### Core Improvements
- Detailed health checks — `/health` should verify model state, GPU status, and Ray cluster connectivity per model
- Model hot-reload — apply `models.yaml` changes without full server restart
- Docker Compose for simpler non-K8s deployments
- Helm chart and Kubernetes manifests with proper GPU scheduling, probes, and resource limits

### Plugin Ecosystem
- More TTS / STT backends (community-contributed)
- Plugin template / scaffolding CLI for faster plugin development
- Broader plugin types beyond TTS/STT (e.g. custom pre/post-processing, custom loaders)
- Plugin sandboxing or signature verification (plugins currently run with full server privileges)

### Documentation
- OpenAPI/Swagger spec for the API
- Performance tuning guide (vLLM engine kwargs, batch sizes, KV cache)
- Capacity planning guide — model co-location recommendations per GPU size
- Multi-node Ray cluster setup (head + workers, networking, failure handling)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started. The **Testing** and **Plugin Ecosystem** sections above are great places to make an impact — most items are self-contained and well-defined.
