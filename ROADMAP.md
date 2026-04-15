# Roadmap

High-level view of where Modelship is headed. If something here interests you, open an issue or jump into a discussion — contributions are welcome.

## In Progress

- **Transformers CPU backend** — run chat, embeddings, STT, and TTS on CPU-only machines via HuggingFace Transformers (landed in v0.1.22, expanding model support)
- **Observability** — Prometheus metrics, Grafana dashboard, structured JSON logging, syslog and OpenTelemetry export (mostly complete)

## Up Next

### Testing
- API endpoint tests for all `/v1/` routes
- Integration tests with real model loading (small models)
- Streaming (SSE) correctness tests
- Plugin lifecycle tests

### Core Improvements
- Detailed health checks — `/health` should verify model state, GPU status, and Ray cluster connectivity
- Model hot-reload — apply `models.yaml` changes without full server restart
- Docker Compose for simpler non-K8s deployments

### Plugin Ecosystem
- More TTS backends (community-contributed)
- Plugin template / scaffolding CLI for faster plugin development
- Broader plugin types beyond TTS (e.g. custom pre/post-processing, custom loaders)

### Documentation
- OpenAPI/Swagger spec for the API
- Performance tuning guide (vLLM engine kwargs, batch sizes, KV cache)
- Capacity planning guide — model co-location recommendations per GPU size

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started. The **Testing** and **Plugin Ecosystem** sections above are great places to make an impact — most items are self-contained and well-defined.
