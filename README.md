# Modelship

Self-hosted, multi-model AI inference server. Runs LLMs alongside specialized models (TTS, speech-to-text, embeddings, image generation) on one or more GPUs, exposing an OpenAI-compatible API. Built on [vLLM](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

## Architecture

```mermaid
graph TD
    Client["Client (OpenAI SDK / curl)"]
    API["FastAPI Gateway<br/>OpenAI-compatible API<br/>:8000"]

    Client -->|HTTP| API
    API -->|round-robin| LLM_GPU
    API -->|round-robin| LLM_CPU
    API -->|round-robin| TTS
    API -->|round-robin| STT
    API -->|round-robin| EMB

    subgraph GPU0["GPU 0"]
        LLM_GPU["LLM Deployment<br/>e.g. Llama 3.1 8B<br/>70% GPU"]
        TTS["TTS Deployment<br/>e.g. Kokoro 82M<br/>5% GPU"]
    end

    subgraph GPU1["GPU 1"]
        STT["STT Deployment<br/>e.g. Whisper<br/>50% GPU"]
        EMB["Embedding Deployment<br/>e.g. Nomic Embed<br/>50% GPU"]
    end

    subgraph CPU["CPU-only"]
        LLM_CPU["LLM Deployment<br/>e.g. Llama 3.1 8B<br/>CPU-only replica"]
    end
```

Each model runs as an isolated [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) deployment with its own lifecycle, health checks, and resource budget. Models can be deployed across multiple GPUs, run on CPU-only, or both — multiple deployments of the same model (e.g. one on GPU, one on CPU) are load-balanced with round-robin routing. Each deployment can also scale horizontally with `num_replicas`.

## Requirements

- **NVIDIA GPU** — 16 GB+ VRAM recommended for a full stack (LLM + TTS + STT + embeddings); 8 GB is sufficient for lighter setups
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **HuggingFace token** for gated models

## Features

- **Multi-model, multi-GPU** — run chat, embedding, STT, TTS, and image generation models simultaneously across one or more GPUs with tunable per-model GPU memory allocation; models can also run on CPU-only
- **Per-model isolated deployments** — each model runs in its own Ray Serve deployment with independent lifecycle, health checks, failure isolation, and configurable replica count
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK client
- **Streaming** — SSE streaming for chat completions and TTS audio
- **Tool/function calling** — auto tool choice with configurable parsers
- **Plugin system** — opt-in TTS backends installed as isolated uv workspace packages
- **Multi-GPU & hybrid routing** — assign models to specific GPUs or run them on CPU-only; deploy the same model on both GPU and CPU and requests are load-balanced via round-robin; full tensor parallelism support for large models spanning multiple GPUs
- **Client disconnect detection** — cancels in-flight inference when the client disconnects, freeing GPU resources immediately
- **Prometheus metrics & Grafana dashboard** — built-in observability with custom `modelship:*` metrics, vLLM engine stats, and Ray cluster metrics on a single scrape endpoint; pre-built Grafana dashboard included
- **Ray dashboard** — monitor deployments, resources, and request logs

## Supported OpenAI Endpoints

| Endpoint | Usecase |
|---|---|
| `POST /v1/chat/completions` | Chat / text generation (streaming and non-streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text |
| `POST /v1/audio/translations` | Audio translation |
| `POST /v1/audio/speech` | Text-to-speech (SSE streaming or single-response) |
| `POST /v1/images/generations` | Image generation |
| `GET /v1/models` | List available models |

## Quick Start

Pull the latest image from GHCR:

```bash
docker pull ghcr.io/alez007/modelship:latest
```

Create a `models.yaml` config file (see [config/models.yaml](config/models.yaml) for an example):

```yaml
models:
  - name: qwen
    model: Qwen/Qwen3-0.6B
    loader: vllm
```

Start the server:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -e MSHIP_PLUGINS=kokoro \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/modelship/.cache/models \
  -p 8265:8265 -p 8000:8000 -p 8079:8079 ghcr.io/alez007/modelship:latest
```

Try it out:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

- API: `http://localhost:8000`
- Prometheus metrics: `http://localhost:8079`
- Ray dashboard: `http://localhost:8265`

### Additive Deploys

By default, running `start.py` with a new config adds models to the running cluster without disrupting existing deployments:

```bash
# Deploy LLMs
python start.py --config config/llm.yaml

# Later, add TTS models — LLMs keep running
python start.py --config config/tts.yaml
```

Use `--redeploy` to tear down everything and start fresh. See [Model Configuration](docs/model-configuration.md) for the full CLI reference.

## Plugin Support

Modelship's TTS system is built around a plugin architecture — each TTS backend is an opt-in package with its own isolated dependencies. Plugins ship inside this repo (`plugins/`) or can be installed from PyPI.

To enable plugins, pass them as extras at sync time:

```bash
uv sync --extra kokoro
uv sync --extra kokoro --extra orpheus  # multiple plugins
```

When using Docker, set the `MSHIP_PLUGINS` environment variable:

```
MSHIP_PLUGINS=kokoro,orpheus
```

For a full guide on writing your own plugin, see [Plugin Development](docs/plugins.md).

## Documentation

- [Development](docs/development.md) — dev environment setup, building, and running locally
- [Model Configuration](docs/model-configuration.md) — full `models.yaml` reference, GPU pinning, environment variables
- [Architecture](docs/architecture.md) — system design, request lifecycle, plugin loading
- [Plugin Development](docs/plugins.md) — writing custom TTS backends
- [Home Assistant Integration](docs/home-assistant.md) — Wyoming protocol setup for voice automation
- [Monitoring & Logging](docs/monitoring.md) — Prometheus metrics, Grafana dashboard, structured logging, health checks

## Monitoring

Modelship exposes Prometheus metrics (Ray cluster, Ray Serve, vLLM, and custom `modelship:*` metrics) through a single scrape endpoint on port 8079. Metrics are **enabled by default** — set `MSHIP_METRICS=false` to disable. A pre-built Grafana dashboard is included.

Logging supports structured JSON output (`MSHIP_LOG_FORMAT=json`) and request ID correlation across Ray actor boundaries. Set `MSHIP_LOG_LEVEL` to `DEBUG` for request bodies or `TRACE` to include library internals.

See [Monitoring & Logging](docs/monitoring.md) for full details.

## Production Readiness

See the full [Production Readiness Plan](docs/production-readiness.md) for details. Summary of current status:

| Area                         | Score | Key Gaps |
|------------------------------|-------|----------|
| Architecture & Design        | 8/10  | Add K8s manifests, improve health checks |
| Monitoring (metrics)         | 9/10  | Excellent — Prometheus + Grafana ready |
| Monitoring (alerting + logs) | 7/10  | Structured logging + request correlation done; alerting rules still needed |
| Security                     | 4/10  | No rate limiting, open CORS, no plugin sandboxing |
| Resilience                   | 5/10  | Good shutdown, weak self-healing |
| Testing                      | 3/10  | Config tests only, no integration/API tests |
| DevOps Experience            | 5/10  | Good docs, no K8s/Helm, no runbooks |
| Update/Deploy Strategy       | 5/10  | Additive deploys supported, no rolling updates for existing models |

### Critical items before production

- Rate limiting per user/model
- Detailed readiness/liveness probes (current `/health` is a no-op)
- Integration and API test coverage
- Kubernetes manifests and Helm chart
- Prometheus alerting rules and SLO definitions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the dev environment, code style, and submitting pull requests.
