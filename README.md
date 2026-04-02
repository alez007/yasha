# Yasha

Self-hosted, multi-model AI inference server. Runs LLMs alongside specialized models (TTS, speech-to-text, embeddings) on a single GPU, exposing an OpenAI-compatible API. Built on [vLLM](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

## Requirements

- **NVIDIA GPU** — 16 GB+ VRAM recommended for a full stack (LLM + TTS + STT + embeddings); 8 GB is sufficient for lighter setups
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **HuggingFace token** for gated models

## Features

- **Multi-model on a single GPU** — run chat, embedding, STT, and TTS models simultaneously with tunable per-model GPU memory allocation
- **Per-model isolated deployments** — each model runs in its own Ray Serve deployment with independent lifecycle, health checks, and failure isolation
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK client
- **Streaming** — SSE streaming for chat completions and TTS audio
- **Tool/function calling** — auto tool choice with configurable parsers
- **Plugin system** — opt-in TTS backends installed as isolated uv workspace packages
- **Multi-GPU support** — assign models to specific GPUs by index or named Ray resource, with full tensor parallelism support
- **Client disconnect detection** — cancels in-flight inference when the client disconnects, freeing GPU resources immediately
- **Ray dashboard** — monitor deployments, resources, and request logs

## Supported OpenAI Endpoints

| Endpoint | Usecase |
|---|---|
| `POST /v1/chat/completions` | Chat / text generation (streaming and non-streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text |
| `POST /v1/audio/translations` | Audio translation |
| `POST /v1/audio/speech` | Text-to-speech (SSE streaming or single-response) |
| `GET /v1/models` | List available models |

## Plugin Support

Yasha's TTS system is built around a plugin architecture — each TTS backend is an opt-in package with its own isolated dependencies. Plugins ship inside this repo (`plugins/`) or can be installed from PyPI.

To enable plugins, pass them as extras at sync time:

```bash
uv sync --extra kokoro
uv sync --extra kokoro --extra orpheus  # multiple plugins
```

When using Docker, set the `YASHA_PLUGINS` environment variable:

```
YASHA_PLUGINS=kokoro,orpheus
```

For a full guide on writing your own plugin, see [Plugin Development](docs/plugins.md).

## Getting Started

Pull the latest image from GHCR:

```bash
docker pull ghcr.io/alez007/yasha:latest
```

Grab an example config for your GPU and edit it to your liking:

```bash
docker run --rm ghcr.io/alez007/yasha:latest cat /yasha/config/models.example.16GB.yaml > models.yaml
```

Start the server, mounting your config and a cache directory so models are only downloaded once:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -e YASHA_PLUGINS=your_plugins_here \
  -v ./models.yaml:/yasha/config/models.yaml \
  -v ./models-cache:/yasha/.cache/models \
  -p 8265:8265 -p 8000:8000 ghcr.io/alez007/yasha:latest
```

- API: `http://localhost:8000`
- Ray dashboard: `http://localhost:8265`

Example configs are included for 8 GB, 16 GB, 24 GB, and 2×16 GB GPU setups.

## Development

See [Development](docs/development.md) for instructions on building the dev image, running with live source mounting, and attaching to the container.

## Model Configuration

See [Model Configuration](docs/model-configuration.md) for a full reference of all `models.yaml` fields, GPU pinning options, and environment variables.

## Home Assistant Integration

Yasha can serve as a voice backend for [Home Assistant](https://www.home-assistant.io/) via the Wyoming protocol. See [Home Assistant Integration](docs/home-assistant.md) for setup instructions.
