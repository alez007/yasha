# Yasha

Self-hosted, multi-model AI inference server. Runs LLMs alongside specialized models (TTS, speech-to-text, embeddings) on a single GPU, exposing an OpenAI-compatible API. Built on [VLLm](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

## Features

- **Multi-model on a single GPU** — run chat, embedding, STT, and TTS models simultaneously with tunable per-model GPU memory allocation
- **Per-model isolated deployments** — each model runs in its own Ray Serve deployment with independent lifecycle, health checks, and failure isolation
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK client
- **Streaming** — SSE streaming for chat completions and TTS audio
- **Tool/function calling** — auto tool choice with configurable parsers
- **TTS plugin system** — pluggable backends (Kokoro, Bark, Orpheus) with voice selection, speed control, dual streaming modes, and multiple output formats (MP3, Opus, AAC, FLAC, WAV, PCM)
- **Multi-GPU support** — assign models to specific GPUs by index (`use_gpu: int`) or by named Ray resource (`use_gpu: str`), with full tensor parallelism support
- **Client disconnect detection** — cancels in-flight inference when the client disconnects, freeing GPU resources immediately
- **Ray dashboard** — monitor deployments, resources, and request logs
- **Wyoming protocol** — use as a voice backend for Home Assistant

## Supported usecases

| Usecase | Endpoint | Engine | Example model |
|---|---|---|---|
| Chat / text generation | `POST /v1/chat/completions` | VLLm | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` |
| Embeddings | `POST /v1/embeddings` | VLLm | `nomic-ai/nomic-embed-text-v1.5` |
| Speech-to-text | `POST /v1/audio/transcriptions` | VLLm | `openai/whisper-small` |
| Audio translation | `POST /v1/audio/translations` | VLLm | `openai/whisper-small` |
| Text-to-speech | `POST /v1/audio/speech` | Plugin | `hexgrad/Kokoro-82M`, `suno/bark-small`, `unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit` |

### TTS plugins

| Plugin | Backend | Streaming | Notes |
|---|---|---|---|
| Kokoro | ONNX Runtime (GPU) | Yes | Lightweight, auto-downloads model files |
| Bark | HuggingFace Transformers | No (WAV only) | Voice cloning capability |
| Orpheus | VLLm | Yes | 4-bit quantized, token-streaming via SNAC decoder |

#### TTS streaming modes

The `/v1/audio/speech` endpoint supports two streaming modes via the `stream_format` field:

- `stream_format: "sse"` — server-sent events; each chunk is a base64-encoded audio fragment, compatible with streaming clients
- `stream_format: "audio"` _(default)_ — returns a single WAV file after generation completes

## Requirements

- **NVIDIA GPU** — no strict minimum, but **16 GB+ VRAM** recommended for a full stack (LLM + TTS + STT + embeddings); **8 GB** is enough for lighter setups (e.g. Home Assistant with smaller models)
- **Docker** with NVIDIA Container Toolkit
- **HuggingFace token** (for gated models)

## Getting started

1. Clone the repository
2. Copy `.env.example` to `.env` and set your `HF_TOKEN`
3. Copy one of the example configs to `config/models.yaml` and adjust to your hardware — `config/` ships with ready-to-use presets for 8 GB, 16 GB, 24 GB, and 2×16 GB GPUs, or play with your own models and percentages

### Development

```bash
# Build
docker build -t yasha_dev -f Dockerfile.dev .

# Run (mounts local source for live editing)
docker run -it --rm --shm-size=8g --env-file .env --gpus all \
  --mount type=bind,src=./,dst=/yasha \
  -p 8265:8265 -p 8000:8000 yasha_dev

# Inside the container, start the server
serve run --route-prefix=/ --name='yasha api' --working-dir=/yasha --address=0.0.0.0:${RAY_REDIS_PORT} start:app
```

- Ray dashboard: `http://localhost:8265`
- API: `http://localhost:8000`

Use VS Code Dev Containers (or similar) to attach to the running container for development.

### Production

Pull a specific version from GHCR:

```bash
docker pull ghcr.io/alez007/yasha:0.1.0
```

Or use `latest` for the most recent release:

```bash
docker pull ghcr.io/alez007/yasha:latest
```

Mount your own `models.yaml` and a cache directory so models are only downloaded once:

```bash
docker run --rm --shm-size=8g --env-file .env --gpus all \
  -v ./models.yaml:/yasha/config/models.yaml \
  -v ./models-cache:/yasha/.cache/models \
  -p 8265:8265 -p 8000:8000 ghcr.io/alez007/yasha:0.1.0
```

The first startup downloads models from HuggingFace into the cache volume. Subsequent restarts reuse the cached models with no re-download.

The image ships with example configs for 8 GB, 16 GB, 24 GB, and 2×16 GB setups — inspect them with e.g. `docker run --rm ghcr.io/alez007/yasha:0.1.0 cat /yasha/config/models.example.16GB.yaml`.

To build locally instead:

```bash
docker build -t yasha -f Dockerfile.prod .
docker run --rm --shm-size=8g --env-file .env --gpus all \
  -v ./models.yaml:/yasha/config/models.yaml \
  -v ./models-cache:/yasha/.cache/models \
  -p 8265:8265 -p 8000:8000 yasha
```

## Model configuration

Each entry in `models.yaml` configures one model deployment. All fields:

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests |
| `model` | string | HuggingFace model ID |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, or `tts` |
| `loader` | string | `vllm`, `transformers`, or `custom` |
| `plugin` | string | Plugin name (required when `loader: custom`) |
| `num_gpus` | float | Fraction of a GPU to allocate (0.0–1.0); also sets VLLm `gpu_memory_utilization` |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `use_gpu` | int \| string | Pin to a specific GPU (see below) |
| `vllm_engine_kwargs` | object | Passed directly to the VLLm engine (see [VLLm engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html)) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |

### GPU pinning

`use_gpu` controls how a model is pinned to specific hardware:

- **`use_gpu: <int>`** — pins via `CUDA_VISIBLE_DEVICES`; only compatible with `tensor_parallel_size: 1`
- **`use_gpu: "<resource-name>"`** — requests a named Ray custom resource; compatible with tensor parallelism; requires registering the resource when starting the Ray head:
  ```bash
  ray start --head --resources='{"dual_16gb": 1}'
  ```
  The name is arbitrary — it just has to match the value in `use_gpu`. The `models.example.2x16GB.yaml` preset uses `"dual_16gb"` for the TP=2 LLM deployment.
- **omit** — Ray schedules the deployment freely across available GPUs

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
| `YASHA_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/yasha/.cache/models` |
| `RAY_REDIS_PORT` | Ray GCS server port | `6379` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | CPUs allocated to Ray head | `2` |
| `RAY_HEAD_GPU_NUM` | GPUs allocated to Ray head | `2` |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use VLLm v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.9.1` |

See `.env.example` for the full list.

## Wyoming protocol integration

Yasha can serve as a voice backend for [Home Assistant](https://www.home-assistant.io/) via the [Wyoming OpenAI](https://github.com/roryeckel/wyoming_openai) bridge:

```bash
docker run -it -p 10300:10300 \
  -e WYOMING_URI="tcp://0.0.0.0:10300" \
  -e WYOMING_LOG_LEVEL="INFO" \
  -e WYOMING_LANGUAGES="en" \
  -e STT_OPENAI_URL="http://host.docker.internal:8000/v1" \
  -e STT_MODELS="whisper" \
  -e STT_STREAMING_MODELS="whisper" \
  -e STT_BACKEND="OPENAI" \
  -e TTS_OPENAI_URL="http://host.docker.internal:8000/v1" \
  -e TTS_MODELS="kokoro" \
  -e TTS_STREAMING_MODELS="kokoro" \
  -e TTS_VOICES="af_heart" \
  -e TTS_SPEED="1.0" \
  -e TTS_BACKEND="OPENAI" \
  ghcr.io/roryeckel/wyoming_openai:latest
```
