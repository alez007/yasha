# Yasha

A self-hosted, multi-model AI inference server that runs multiple LLMs and specialized models (TTS, ASR, embeddings) concurrently on a single GPU. Built on [VLLm](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray), it exposes an OpenAI-compatible API for drop-in integration with existing applications.

## Features

- **Multi-model on a single GPU** — run chat, embedding, speech-to-text, and text-to-speech models simultaneously with tunable per-model GPU memory allocation
- **OpenAI-compatible API** — works with any OpenAI SDK client out of the box
- **Streaming support** — SSE streaming for chat completions and TTS audio
- **Tool/function calling** — auto tool choice with configurable parsers for supported LLMs
- **TTS plugin system** — pluggable text-to-speech backends (Kokoro, Bark, Orpheus) with voice selection, speed control, and multiple output formats (MP3, Opus, AAC, FLAC, WAV, PCM)
- **YAML-based model configuration** — declare models, usecases, engine options, and memory budgets in a single config file
- **Multi-GPU support** — assign models to specific GPUs via `use_gpu`
- **Ray dashboard** — monitor model deployments, resource usage, and request logs via the built-in web UI
- **Wyoming protocol integration** — use as a voice backend for Home Assistant

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
| Kokoro | ONNX Runtime (GPU) | Yes (SSE) | Lightweight, auto-downloads model files |
| Bark | HuggingFace Transformers | No (WAV only) | Voice cloning capability |
| Orpheus | VLLm | Yes (SSE) | 4-bit quantized, token-streaming via SNAC decoder |

## Requirements

- NVIDIA GPU — no strict minimum VRAM, but **16 GB+** is recommended for a full conversational stack (LLM + TTS + STT + embeddings). **8 GB** is enough for simpler Home Assistant usage with smaller models
- Docker with NVIDIA Container Toolkit
- HuggingFace token (for gated models)

## Quick start

1. Clone the project
2. Copy `.env.example` to `.env` and set your `HF_TOKEN`
3. Configure models in `config/models.yaml`

### Development

Build the image:

```bash
docker build -t yasha_dev -f Dockerfile.dev .
```

Run the container:

```bash
docker run -it --rm --shm-size=8g --env-file .env --gpus all \
  --mount type=bind,src=./,dst=/yasha \
  -p 8265:8265 -p 8000:8000 yasha_dev
```

Inside the container, start the server:

```bash
serve run --route-prefix=/ --name='yasha api' --working-dir=/yasha --address=0.0.0.0:${RAY_REDIS_PORT} start:app
```

- Ray dashboard: `http://localhost:8265`
- OpenAI API: `http://localhost:8000`

Use the VS Code Dev Containers extension (or similar) to attach to the running container for development.

### Production

```bash
docker build -t yasha -f Dockerfile.prod .
docker run --rm --shm-size=8g --env-file .env --gpus all \
  -p 8265:8265 -p 8000:8000 yasha
```

## Model configuration

Models are declared in `config/models.yaml`:

```yaml
models:
  - name: "kokoro"
    model: "hexgrad/Kokoro-82M"
    usecase: "tts"
    plugin: "kokoro"

  - name: "whisper"
    model: "openai/whisper-small"
    usecase: "transcription"
    vllm_engine_kwargs:
      gpu_memory_utilization: 0.15

  - name: "nomic-embed-text-v1.5"
    model: "nomic-ai/nomic-embed-text-v1.5"
    usecase: "embed"
    vllm_engine_kwargs:
      gpu_memory_utilization: 0.05

  - name: "llama-3.1"
    model: "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    usecase: "generate"
    use_gpu: 1
    vllm_engine_kwargs:
      max_model_len: 32000
      gpu_memory_utilization: 0.9
      enable_auto_tool_choice: 1
```

Key options per model:

- `usecase` — `generate`, `embed`, `transcription`, `translation`, or `tts`
- `plugin` — TTS plugin name (`kokoro`, `bark`, `orpheus`); omit for VLLm inference
- `use_gpu` — GPU device index (default `0`)
- `vllm_engine_kwargs` — VLLm engine settings: `gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`, `dtype`, `quantization`, `kv_cache_dtype`, `enable_auto_tool_choice`, `tool_call_parser`, etc. See [VLLm engine args reference](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- `transformers_config` — Transformers fallback settings (`device`, etc.)

Additional TTS plugins (Bark, Orpheus) are available but commented out in the default config. Uncomment and adjust memory settings to enable them.

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
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
