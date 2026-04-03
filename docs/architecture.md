# Architecture

## Overview

Yasha is built on two core technologies:
- **[Ray Serve](https://docs.ray.io/en/latest/serve/)** — manages model deployments as isolated actors with independent scaling and failure handling
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM inference engine with continuous batching and PagedAttention

A **FastAPI gateway** sits in front, exposing an OpenAI-compatible API that routes requests to the appropriate model deployment.

## Request Lifecycle

1. Client sends a request to the FastAPI gateway (e.g. `POST /v1/chat/completions`)
2. The gateway identifies the target model from the request body
3. A `RequestWatcher` begins monitoring the client connection for disconnects
4. The request is forwarded to the model's Ray Serve deployment via a `DisconnectProxy` (serializable headers + cancellation event)
5. The model deployment runs inference (vLLM, transformers, or plugin)
6. Response streams back as JSON or SSE
7. If the client disconnects mid-inference, the watcher fires the cancellation event, freeing GPU resources immediately

## Model Deployments

Each model in `models.yaml` becomes an isolated Ray Serve deployment (`ModelDeployment` actor). This gives:

- **Independent lifecycle** — one model crashing doesn't affect others
- **Per-model GPU budgeting** — `num_gpus` controls VRAM allocation (e.g. 0.70 for 70%)
- **Sequential startup** — models deploy one at a time to prevent memory spikes, ordered by tensor parallelism size (TP > 1 first)

### Inference Loaders

Each deployment uses one of three loaders:

| Loader | Backend | Use cases |
|--------|---------|-----------|
| `vllm` | vLLM engine | Chat/generation, embeddings, transcription, translation |
| `transformers` | PyTorch + HuggingFace | Custom model implementations |
| `custom` | Plugin system | TTS backends (Kokoro, Bark, Orpheus) |

## GPU Allocation

Models can be assigned to GPUs in three ways:

| Config | Behavior |
|--------|----------|
| `use_gpu: 0` | Pinned to GPU index 0 via `CUDA_VISIBLE_DEVICES` (TP=1 only) |
| `use_gpu: "dual_16gb"` | Assigned via named Ray resource (supports TP > 1) |
| *(omitted)* | Ray schedules freely across available GPUs |

## Plugin System

TTS backends are isolated `uv` workspace packages under `plugins/`. Each plugin:

- Implements `BasePlugin` with `start()` and `generate()` methods
- Has its own dependencies, isolated from the main project
- Is opt-in via `uv sync --extra <plugin>` or the `YASHA_PLUGINS` env var
- Returns audio as a single response or as an SSE async generator

See [Plugin Development](plugins.md) for details.

## Key Files

| File | Purpose |
|------|---------|
| `start.py` | Entry point — initializes Ray, deploys models |
| `yasha/openai/api.py` | FastAPI gateway with OpenAI endpoints |
| `yasha/infer/model_deployment.py` | Ray Serve deployment actor |
| `yasha/infer/infer_config.py` | Pydantic config models and protocols |
| `yasha/infer/vllm/vllm_infer.py` | vLLM engine wrapper |
| `yasha/plugins/base_plugin.py` | Plugin base classes |
| `config/models.yaml` | Model configuration |
