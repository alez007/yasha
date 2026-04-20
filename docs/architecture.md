# Architecture

## Overview

Modelship is built on [Ray Serve](https://docs.ray.io/en/latest/serve/) for deployment orchestration and a **FastAPI gateway** that exposes an OpenAI-compatible API. Multiple inference backends are supported:

- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput GPU inference with continuous batching and PagedAttention
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — CPU and lightweight GPU inference for chat, embeddings, transcription, and TTS
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — image generation via `AutoPipelineForText2Image`
- **Plugin system** — custom TTS and STT backends (Kokoro ONNX, Bark, Orpheus, whisper.cpp)

## Request Lifecycle

1. Client sends a request to the FastAPI gateway (e.g. `POST /v1/chat/completions`)
2. The gateway identifies the target model from the request body
3. A `RequestWatcher` begins monitoring the client connection for disconnects
4. The request is forwarded to the model's Ray Serve deployment via a `RawRequestProxy` (serializable headers + cancellation event)
5. The model deployment runs inference (vLLM, transformers, or plugin)
6. Response streams back as JSON or SSE
7. If the client disconnects mid-inference, the watcher fires the cancellation event, freeing GPU resources immediately

## Model Deployments

Each model in `models.yaml` becomes an isolated Ray Serve deployment (`ModelDeployment` actor). This gives:

- **Independent lifecycle** — one model crashing doesn't affect others
- **Per-model GPU budgeting** — `num_gpus` controls VRAM allocation (e.g. 0.70 for 70%)
- **Sequential startup** — models deploy one at a time to prevent memory spikes, ordered by tensor parallelism size (TP > 1 first)
- **Additive deploys** — by default, `start.py` adds models to a running cluster without disrupting existing deployments, enabling incremental composition from multiple config files. Use `--redeploy` to tear down and start fresh
- **Multi-deployment routing** — the same model name can appear multiple times with different configs (e.g. GPU + CPU). The gateway round-robins requests across all deployments sharing a name. Each deployment also supports `num_replicas` for scaling identical copies via Ray Serve's built-in load balancing
- **Multi-gateway support** — multiple independent gateways can run on the same cluster via `--gateway-name`, each managing its own set of models

### Inference Loaders

Each deployment uses one of the following loaders:

| Loader | Backend | Use cases | GPU required |
|--------|---------|-----------|--------------|
| `vllm` | vLLM engine | Chat/generation, embeddings, transcription, translation | Yes |
| `llama_cpp` | llama-cpp-python | Chat/generation, embeddings (GGUF models) | No — currently CPU-only |
| `transformers` | PyTorch + HuggingFace | Chat/generation, embeddings, transcription, translation, TTS | No — runs on CPU or GPU |
| `diffusers` | HuggingFace Diffusers | Image generation (any `AutoPipelineForText2Image` model) | Yes |
| `custom` | Plugin system | TTS backends (Kokoro ONNX, Bark, Orpheus), STT backends (whisper.cpp) | No |

The `transformers` loader is ideal for CPU-only deployments, smaller models, or development/testing without a GPU. It uses HuggingFace `pipeline()` under the hood and handles audio resampling automatically for speech-to-text models. The `llama_cpp` loader provides high-efficiency inference for quantized GGUF models on CPU. The `vllm` loader provides higher throughput on GPU with continuous batching and PagedAttention.

## GPU Allocation

Ray automatically schedules model deployments across available GPUs based on the `num_gpus` fraction each model requests. For example, two models each requesting `num_gpus: 0.9` will be placed on separate GPUs.

## Plugin System

Custom backends are isolated `uv` workspace packages under `plugins/`. Each plugin:

- Implements `BasePlugin` and overrides the `create_*` method(s) matching its `usecase` (e.g. `create_speech` for TTS, `create_transcription` for STT)
- Has its own dependencies, isolated from the main project
- Is opt-in via `uv sync --extra <plugin>` or the `MSHIP_PLUGINS` env var
- Returns raw, protocol-agnostic outputs; OpenAI-shape adaptation is handled by the serving wrappers in `modelship/infer/custom/openai/`

See [Plugin Development](plugins.md) for details.

## Key Files

| File | Purpose |
|------|---------|
| `start.py` | Entry point — initializes Ray, deploys models additively (or fresh with `--redeploy`) |
| `modelship/openai/api.py` | FastAPI gateway with OpenAI endpoints |
| `modelship/infer/model_deployment.py` | Ray Serve deployment actor |
| `modelship/infer/infer_config.py` | Pydantic config models and protocols |
| `modelship/infer/vllm/vllm_infer.py` | vLLM engine wrapper |
| `modelship/infer/transformers/transformers_infer.py` | Transformers pipeline wrapper (CPU/GPU) |
| `modelship/infer/diffusers/diffusers_infer.py` | Diffusers pipeline wrapper |
| `modelship/plugins/base_plugin.py` | Plugin base classes |
| `config/models.yaml` | Model configuration |
