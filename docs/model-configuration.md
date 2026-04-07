# Model Configuration

Models are configured in `config/models.yaml`. Each entry defines one deployment.

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests |
| `model` | string | HuggingFace model ID |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, `tts`, or `image` |
| `loader` | string | `vllm`, `transformers`, `diffusers`, or `custom` |
| `plugin` | string | Plugin module name (required when `loader: custom`); must be installed via `uv sync --extra <plugin>` |
| `num_gpus` | float | Fraction of a GPU to allocate (0.0–1.0); also sets vLLM `gpu_memory_utilization` |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `use_gpu` | int \| string | Pin to a specific GPU (see below) |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine — see [vLLM engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |

## GPU Pinning

`use_gpu` controls how a model is pinned to specific hardware:

- **`use_gpu: <int>`** — pins via `CUDA_VISIBLE_DEVICES`; only compatible with `tensor_parallel_size: 1`
- **`use_gpu: "<resource-name>"`** — requests a named Ray custom resource; compatible with tensor parallelism; requires registering the resource when starting the Ray head:
  ```bash
  ray start --head --resources='{"dual_16gb": 1}'
  ```
  The name is arbitrary — it must match the value in `use_gpu`. The `models.example.2x16GB.yaml` preset uses `"dual_16gb"` for a TP=2 LLM deployment.
- **omit** — Ray schedules the deployment freely across available GPUs

## Diffusers Config

Options for `loader: diffusers` models (image generation via HuggingFace Diffusers):

| Field | Type | Default | Description |
|---|---|---|---|
| `torch_dtype` | string | `float16` | Torch dtype (`float16`, `bfloat16`, `float32`) |
| `num_inference_steps` | int | `30` | Default denoising steps (can be overridden per request) |
| `guidance_scale` | float | `7.5` | Default classifier-free guidance scale (can be overridden per request) |

Any model supported by `AutoPipelineForText2Image` works out of the box — Stable Diffusion 1.5/2.x/XL/3.x, SDXL Turbo, Flux, PixArt, Kandinsky, etc.

Example:

```yaml
- name: "sdxl-turbo"
  model: "stabilityai/sdxl-turbo"
  usecase: "image"
  loader: "diffusers"
  num_gpus: 0.35
  diffusers_config:
    torch_dtype: "float16"
    num_inference_steps: 4
    guidance_scale: 0.0
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
| `YASHA_PLUGINS` | Comma-separated list of plugins to install at startup (e.g. `kokoro,orpheus`) | — |
| `YASHA_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/yasha/.cache/models` |
| `YASHA_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `RAY_REDIS_PORT` | Ray GCS server port | `6379` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | CPUs allocated to Ray head | `2` |
| `RAY_HEAD_GPU_NUM` | GPUs allocated to Ray head | `2` |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.9.1` |
