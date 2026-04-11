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
| `num_replicas` | int | Number of identical Ray Serve replicas for this deployment (default `1`) |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine — see [vLLM engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |

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

## Multi-Deployment Routing

You can run the same model on different hardware (e.g. GPU and CPU) by repeating the same `name` with different settings. The API exposes the model once under `/v1/models`, and round-robins requests across all deployments sharing that name.

Use `num_replicas` to scale identical copies of a single deployment (Ray Serve handles load balancing between replicas automatically).

```yaml
models:
  # GPU instance with 2 replicas
  - name: "kokoro"
    model: "hexgrad/Kokoro-82M"
    usecase: "tts"
    loader: "custom"
    plugin: "kokoro"
    num_gpus: 0.07
    num_replicas: 2
    plugin_config:
      onnx_provider: "CUDAExecutionProvider"

  # CPU fallback
  - name: "kokoro"
    model: "hexgrad/Kokoro-82M"
    usecase: "tts"
    loader: "custom"
    plugin: "kokoro"
    num_gpus: 0
    plugin_config:
      onnx_provider: "CPUExecutionProvider"
```

In this example, requests to model `kokoro` are distributed across three backends: two GPU replicas and one CPU instance.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
| `MSHIP_PLUGINS` | Comma-separated list of plugins to install at startup (e.g. `kokoro,orpheus`) | — |
| `MSHIP_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/modelship/.cache/models` |
| `MSHIP_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `CUDA_DEVICE_ORDER` | GPU enumeration order; set to `PCI_BUS_ID` for deterministic ordering in multi-GPU systems | `PCI_BUS_ID` |
| `RAY_REDIS_PORT` | Ray GCS server port | `6379` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | CPUs allocated to Ray head | `2` |
| `RAY_HEAD_GPU_NUM` | GPUs allocated to Ray head | `2` |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.9.1` |
