# Model Configuration

Models are configured in a YAML file (default: `config/models.yaml`). Each entry defines one deployment.

## CLI Options

`start.py` accepts the following arguments (env vars work as fallbacks):

| Argument | Env Var | Default | Description |
|---|---|---|---|
| `--config` | — | `config/models.yaml` | Path to models config file |
| `--gateway-name` | `MSHIP_GATEWAY_NAME` | `modelship api` | Name for the API gateway app |
| `--ray-cluster-address` | `RAY_CLUSTER_ADDRESS` | — | Ray cluster address |
| `--ray-redis-port` | `RAY_REDIS_PORT` | — | Ray Redis port |
| `--use-existing-ray-cluster` | `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Connect to an existing Ray cluster |
| `--redeploy` | — | `false` | Tear down all existing deployments before deploying |
| `--cache-dir` | `MSHIP_CACHE_DIR` | `/.cache` | Base cache directory |
| `--log-level` | `MSHIP_LOG_LEVEL` | `INFO` | Log level |
| `--log-format` | `MSHIP_LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `--log-target` | `MSHIP_LOG_TARGET` | `console` | Log target: `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) |
| `--otel-endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OpenTelemetry OTLP endpoint (e.g. `http://collector:4317`) |
| `--no-metrics` | `MSHIP_METRICS` | enabled | Disable Prometheus metrics |
| `--api-keys` | `MSHIP_API_KEYS` | — | Comma-separated API keys |
| `--max-request-body-bytes` | `MSHIP_MAX_REQUEST_BODY_BYTES` | `52428800` | Max request body size in bytes |

### Cache Directory Structure

The base cache directory (`MSHIP_CACHE_DIR`, default: `/.cache`) is organized into the following subdirectories:

- `{base_cache}/huggingface`: HuggingFace models and tokenizers (via `HF_HOME`).
- `{base_cache}/vllm`: vLLM-specific compiled artifacts and caches (via `VLLM_CACHE_ROOT`).
- `{base_cache}/flashinfer`: FlashInfer kernels (via `FLASHINFER_CACHE_DIR`).
- `{base_cache}/plugins`: Downloaded weights and artifacts used by custom plugins.

### Additive Deploys

By default, `start.py` adds models to a running cluster without disrupting existing deployments. This allows incremental composition:

```bash
# Deploy LLM models
python start.py --config config/llm.yaml

# Later, add TTS without touching the running LLMs
python start.py --config config/tts.yaml

# Add more models from another config
python start.py --config config/embeddings.yaml
```

Use `--redeploy` to tear down everything and start fresh:

```bash
python start.py --config config/models.yaml --redeploy
```

Multiple gateways can run independently by using `--gateway-name`:

```bash
python start.py --config config/llm.yaml --gateway-name "llm-api"
python start.py --config config/tts.yaml --gateway-name "tts-api"
```

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests |
| `model` | string | HuggingFace model ID |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, `tts`, or `image` |
| `loader` | string | `vllm`, `transformers`, `diffusers`, or `custom` |
| `plugin` | string | Plugin module name (required when `loader: custom`); must be installed via `uv sync --extra <plugin>` |
| `num_gpus` | float | Fraction of a GPU to allocate (0.0-1.0); also sets vLLM `gpu_memory_utilization` |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `num_replicas` | int | Number of identical Ray Serve replicas for this deployment (default `1`) |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine (see below) |
| `transformers_config` | object | Transformers loader options (see below) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |

## vLLM Loader

The `vllm` loader supports chat/generation, embeddings, transcription, and translation. Configuration is passed via `vllm_engine_kwargs`:

| Field | Type | Default | Description |
|---|---|---|---|
| `tensor_parallel_size` | int | `1` | Number of GPUs for tensor parallelism |
| `max_model_len` | int | auto | Maximum sequence length |
| `dtype` | string | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `tokenizer` | string | model default | Custom tokenizer path |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `gpu_memory_utilization` | float | `0.9` | VRAM fraction (overridden by `num_gpus` when set) |
| `distributed_executor_backend` | string | auto | `ray` or `mp` for multi-GPU |
| `quantization` | string | — | Quantization method (e.g. `awq`, `gptq`) |
| `enable_auto_tool_choice` | bool | — | Enable automatic tool/function calling |
| `tool_call_parser` | string | — | Tool call parser (e.g. `llama3_json`, `hermes`) |
| `enforce_eager` | bool | — | Disable CUDA graph capture |
| `kv_cache_dtype` | string | — | KV cache dtype (e.g. `fp8`) |

### Chat / Text Generation

```yaml
models:
  - name: qwen
    model: Qwen/Qwen3-0.6B
    usecase: generate
    loader: vllm
    num_gpus: 0.30
    vllm_engine_kwargs:
      max_model_len: 8192
```

### LLM with Tool Calling

```yaml
models:
  - name: llama
    model: meta-llama/Llama-3.1-8B-Instruct
    usecase: generate
    loader: vllm
    num_gpus: 0.70
    vllm_engine_kwargs:
      enable_auto_tool_choice: true
      tool_call_parser: llama3_json
```

### Multi-GPU with Tensor Parallelism

```yaml
models:
  - name: llama-70b
    model: meta-llama/Llama-3.1-70B-Instruct
    usecase: generate
    loader: vllm
    vllm_engine_kwargs:
      tensor_parallel_size: 2
      distributed_executor_backend: ray
```

### Embeddings

```yaml
models:
  - name: nomic-embed
    model: nomic-ai/nomic-embed-text-v1.5
    usecase: embed
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true
```

### Speech-to-Text (Whisper)

```yaml
models:
  - name: whisper
    model: openai/whisper-small
    usecase: transcription
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true
```

## Transformers Loader

The `transformers` loader uses PyTorch with HuggingFace Transformers. Supports chat/generation, embeddings, transcription, translation, and TTS. Unlike the vLLM loader, it can run entirely on CPU — making it ideal for smaller models, development, or environments without a GPU.

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | string | `cpu` | Device to run on (`cpu`, `cuda`, `cuda:0`, etc.) |
| `torch_dtype` | string | `auto` | Model dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `model_kwargs` | object | `{}` | Extra keyword arguments passed to the model constructor |
| `pipeline_kwargs` | object | `{}` | Extra keyword arguments passed to the pipeline at inference time |

### Chat / Text Generation (CPU)

```yaml
models:
  - name: qwen
    model: Qwen/Qwen3-0.6B
    usecase: generate
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### Speech-to-Text (CPU)

Audio is automatically decoded and resampled to the model's expected sample rate (e.g. 16kHz for Whisper).

```yaml
models:
  - name: whisper
    model: openai/whisper-small
    usecase: transcription
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### Embeddings (CPU)

Uses `sentence-transformers` under the hood.

```yaml
models:
  - name: embeddings
    model: sentence-transformers/all-MiniLM-L6-v2
    usecase: embed
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### TTS (GPU)

```yaml
models:
  - name: my-tts
    model: some-org/some-tts-model
    usecase: tts
    loader: transformers
    num_gpus: 0.20
    transformers_config:
      device: "cuda:0"
```

## Diffusers Loader

The `diffusers` loader uses HuggingFace Diffusers for image generation. Any model supported by `AutoPipelineForText2Image` works out of the box.

| Field | Type | Default | Description |
|---|---|---|---|
| `torch_dtype` | string | `float16` | Torch dtype (`float16`, `bfloat16`, `float32`) |
| `num_inference_steps` | int | `30` | Default denoising steps (can be overridden per request) |
| `guidance_scale` | float | `7.5` | Default classifier-free guidance scale (can be overridden per request) |

```yaml
models:
  - name: sdxl-turbo
    model: stabilityai/sdxl-turbo
    usecase: image
    loader: diffusers
    num_gpus: 0.35
    diffusers_config:
      torch_dtype: "float16"
      num_inference_steps: 4
      guidance_scale: 0.0
```

## Custom Loader (Plugins)

The `custom` loader delegates to a plugin module. The `plugin` field is required and must match an installed plugin package. Plugin-specific options are passed via `plugin_config`.

See each plugin's README for configuration details:
- [Kokoro TTS](../plugins/kokoro/README.md)
- [Bark TTS](../plugins/bark/README.md)
- [Orpheus TTS](../plugins/orpheus/README.md)

For writing your own plugin, see [Plugin Development](plugins.md).

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
| `MSHIP_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/.cache` |
| `MSHIP_GATEWAY_NAME` | Name for the API gateway app | `modelship api` |
| `MSHIP_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `MSHIP_LOG_TARGET` | Log target: `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) | `console` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP endpoint for log export (e.g. `http://collector:4317`). Requires `uv sync --extra otel`. | — |
| `CUDA_DEVICE_ORDER` | GPU enumeration order; set to `PCI_BUS_ID` for deterministic ordering in multi-GPU systems | `PCI_BUS_ID` |
| `RAY_REDIS_PORT` | Ray GCS server port | `6379` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | CPUs allocated to Ray head | `2` |
| `RAY_HEAD_GPU_NUM` | GPUs allocated to Ray head | `2` |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.8.1` |
