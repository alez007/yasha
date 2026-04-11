# Monitoring & Logging

Yasha exposes Prometheus metrics through a single port via Ray's metrics agent. When enabled, all metrics — Ray cluster, Ray Serve, vLLM engine, and custom Yasha metrics — are available on one scrape endpoint.

## Logging

Yasha uses a centralized logging system with structured output and request correlation. All application logs go through the `yasha.*` logger hierarchy, separate from library logs (Ray, vLLM, etc.).

### Configuration

| Env Var | Default | Description |
|---|---|---|
| `YASHA_LOG_LEVEL` | `INFO` | App log level. Set to `DEBUG` for full request/response bodies. Set to `TRACE` to also enable library debug logs. |
| `YASHA_LOG_FORMAT` | `text` | `text` for human-readable output, `json` for structured JSON lines (for log aggregation with ELK/Loki/Splunk). |

### Log Levels

| Level | App logs (`yasha.*`) | Library logs (Ray, vLLM, transformers) |
|---|---|---|
| `INFO` (default) | Startup, deployment, request summaries | `WARNING` only |
| `DEBUG` | Full request/response bodies, per-chunk details | `WARNING` only |
| `TRACE` | Same as `DEBUG` | `DEBUG` — all library internals |

### Request Correlation

Every API request is assigned a unique request ID that appears in all log lines for that request — both in the API gateway process and in the model deployment actor. This allows tracing a request end-to-end across Ray actor boundaries.

Text format example:
```
[2025-04-09 14:06:54] INFO     yasha.api [a1b2c3d4] | chat_completion model=llama messages=3 stream=True max_tokens=512
```

JSON format example:
```json
{"timestamp": "2025-04-09T14:06:54", "level": "INFO", "logger": "yasha.api", "message": "chat_completion model=llama messages=3 stream=True max_tokens=512", "request_id": "a1b2c3d4", "pid": 12345}
```

### Logger Names

| Logger | Scope |
|---|---|
| `yasha.startup` | Application initialization and shutdown |
| `yasha.api` | API gateway endpoints |
| `yasha.api.auth` | Authentication middleware |
| `yasha.infer` | Base inference layer |
| `yasha.infer.deployment` | Ray Serve model deployment actor |
| `yasha.infer.vllm` | vLLM inference backend |
| `yasha.infer.transformers` | Transformers inference backend |
| `yasha.infer.diffusers` | Diffusers inference backend |
| `yasha.infer.custom` | Custom/plugin inference backend |
| `yasha.plugin.<name>` | Individual plugins (kokoro, bark, orpheus) |

## Architecture

```
Prometheus  ──scrape──>  Ray Metrics Agent (:8079)
                              |
                              |-- ray_node_*          Ray cluster: GPU, CPU, memory
                              |-- ray_serve_*         Ray Serve: HTTP requests, latency, replicas
                              |-- ray_vllm_*          vLLM engine: KV cache, TTFT, tokens, queue
                              |-- ray_yasha_*         Custom: per-model latency, errors, load time
```

> **Note:** All metrics are prefixed with `ray_` by Ray's metrics agent. vLLM metric names are also sanitized (`:` → `_`), so e.g. the vLLM-native `vllm:kv_cache_usage_perc` becomes `ray_vllm_kv_cache_usage_perc`.

## Enabling Metrics

Metrics are enabled by default. Set `YASHA_METRICS=false` to disable:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token \
  -e YASHA_METRICS=true \
  -v ./models.yaml:/yasha/config/models.yaml \
  -p 8000:8000 -p 8079:8079 -p 8265:8265 \
  ghcr.io/alez007/yasha:latest
```

| Env Var | Default | Description |
|---|---|---|
| `YASHA_METRICS` | `true` | Master toggle. Enables all metrics and the Ray metrics export port. |
| `RAY_METRICS_EXPORT_PORT` | `8079` | Port for the Ray metrics agent (only active when `YASHA_METRICS=true`). |

Set `YASHA_METRICS=false` to disable all metrics collection. When disabled, port 8079 is not exposed and there is zero overhead.

## Connecting to Prometheus

Add Yasha as a scrape target in your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: yasha
    scrape_interval: 15s
    static_configs:
      - targets: ["<yasha-host>:8079"]
```

For multi-node Ray clusters, use Ray's auto-generated service discovery file instead of static targets:

```yaml
scrape_configs:
  - job_name: yasha
    file_sd_configs:
      - files: ["/tmp/ray/prom_metrics_service_discovery.json"]
```

## Connecting to Grafana

A pre-built Grafana dashboard is included at [`docs/grafana-dashboard.json`](grafana-dashboard.json).

To import it:

1. Open Grafana and go to **Dashboards > Import**
2. Upload `grafana-dashboard.json` or paste its contents
3. Select your Prometheus datasource when prompted

The dashboard has 6 rows:

| Row | What it shows | Metric sources |
|---|---|---|
| **Overview** | Request rate, error rate, in-flight requests, models loaded, client disconnects | `ray_yasha_*` |
| **Latency** | Gateway P50/P95/P99, per-model latency, per-usecase latency (generate, TTS, image, STT, embed) | `ray_yasha_*` |
| **vLLM Engine** | KV cache usage, TTFT, inter-token latency, token throughput, queue depth, preemptions, prefix cache hit rate | `ray_vllm_*` |
| **GPU & System** | GPU utilization, GPU memory, CPU, system memory | `ray_node_*` |
| **Ray Serve** | Health check latency, request count, deployment processing latency, HTTP request latency | `ray_serve_*` |
| **Operational** | Model load time, load failures, resource cleanup errors, streaming chunks/s | `ray_yasha_*` |

## Health Check

A health endpoint is always available regardless of the metrics toggle:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Yasha Metrics Reference

All custom metrics are defined via `ray.serve.metrics` and exported through Ray's metrics agent with a `ray_` prefix.

### Gateway

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_yasha_request_total` | Counter | `model`, `endpoint`, `status` | Total requests by model and API method |
| `ray_yasha_request_duration_seconds` | Histogram | `model`, `endpoint` | End-to-end request latency |
| `ray_yasha_request_errors_total` | Counter | `model`, `endpoint`, `error_type` | Errors: `inference_error`, `stream_error`, `unhandled` |
| `ray_yasha_request_in_progress` | Gauge | `model`, `endpoint` | Currently processing requests |
| `ray_yasha_client_disconnects_total` | Counter | `model`, `endpoint` | Client disconnected before response completed |
| `ray_yasha_stream_chunks_total` | Counter | `model` | Streaming chunks emitted |

### Model Deployment

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_yasha_model_load_duration_seconds` | Histogram | `model`, `loader` | Time to initialize a model |
| `ray_yasha_model_load_failures_total` | Counter | `model`, `loader` | Failed model initializations |
| `ray_yasha_models_loaded` | Gauge | | Number of loaded and ready models |

### Inference Timing

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_yasha_generation_duration_seconds` | Histogram | `model` | Chat/text generation latency |
| `ray_yasha_tts_generation_duration_seconds` | Histogram | `model` | Text-to-speech latency |
| `ray_yasha_image_generation_duration_seconds` | Histogram | `model` | Image generation latency |
| `ray_yasha_transcription_duration_seconds` | Histogram | `model` | Speech-to-text latency |
| `ray_yasha_embedding_duration_seconds` | Histogram | `model` | Embedding latency |

### Resource Cleanup

| Metric | Type | Tags | Description |
|---|---|---|---|
| `ray_yasha_resource_cleanup_errors_total` | Counter | `model`, `component` | Errors during engine/model cleanup |

## Built-in Metrics from vLLM and Ray

These are automatically available when `YASHA_METRICS=true` — no additional configuration needed.

### vLLM (`ray_vllm_*`)

vLLM metrics are routed through Ray's metrics agent via `RayPrometheusStatLogger`. The native `vllm:` prefix is sanitized to `ray_vllm_`.

- `ray_vllm_num_requests_running` / `ray_vllm_num_requests_waiting` — queue depth
- `ray_vllm_kv_cache_usage_perc` — KV cache utilization (0-1)
- `ray_vllm_time_to_first_token_seconds` — TTFT histogram
- `ray_vllm_inter_token_latency_seconds` — ITL histogram
- `ray_vllm_e2e_request_latency_seconds` — end-to-end latency histogram
- `ray_vllm_request_queue_time_seconds` — time spent waiting in queue
- `ray_vllm_prompt_tokens_total` / `ray_vllm_generation_tokens_total` — token throughput counters
- `ray_vllm_num_preemptions_total` — memory pressure signal
- `ray_vllm_prefix_cache_hits_total` / `ray_vllm_prefix_cache_queries_total` — cache efficiency

Full reference: [vLLM Metrics Documentation](https://docs.vllm.ai/en/stable/design/metrics/)

### Ray Serve (`ray_serve_*`)

- `ray_serve_num_http_requests_total` — request count by route, method, status
- `ray_serve_http_request_latency_ms` — request latency histogram
- `ray_serve_handle_request_counter_total` — request count by deployment
- `ray_serve_deployment_processing_latency_ms` — per-replica processing time
- `ray_serve_health_check_latency_ms` — health check latency histogram

Full reference: [Ray Serve Monitoring](https://docs.ray.io/en/latest/serve/monitoring.html)

### Ray Cluster (`ray_*`)

- `ray_node_gpus_utilization` — GPU utilization by device
- `ray_node_gram_used` / `ray_node_gram_available` — GPU memory
- `ray_node_cpu_utilization` — CPU usage
- `ray_node_mem_used` / `ray_node_mem_total` — system memory

Full reference: [Ray Metrics](https://docs.ray.io/en/latest/cluster/metrics.html)
