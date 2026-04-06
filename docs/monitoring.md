# Monitoring

Yasha exposes Prometheus metrics through a single port via Ray's metrics agent. When enabled, all metrics — Ray cluster, Ray Serve, vLLM engine, and custom Yasha metrics — are available on one scrape endpoint.

## Architecture

```
Prometheus  ──scrape──>  Ray Metrics Agent (:8079)
                              |
                              |-- ray_*          Ray cluster: GPU, CPU, memory, actors
                              |-- serve_*        Ray Serve: HTTP requests, latency, replicas
                              |-- vllm:*         vLLM engine: KV cache, TTFT, tokens, queue
                              |-- yasha:*        Custom: per-model latency, errors, load time
```

## Enabling Metrics

Metrics are disabled by default. Set `YASHA_METRICS=true` to enable:

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
| `YASHA_METRICS` | `false` | Master toggle. Enables all metrics and the Ray metrics export port. |
| `RAY_METRICS_EXPORT_PORT` | `8079` | Port for the Ray metrics agent (only active when `YASHA_METRICS=true`). |

When `YASHA_METRICS=false`, no metrics are collected and port 8079 is not exposed. Zero overhead.

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
| **Overview** | Request rate, error rate, in-flight requests, models loaded, client disconnects | `yasha:*` |
| **Latency** | Gateway P50/P95/P99, per-model latency, per-usecase latency (generate, TTS, image, STT, embed) | `yasha:*` |
| **vLLM Engine** | KV cache usage, TTFT, inter-token latency, token throughput, queue depth, preemptions, prefix cache hit rate | `vllm:*` |
| **GPU & System** | GPU utilization, GPU memory, CPU, system memory | `ray_node_*` |
| **Ray Serve** | Replica health, processing queries, deployment latency, health check failures | `serve_*` |
| **Operational** | Model load time, load failures, resource cleanup errors, streaming chunks/s | `yasha:*` |

## Health Check

A health endpoint is always available regardless of the metrics toggle:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Yasha Metrics Reference

All custom metrics use the `yasha:` prefix and are exported via `ray.serve.metrics`.

### Gateway

| Metric | Type | Tags | Description |
|---|---|---|---|
| `yasha:request_total` | Counter | `model`, `endpoint`, `status` | Total requests by model and API method |
| `yasha:request_duration_seconds` | Histogram | `model`, `endpoint` | End-to-end request latency |
| `yasha:request_errors_total` | Counter | `model`, `endpoint`, `error_type` | Errors: `inference_error`, `stream_error`, `unhandled` |
| `yasha:request_in_progress` | Gauge | `model`, `endpoint` | Currently processing requests |
| `yasha:client_disconnects_total` | Counter | `model`, `endpoint` | Client disconnected before response completed |
| `yasha:stream_chunks_total` | Counter | `model` | Streaming chunks emitted |

### Model Deployment

| Metric | Type | Tags | Description |
|---|---|---|---|
| `yasha:model_load_duration_seconds` | Histogram | `model`, `loader` | Time to initialize a model |
| `yasha:model_load_failures_total` | Counter | `model`, `loader` | Failed model initializations |
| `yasha:models_loaded` | Gauge | | Number of loaded and ready models |

### Inference Timing

| Metric | Type | Tags | Description |
|---|---|---|---|
| `yasha:generation_duration_seconds` | Histogram | `model` | Chat/text generation latency |
| `yasha:tts_generation_duration_seconds` | Histogram | `model` | Text-to-speech latency |
| `yasha:image_generation_duration_seconds` | Histogram | `model` | Image generation latency |
| `yasha:transcription_duration_seconds` | Histogram | `model` | Speech-to-text latency |
| `yasha:embedding_duration_seconds` | Histogram | `model` | Embedding latency |

### Resource Cleanup

| Metric | Type | Tags | Description |
|---|---|---|---|
| `yasha:resource_cleanup_errors_total` | Counter | `model`, `component` | Errors during engine/model cleanup |

## Built-in Metrics from vLLM and Ray

These are automatically available when `YASHA_METRICS=true` — no additional configuration needed.

### vLLM (`vllm:*`)

Key metrics for LLM inference monitoring:

- `vllm:num_requests_running` / `vllm:num_requests_waiting` — queue depth
- `vllm:kv_cache_usage_perc` — KV cache utilization (0-1)
- `vllm:time_to_first_token_seconds` — TTFT histogram
- `vllm:inter_token_latency_seconds` — ITL histogram
- `vllm:e2e_request_latency_seconds` — end-to-end latency histogram
- `vllm:request_queue_time_seconds` — time spent waiting in queue
- `vllm:prompt_tokens` / `vllm:generation_tokens` — token throughput counters
- `vllm:num_preemptions` — memory pressure signal
- `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` — cache efficiency

Full reference: [vLLM Metrics Documentation](https://docs.vllm.ai/en/stable/design/metrics/)

### Ray Serve (`serve_*`)

- `serve_num_http_requests` — request count by route, method, status
- `serve_http_request_latency_ms` — request latency histogram
- `serve_num_ongoing_http_requests` — in-flight requests
- `serve_deployment_processing_latency_ms` — per-replica processing time
- `serve_deployment_replica_health_check` — replica health status

Full reference: [Ray Serve Monitoring](https://docs.ray.io/en/latest/serve/monitoring.html)

### Ray Cluster (`ray_*`)

- `ray_node_gpus_utilization` — GPU utilization by device
- `ray_node_gram_used` / `ray_node_gram_available` — GPU memory
- `ray_node_cpu_utilization` — CPU usage
- `ray_node_mem_used` / `ray_node_mem_total` — system memory

Full reference: [Ray Metrics](https://docs.ray.io/en/latest/cluster/metrics.html)
