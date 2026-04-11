"""Modelship Prometheus metrics — all exported via Ray's metrics agent.

When MSHIP_METRICS=true, metrics are defined using ray.serve.metrics so they
flow through the same Ray metrics agent port as ray_*, serve_*, and vllm:*
metrics.  When disabled, no-op objects are exported so call sites need zero
conditional logic.
"""

import os

_ENABLED = os.environ.get("MSHIP_METRICS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# No-op metric stubs (used when metrics are disabled)
# ---------------------------------------------------------------------------


class _NoOpCounter:
    def inc(self, value=1.0, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


class _NoOpGauge:
    def set(self, value, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


class _NoOpHistogram:
    def observe(self, value, tags=None):
        pass

    def set_default_tags(self, tags):
        pass


# ---------------------------------------------------------------------------
# Latency bucket boundaries (in seconds)
# ---------------------------------------------------------------------------

_REQUEST_LATENCY_BOUNDARIES = [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]
_MODEL_LOAD_BOUNDARIES: list[float] = [1, 5, 10, 30, 60, 120, 300, 600]


def _build_metrics():
    """Construct real or no-op metric objects based on MSHIP_METRICS."""

    if not _ENABLED:
        return {
            # Gateway
            "request_total": _NoOpCounter(),
            "request_duration_seconds": _NoOpHistogram(),
            "request_errors_total": _NoOpCounter(),
            "request_in_progress": _NoOpGauge(),
            "client_disconnects_total": _NoOpCounter(),
            "stream_chunks_total": _NoOpCounter(),
            # Model deployment
            "model_load_duration_seconds": _NoOpHistogram(),
            "model_load_failures_total": _NoOpCounter(),
            "models_loaded": _NoOpGauge(),
            # Inference timing
            "generation_duration_seconds": _NoOpHistogram(),
            "tts_generation_duration_seconds": _NoOpHistogram(),
            "image_generation_duration_seconds": _NoOpHistogram(),
            "transcription_duration_seconds": _NoOpHistogram(),
            "embedding_duration_seconds": _NoOpHistogram(),
            # Resource cleanup
            "resource_cleanup_errors_total": _NoOpCounter(),
            "auth_failures_total": _NoOpCounter(),
        }

    from ray.serve.metrics import Counter, Gauge, Histogram

    # Ray's type stubs over-constrain tag_keys (Tuple[str] instead of variable-length
    # tuples) and boundaries (List[float] vs int literals). Suppressed with type: ignore.
    return {
        # -- Gateway layer --
        "request_total": Counter(
            "modelship_request_total",
            description="Total inference requests by model and endpoint.",
            tag_keys=("model", "endpoint", "status"),  # type: ignore[arg-type]
        ),
        "request_duration_seconds": Histogram(
            "modelship_request_duration_seconds",
            description="End-to-end request latency (gateway to response) in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model", "endpoint"),  # type: ignore[arg-type]
        ),
        "request_errors_total": Counter(
            "modelship_request_errors_total",
            description="Total inference errors by model, endpoint, and error type.",
            tag_keys=("model", "endpoint", "error_type"),  # type: ignore[arg-type]
        ),
        "request_in_progress": Gauge(
            "modelship_request_in_progress",
            description="Number of requests currently being processed per model.",
            tag_keys=("model", "endpoint"),  # type: ignore[arg-type]
        ),
        "client_disconnects_total": Counter(
            "modelship_client_disconnects_total",
            description="Total client disconnects during inference.",
            tag_keys=("model", "endpoint"),  # type: ignore[arg-type]
        ),
        "stream_chunks_total": Counter(
            "modelship_stream_chunks_total",
            description="Total streaming chunks emitted.",
            tag_keys=("model",),
        ),
        # -- Model deployment layer --
        "model_load_duration_seconds": Histogram(
            "modelship_model_load_duration_seconds",
            description="Model initialization time in seconds.",
            boundaries=_MODEL_LOAD_BOUNDARIES,
            tag_keys=("model", "loader"),  # type: ignore[arg-type]
        ),
        "model_load_failures_total": Counter(
            "modelship_model_load_failures_total",
            description="Total failed model deployments.",
            tag_keys=("model", "loader"),  # type: ignore[arg-type]
        ),
        "models_loaded": Gauge(
            "modelship_models_loaded",
            description="Number of models currently loaded.",
        ),
        # -- Inference timing --
        "generation_duration_seconds": Histogram(
            "modelship_generation_duration_seconds",
            description="Chat/text generation latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model",),
        ),
        "tts_generation_duration_seconds": Histogram(
            "modelship_tts_generation_duration_seconds",
            description="TTS inference latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model",),
        ),
        "image_generation_duration_seconds": Histogram(
            "modelship_image_generation_duration_seconds",
            description="Image generation latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model",),
        ),
        "transcription_duration_seconds": Histogram(
            "modelship_transcription_duration_seconds",
            description="Speech-to-text latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model",),
        ),
        "embedding_duration_seconds": Histogram(
            "modelship_embedding_duration_seconds",
            description="Embedding inference latency in seconds.",
            boundaries=_REQUEST_LATENCY_BOUNDARIES,
            tag_keys=("model",),
        ),
        # -- Authentication --
        "auth_failures_total": Counter(
            "modelship_auth_failures_total",
            description="Total rejected requests due to invalid/missing API key.",
            tag_keys=("reason",),
        ),
        # -- Resource cleanup --
        "resource_cleanup_errors_total": Counter(
            "modelship_resource_cleanup_errors_total",
            description="Errors during resource cleanup (engine shutdown, memory release).",
            tag_keys=("model", "component"),  # type: ignore[arg-type]
        ),
    }


_metrics = _build_metrics()

# -- Gateway --
REQUEST_TOTAL = _metrics["request_total"]
REQUEST_DURATION_SECONDS = _metrics["request_duration_seconds"]
REQUEST_ERRORS_TOTAL = _metrics["request_errors_total"]
REQUEST_IN_PROGRESS = _metrics["request_in_progress"]
CLIENT_DISCONNECTS_TOTAL = _metrics["client_disconnects_total"]
STREAM_CHUNKS_TOTAL = _metrics["stream_chunks_total"]

# -- Model deployment --
MODEL_LOAD_DURATION_SECONDS = _metrics["model_load_duration_seconds"]
MODEL_LOAD_FAILURES_TOTAL = _metrics["model_load_failures_total"]
MODELS_LOADED = _metrics["models_loaded"]

# -- Inference timing --
GENERATION_DURATION_SECONDS = _metrics["generation_duration_seconds"]
TTS_GENERATION_DURATION_SECONDS = _metrics["tts_generation_duration_seconds"]
IMAGE_GENERATION_DURATION_SECONDS = _metrics["image_generation_duration_seconds"]
TRANSCRIPTION_DURATION_SECONDS = _metrics["transcription_duration_seconds"]
EMBEDDING_DURATION_SECONDS = _metrics["embedding_duration_seconds"]

# -- Authentication --
AUTH_FAILURES_TOTAL = _metrics["auth_failures_total"]

# -- Resource cleanup --
RESOURCE_CLEANUP_ERRORS_TOTAL = _metrics["resource_cleanup_errors_total"]
