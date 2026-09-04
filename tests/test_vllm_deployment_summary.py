from types import SimpleNamespace

import pytest

from modelship.infer.vllm.vllm_infer import _deployment_summary


def _config(**overrides):
    """A vllm_config shaped like the engine-resolved one. Values mirror a real
    Falcon-H1 deploy unless overridden."""
    model = dict(
        max_model_len=16384,
        original_max_model_len=-1,
        dtype="torch.bfloat16",
        quantization=None,
        is_hybrid=False,
    )
    cache = dict(
        kv_cache_size_tokens=134795,
        kv_cache_max_concurrency=8.227,
        num_gpu_blocks=181,
        block_size=800,
        gpu_memory_utilization=0.25,
    )
    scheduler = dict(max_num_seqs=8)
    parallel = dict(tensor_parallel_size=1, pipeline_parallel_size=1, world_size=1)
    for section, key in (
        (model, "model"),
        (cache, "cache"),
        (scheduler, "scheduler"),
        (parallel, "parallel"),
    ):
        section.update(overrides.get(key, {}))
    return SimpleNamespace(
        model_config=SimpleNamespace(**model),
        cache_config=SimpleNamespace(**cache),
        scheduler_config=SimpleNamespace(**scheduler),
        parallel_config=SimpleNamespace(**parallel),
    )


def _text(**overrides):
    return "\n".join(_deployment_summary("m", _config(**overrides)))


class TestDeploymentSummary:
    def test_reports_context_and_concurrency(self):
        out = _text()
        assert "16,384 tokens" in out
        assert "up to 8 concurrent requests" in out
        assert "8.2 of them at that length" in out

    def test_marks_auto_fit_context(self):
        assert "auto-fit to free memory" in _text()

    def test_marks_requested_context(self):
        out = _text(model={"original_max_model_len": 16384})
        assert "as requested" in out
        assert "auto-fit" not in out

    def test_hybrid_line_only_for_hybrids(self):
        assert "mamba blocks" not in _text()
        assert "mamba blocks" in _text(model={"is_hybrid": True})

    def test_hybrid_line_reports_the_max_num_seqs_ceiling(self):
        # num_gpu_blocks is the ceiling vLLM's own check compares max_num_seqs
        # against; a distinct value keeps this independent of the kv-cache line.
        out = _text(model={"is_hybrid": True}, cache={"num_gpu_blocks": 134, "kv_cache_size_tokens": 20679})
        assert "134 mamba blocks — max_num_seqs cannot exceed this" in out
        assert "134" not in out.split("hybrid")[0]

    def test_unknown_origin_claims_neither(self):
        # A vLLM without the field must not be reported as an explicit request.
        out = _text(model={"original_max_model_len": None})
        assert "16,384 tokens" in out
        assert "as requested" not in out
        assert "auto-fit" not in out

    def test_zero_concurrency_is_reported_not_hidden(self):
        # 0.0 means the pool cannot hold one full-length request — the most
        # important case to show, and the one truthiness would swallow.
        out = _text(cache={"kv_cache_max_concurrency": 0.0})
        assert "holds 0.0 of them at that length" in out

    def test_zero_mamba_blocks_is_reported(self):
        out = _text(model={"is_hybrid": True}, cache={"num_gpu_blocks": 0})
        assert "0 mamba blocks" in out

    def test_kv_cache_reports_group_aware_capacity_only(self):
        # Whisper's real numbers: requests span the self- and cross-attention
        # groups, so blocks x block_size (115,216) is unrelated to capacity.
        out = _text(cache={"kv_cache_size_tokens": 20679, "num_gpu_blocks": 7201, "block_size": 16})
        assert "kv cache     20,679 tokens" in out
        assert "7,201" not in out
        assert "115,216" not in out

    def test_kv_cache_line_omitted_before_kv_init(self):
        # These are populated during KV-cache init; absent on an early read.
        out = _text(cache={"kv_cache_size_tokens": None, "kv_cache_max_concurrency": None})
        assert "kv cache" not in out
        assert "up to 8 concurrent requests" in out
        assert "at that length" not in out

    def test_parallelism_shown_only_when_distributed(self):
        assert "tp=" not in _text()
        assert "tp=2 pp=2" in _text(parallel={"tensor_parallel_size": 2, "pipeline_parallel_size": 2, "world_size": 4})

    def test_quantization_shown_when_present(self):
        assert "bfloat16, awq" in _text(model={"quantization": "awq"})

    @pytest.mark.parametrize("missing", ["kv_cache_max_concurrency", "num_gpu_blocks"])
    def test_survives_missing_optional_fields(self, missing):
        # Diagnostics must never break a deploy.
        assert _text(cache={missing: None}).startswith("deployed 'm':")
