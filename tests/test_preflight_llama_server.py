"""Tests for the LlamaServerPreflight estimator."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.preflight import GPUInfo, HardwareProfile, merge_with_user_overrides
from modelship.preflight._mla import MLAInfo
from modelship.preflight._sliding_window import SlidingWindowInfo
from modelship.preflight.llama_cpp import LlamaServerPreflight, _GGUFMeta, _weight_bytes

_LLAMA_META = _GGUFMeta(block_count=32, head_count_kv=8, head_dim=128, context_length=131072)

# DeepSeek-V2-Lite converted before llama.cpp's MLA support: fused attn_kv_b,
# so the cache holds full per-head K/V. kv/token = (192+128)*27*16*2 = 276480 B,
# matching a real deploy's kv-cache cudaMalloc.
_MLA_LEGACY_META = _GGUFMeta(
    block_count=27,
    head_count_kv=16,
    head_dim=192,
    context_length=163840,
    v_head_dim=128,
)

# Same model converted with split attn_k_b/attn_v_b: llama.cpp caches the
# compressed latent, and key/value_length become the compressed dims
# (576/512, not 192/128). kv/token = (512+64)*27*2 = 31104 B.
_MLA_COMPRESSED_META = _GGUFMeta(
    block_count=27,
    head_count_kv=16,
    head_dim=576,
    context_length=163840,
    v_head_dim=512,
    mla=MLAInfo(kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=0, num_heads=16),
)

# Gemma-4-shaped: 48 layers, 40 sliding (window 1024) + 8 full, matching the
# real GGUF header. kv/token = 2*48*8*256*2 = 393216 B.
_GEMMA4_SLIDING_META = _GGUFMeta(
    block_count=48,
    head_count_kv=8,
    head_dim=256,
    context_length=262144,
    sliding=SlidingWindowInfo(n_full_layers=8, n_sliding_layers=40, n_total_layers=48, window=1024),
)


class _FakeGGUFField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _FakeGGUFTensor:
    def __init__(self, name):
        self.name = name


class _FakeGGUFReader:
    """Minimal stand-in for `gguf.GGUFReader` — `_read_field_value` only calls
    `.get_field(key).contents()`, and MLA detection reads `.tensors`."""

    def __init__(self, fields: dict, tensor_names: Sequence[str] | None = ()):
        self._fields = fields
        # Explicit None models a reader with no usable tensor list; the default
        # empty tuple is just "no tensors declared".
        self.tensors = None if tensor_names is None else [_FakeGGUFTensor(n) for n in tensor_names]

    def get_field(self, key: str):
        if key not in self._fields:
            return None
        return _FakeGGUFField(self._fields[key])


def _make_config(
    *,
    resolved_path: str | None = None,
    llama_server_kwargs: dict | None = None,
    num_gpus: float = 0,
    num_cpus: float = 0.1,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_server,
        llama_server_config=LlamaServerConfig(**(llama_server_kwargs or {})),
        num_gpus=num_gpus,
        num_cpus=num_cpus,
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_dummy_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 1024)
    return path


class TestLlamaServerPreflightCpu:
    def test_single_slot_matches_ram_budget_math(self, tmp_path):
        # parallel=1 (default): no division.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert "n_ctx" in rec
        assert rec["n_ctx"] % 256 == 0

    def test_parallel_divides_total_budget(self, tmp_path):
        # Same hardware/model, parallel=4 should yield a per-slot n_ctx roughly 1/4 of
        # the single-slot recommendation (the launch command re-multiplies by parallel).
        cfg_single = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 1})
        cfg_parallel = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 4})
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec_single = LlamaServerPreflight().recommend(cfg_single, hw)
            rec_parallel = LlamaServerPreflight().recommend(cfg_parallel, hw)

        assert rec_parallel["n_ctx"] * 4 <= rec_single["n_ctx"] + 256 * 4
        assert rec_parallel["n_ctx"] < rec_single["n_ctx"]

    def test_parallel_too_high_for_budget_returns_empty(self, tmp_path):
        # A tiny budget divided across many slots drops below the minimum usable n_ctx;
        # the estimator should decline rather than recommend something unusably small.
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 64}, num_gpus=0
        )
        hw = HardwareProfile(ram_bytes=2 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.9 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_run_preflight_dispatches_to_llama_server(self, tmp_path):
        from modelship.preflight import run_preflight

        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = run_preflight(cfg, hw)
        assert "n_ctx" in rec


class TestLlamaServerPreflightGpu:
    """`config.num_gpus >= 1` routes to `_recommend_gpu`. `_LLAMA_META` has
    block_count=32 (total_layers=33), head_count_kv=8, head_dim=128 → kv/token
    = 2*32*8*128*2 = 131072 B, kv/layer = 131072/32 = 4096 B."""

    def test_no_discoverable_gpus_returns_empty(self, tmp_path):
        # Ray only sets CUDA_VISIBLE_DEVICES for GPU-owning actors, and the
        # node-level pynvml view can be empty too — nothing to size against.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        with patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META):
            assert LlamaServerPreflight().recommend(cfg, hw) == {}

    def test_full_offload_when_vram_is_roomy(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        # block_count(32) + 1 non-block layer equivalent = full offload.
        assert rec["n_gpu_layers"] == 33
        # VRAM budget vastly exceeds what the model's own context_length needs,
        # so n_ctx caps at context_length (131072, already 256-aligned).
        assert rec["n_ctx"] == 131072

    def test_partial_offload_fits_fewer_layers_at_default_target(self, tmp_path):
        # VRAM too small to fit every layer's weights, but RAM is roomy enough
        # that the default 8192-token partial-offload target doesn't shrink.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec
        assert 0 < rec["n_gpu_layers"] < 33
        assert rec["n_ctx"] == 8192

    def test_ram_constrained_partial_shrinks_ctx(self, tmp_path):
        # Same tight VRAM as the previous test, but RAM is now also tight — the
        # CPU-resident layers' KV cache doesn't fit at 8192, so context shrinks.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "test")], ram_bytes=6 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec
        assert rec["n_ctx"] < 8192
        assert rec["n_ctx"] % 256 == 0
        assert rec["n_gpu_layers"] > 0

    def test_output_layer_ram_shortfall_returns_empty(self, tmp_path):
        # VRAM fits all 32 blocks but not the output layer, so cpu_blocks == 0 — but
        # the output layer's weights still need host RAM, and here RAM is too small for that.
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"n_ctx": 1024}, num_gpus=1
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, int(5.8 * 1024**3), "test")], ram_bytes=700 * 1024**2)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_doesnt_fit_anywhere_returns_empty(self, tmp_path):
        # Neither VRAM nor RAM can absorb even a minimal context.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 512 * 1024**2, "test")], ram_bytes=512 * 1024**2)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_multi_gpu_picks_smallest_n_as_lower_bound(self, tmp_path):
        # 4 GPUs discoverable at the node level but only 2 reserved: picking the 2
        # smallest-free is a lower bound over any 2-subset.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=2)
        hw_all_four = HardwareProfile(
            gpus=[
                GPUInfo(0, 40 * 1024**3, "test"),
                GPUInfo(1, 10 * 1024**3, "test"),
                GPUInfo(2, 30 * 1024**3, "test"),
                GPUInfo(3, 20 * 1024**3, "test"),
            ],
            ram_bytes=64 * 1024**3,
        )
        hw_two_smallest = HardwareProfile(
            gpus=[GPUInfo(0, 10 * 1024**3, "test"), GPUInfo(1, 20 * 1024**3, "test")],
            ram_bytes=64 * 1024**3,
        )

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec_all_four = LlamaServerPreflight().recommend(cfg, hw_all_four)
            rec_two_smallest = LlamaServerPreflight().recommend(cfg, hw_two_smallest)

        assert rec_all_four == rec_two_smallest

    def test_parallel_division_preserves_n_gpu_layers(self, tmp_path):
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1, llama_server_kwargs={"parallel": 4}
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        # n_gpu_layers is per-process, not per-slot — must survive the
        # parallel-division step untouched, unlike n_ctx.
        assert rec["n_gpu_layers"] == 33
        assert rec["n_ctx"] == 131072 // 4 // 256 * 256


class TestLlamaServerPreflightUnifiedMemory:
    """hw.unified_memory=True (Apple Silicon, GPUInfo.kind='mps') must never fall
    through to _recommend_gpu_partial — its separate RAM budget would double-count
    the same physical bytes already spent against vram_budget."""

    def test_full_offload_still_works_on_unified_memory(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "Apple GPU", kind="mps")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_gpu_layers"] == 33
        assert rec["n_ctx"] == 131072

    def test_declines_instead_of_partial_offload_when_budget_tight(self, tmp_path):
        # Same numbers as test_partial_offload_fits_fewer_layers_at_default_target
        # (partial offload there); on unified memory this must decline instead.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "Apple GPU", kind="mps")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_cuda_gpu_with_same_budget_still_goes_partial(self, tmp_path):
        # Control: identical numbers but kind="cuda" (the default) must be
        # unaffected — partial offload is correct there (separate VRAM/RAM pools).
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec
        assert 0 < rec["n_gpu_layers"] < 33


class TestLlamaServerPreflightFractionalGpu:
    """0 < num_gpus < 1 shares one physical GPU; budget is sized from the declared
    share of total capacity, not free VRAM."""

    def test_fractional_num_gpus_not_floored_to_zero(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=0.3)
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test", total_bytes=80 * 1024**3)], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_gpu_layers"] == 33

    def test_budget_derives_from_total_not_available(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=0.3)
        hw_roomy_free = HardwareProfile(
            gpus=[GPUInfo(0, 79 * 1024**3, "test", total_bytes=80 * 1024**3)], ram_bytes=64 * 1024**3
        )
        hw_tight_free = HardwareProfile(
            gpus=[GPUInfo(0, 1 * 1024**3, "test", total_bytes=80 * 1024**3)], ram_bytes=64 * 1024**3
        )

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec_roomy = LlamaServerPreflight().recommend(cfg, hw_roomy_free)
            rec_tight = LlamaServerPreflight().recommend(cfg, hw_tight_free)

        assert rec_roomy == rec_tight


class TestShardedGgufWeightBytes:
    def test_sums_sibling_shards(self, tmp_path):
        shard1 = tmp_path / "model-00001-of-00002.gguf"
        shard2 = tmp_path / "model-00002-of-00002.gguf"
        shard1.write_bytes(b"\0" * 1000)
        shard2.write_bytes(b"\0" * 2000)

        assert _weight_bytes(str(shard1)) == 3000

    def test_missing_sibling_shard_is_skipped(self, tmp_path):
        shard1 = tmp_path / "model-00001-of-00002.gguf"
        shard1.write_bytes(b"\0" * 1000)
        # shard 2 is missing entirely.

        assert _weight_bytes(str(shard1)) == 1000

    def test_non_sharded_file_uses_plain_size(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(b"\0" * 4096)
        assert _weight_bytes(str(path)) == 4096


class TestLlamaServerThreadsRecommendation:
    def test_whole_num_cpus_recommends_threads(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=4)
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec["threads"] == 4

    def test_default_fractional_num_cpus_has_no_threads_recommendation(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))  # num_cpus defaults to 0.1
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert "threads" not in rec

    def test_threads_recommended_even_when_gguf_metadata_unreadable(self, tmp_path):
        # Thread alignment doesn't depend on being able to size n_ctx.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=8)
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=None):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec == {"threads": 8}

    def test_user_set_threads_wins_at_merge_level(self):
        merged = merge_with_user_overrides({"threads": 4}, {"threads": 16}, model_name="m")
        assert merged["threads"] == 16

    def test_threads_declined_when_it_would_undercut_parallel_slots(self, tmp_path):
        # num_cpus=2 with parallel=4: capping to 2 threads would starve the 4 concurrent
        # slots of compute — decline and let llama-server keep all cores.
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_cpus=2,
            llama_server_kwargs={"parallel": 4},
        )
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert "threads" not in rec

    def test_threads_recommended_when_it_covers_parallel_slots(self, tmp_path):
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_cpus=4,
            llama_server_kwargs={"parallel": 4},
        )
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec["threads"] == 4


class TestResolveSlidingWindowGguf:
    def test_bool_pattern_counts_sliding_layers(self):
        from modelship.preflight.llama_cpp import _resolve_sliding_window_gguf

        reader = _FakeGGUFReader(
            {
                "gemma4.attention.sliding_window": 1024,
                "gemma4.attention.sliding_window_pattern": [True] * 5 + [False],
            }
        )
        sw = _resolve_sliding_window_gguf(reader, "gemma4", block_count=6)
        assert sw == SlidingWindowInfo(n_full_layers=1, n_sliding_layers=5, n_total_layers=6, window=1024)

    def test_int_period_pattern_every_nth_layer_full(self):
        from modelship.preflight.llama_cpp import _resolve_sliding_window_gguf

        reader = _FakeGGUFReader({"llama.attention.sliding_window": 4096, "llama.attention.sliding_window_pattern": 4})
        sw = _resolve_sliding_window_gguf(reader, "llama", block_count=8)
        assert sw == SlidingWindowInfo(n_full_layers=2, n_sliding_layers=6, n_total_layers=8, window=4096)

    def test_bare_window_with_no_pattern_means_every_layer_slides(self):
        from modelship.preflight.llama_cpp import _resolve_sliding_window_gguf

        reader = _FakeGGUFReader({"mistral.attention.sliding_window": 4096})
        sw = _resolve_sliding_window_gguf(reader, "mistral", block_count=8)
        assert sw == SlidingWindowInfo(n_full_layers=0, n_sliding_layers=8, n_total_layers=8, window=4096)

    def test_no_window_field_returns_none(self):
        from modelship.preflight.llama_cpp import _resolve_sliding_window_gguf

        reader = _FakeGGUFReader({})
        assert _resolve_sliding_window_gguf(reader, "llama", block_count=32) is None


class TestReadIntAt:
    def test_array_field_returns_value_at_index(self):
        from modelship.preflight.llama_cpp import _read_int_at

        reader = _FakeGGUFReader({"gemma4.attention.head_count_kv": [8, 8, 1, 8]})
        assert _read_int_at(reader, "gemma4.attention.head_count_kv", 2, default=0) == 1

    def test_scalar_field_ignores_index(self):
        from modelship.preflight.llama_cpp import _read_int_at

        reader = _FakeGGUFReader({"llama.attention.head_count_kv": 8})
        assert _read_int_at(reader, "llama.attention.head_count_kv", 5, default=0) == 8

    def test_missing_field_returns_default(self):
        from modelship.preflight.llama_cpp import _read_int_at

        reader = _FakeGGUFReader({})
        assert _read_int_at(reader, "missing.key", 0, default=42) == 42


class TestReadGgufMetadataSliding:
    """End-to-end through `_read_gguf_metadata` against a fake reader shaped
    like the real Gemma 4 GGUF header."""

    def test_gemma4_header_resolves_sliding_and_swa_head_dim(self):
        # patch("gguf.GGUFReader", ...) needs the real module importable.
        pytest.importorskip("gguf")
        from modelship.preflight.llama_cpp import _read_gguf_metadata

        fields = {
            "general.architecture": "gemma4",
            "gemma4.block_count": 48,
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 1] * 8,
            "gemma4.embedding_length": 2560,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False] * 8,
            "gemma4.context_length": 262144,
        }
        with patch("gguf.GGUFReader", return_value=_FakeGGUFReader(fields)):
            meta = _read_gguf_metadata("/fake/path.gguf")

        assert meta is not None
        assert meta.head_dim == 256  # SWA head_dim, not the global layers' 512.
        assert meta.head_count_kv == 8  # sampled at a sliding-layer position.
        assert meta.sliding == SlidingWindowInfo(n_full_layers=8, n_sliding_layers=40, n_total_layers=48, window=1024)


class TestLlamaServerPreflightGpuSliding:
    """kv/token = 393216 B for _GEMMA4_SLIDING_META (sliding-aware, not
    linear)."""

    def test_full_offload_hits_context_cap_not_the_naive_linear_ctx(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 30 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_GEMMA4_SLIDING_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_ctx"] == 262144
        assert rec["n_gpu_layers"] == 49
        # Naive linear division of the same budget lands at 59392 — proves the
        # sliding-aware fit, not a roomier budget, is what reached the cap.
        assert rec["n_ctx"] > 59392


class TestLlamaServerPreflightCpuSliding:
    def test_sliding_aware_fit_reaches_cap_naive_division_would_miss(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=0)
        hw = HardwareProfile(ram_bytes=25 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_GEMMA4_SLIDING_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_ctx"] == 262144
        # Naive linear division of the same budget lands at 50432 tokens.
        assert rec["n_ctx"] > 50432


def _deepseek_fields(*, key_length: int, value_length: int) -> dict:
    return {
        "general.architecture": "deepseek2",
        "deepseek2.block_count": 27,
        "deepseek2.attention.head_count": 16,
        "deepseek2.attention.head_count_kv": 16,
        "deepseek2.embedding_length": 2048,
        "deepseek2.attention.key_length": key_length,
        "deepseek2.attention.value_length": value_length,
        "deepseek2.attention.kv_lora_rank": 512,
        "deepseek2.rope.dimension_count": 64,
        "deepseek2.context_length": 163840,
    }


# Both taken from the real files: a fused conversion reports per-head dims, a
# split one reports the compressed dims instead.
_FUSED_FIELDS = _deepseek_fields(key_length=192, value_length=128)
_SPLIT_FIELDS = _deepseek_fields(key_length=576, value_length=512)

# One per block, as a real conversion ships them.
_SPLIT_MLA_TENSORS = [f"blk.{i}.attn_{p}_b.weight" for i in range(27) for p in ("k", "v")]
_FUSED_MLA_TENSORS = [f"blk.{i}.attn_kv_b.weight" for i in range(27)]


class TestReadGgufMetadataMla:
    """`_read_gguf_metadata` against fake readers shaped like the two real
    DeepSeek-V2-Lite GGUF conversions."""

    def test_split_tensors_resolve_compressed_mla(self):
        pytest.importorskip("gguf")
        from modelship.preflight.llama_cpp import _read_gguf_metadata

        reader = _FakeGGUFReader(_SPLIT_FIELDS, _SPLIT_MLA_TENSORS)
        with patch("gguf.GGUFReader", return_value=reader):
            meta = _read_gguf_metadata("/fake/path.gguf")

        assert meta is not None
        assert meta.mla == MLAInfo(
            kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=0, num_heads=16
        )

    def test_fused_tensors_leave_mla_unset(self):
        # Same metadata, pre-MLA tensor layout: detection must not fire on
        # metadata alone, or the compressed formula underestimates 8.9x.
        pytest.importorskip("gguf")
        from modelship.preflight.llama_cpp import _read_gguf_metadata

        reader = _FakeGGUFReader(_FUSED_FIELDS, _FUSED_MLA_TENSORS)
        with patch("gguf.GGUFReader", return_value=reader):
            meta = _read_gguf_metadata("/fake/path.gguf")

        assert meta is not None
        assert meta.mla is None
        assert (meta.head_dim, meta.v_head_dim) == (192, 128)


class TestKvBytesPerTokenMla:
    def test_fused_matches_real_deploy_oom(self):
        # Anchored to a real DeepSeek-V2-Lite-Chat deploy: ggml reported
        # "failed to allocate CUDA0 buffer of size 24347934720" at n_ctx=88064,
        # i.e. exactly 276480 B/token.
        from modelship.preflight.llama_cpp import _kv_bytes_per_token

        assert _kv_bytes_per_token(_MLA_LEGACY_META) == 276480

    def test_split_uses_compressed_latent(self):
        from modelship.preflight.llama_cpp import _kv_bytes_per_token

        assert _kv_bytes_per_token(_MLA_COMPRESSED_META) == 31104

    def test_symmetric_archs_are_unchanged(self):
        # The old `2 * head_dim` form, still exact when V is as wide as K.
        from modelship.preflight.llama_cpp import _kv_bytes_per_token

        assert _kv_bytes_per_token(_LLAMA_META) == 2 * 32 * 8 * 128 * 2
        assert _kv_bytes_per_token(_GEMMA4_SLIDING_META) == 2 * 48 * 8 * 256 * 2


class TestLlamaServerPreflightGpuMla:
    def test_fused_sizes_ctx_to_full_per_head_cache(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 30 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_MLA_LEGACY_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_gpu_layers"] == 28
        assert rec["n_ctx"] == 84480

    def test_split_reaches_context_cap_on_the_same_budget(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 30 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_MLA_COMPRESSED_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec["n_ctx"] == 163840


# Mirrors the real Qwen3.8-27B GGUF: 65 blocks, of which 48 are Gated DeltaNet
# (ssm_*) and 17 carry attention K/V.
_QWEN35_FIELDS = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.embedding_length": 5120,
    "qwen35.context_length": 262144,
}
_QWEN35_ATTN_BLOCKS = [*range(3, 64, 4), 64]
_QWEN35_TENSORS = [f"blk.{i}.{'attn_k.weight' if i in _QWEN35_ATTN_BLOCKS else 'ssm_conv1d.weight'}" for i in range(65)]


class TestAttentionBlockCount:
    """Hybrid recurrent archs cache KV only on their attention blocks; counting
    every block overestimates kv/token by the hybrid ratio."""

    def _meta(self, fields, tensors):
        pytest.importorskip("gguf")
        from modelship.preflight.llama_cpp import _read_gguf_metadata

        with patch("gguf.GGUFReader", return_value=_FakeGGUFReader(fields, tensors)):
            return _read_gguf_metadata("/fake/path.gguf")

    def test_recurrent_blocks_are_excluded(self):
        from modelship.preflight.llama_cpp import _kv_bytes_per_token

        meta = self._meta(_QWEN35_FIELDS, _QWEN35_TENSORS)
        assert meta is not None
        assert meta.block_count == 65
        assert meta.attn_block_count == 17
        assert _kv_bytes_per_token(meta) == (256 + 256) * 17 * 4 * 2

    def test_dense_model_counts_every_block(self):
        fields = {**_QWEN35_FIELDS, "general.architecture": "llama", "llama.block_count": 4}
        fields = {k.replace("qwen35.", "llama."): v for k, v in fields.items()}
        meta = self._meta(fields, [f"blk.{i}.attn_k.weight" for i in range(4)])
        assert meta is not None
        assert meta.attn_block_count == meta.block_count == 4

    def test_attention_free_model_falls_back(self):
        # Dividing by a zero kv/token would blow up the callers.
        fields = {**_QWEN35_FIELDS, "qwen35.block_count": 8}
        meta = self._meta(fields, [f"blk.{i}.ssm_conv1d.weight" for i in range(8)])
        assert meta is not None
        assert meta.attn_block_count == 8

    def test_unreadable_tensor_list_falls_back(self):
        meta = self._meta(_QWEN35_FIELDS, None)
        assert meta is not None
        assert meta.attn_block_count == meta.block_count == 65

    def test_partial_tensor_list_falls_back(self):
        # Only the first shard is opened; counting its blocks alone would
        # undercount KV and oversize n_ctx.
        shard = [t for t in _QWEN35_TENSORS if int(t.split(".")[1]) < 22]
        meta = self._meta(_QWEN35_FIELDS, shard)
        assert meta is not None
        assert meta.attn_block_count == meta.block_count == 65
