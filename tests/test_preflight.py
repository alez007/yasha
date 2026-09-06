"""Tests for the preflight estimator framework and VllmPreflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
    default_gpu_memory_utilization,
    resolve_gpu_memory_utilization,
)
from modelship.preflight import (
    GPUInfo,
    HardwareProfile,
    merge_with_user_overrides,
    run_preflight,
)
from modelship.preflight.vllm import VllmPreflight


def _make_config(
    *,
    resolved_path: str | None = None,
    vllm_kwargs: dict | None = None,
    num_gpus: float = 1,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.vllm,
        num_gpus=num_gpus,
        vllm_engine_kwargs=VllmEngineConfig(**(vllm_kwargs or {})),
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_model_snapshot(
    tmp_path: Path,
    *,
    config_json: dict,
    weight_bytes: int,
) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(config_json))
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": weight_bytes}, "weight_map": {}})
    )
    return snapshot


class TestGpuDiscoveryUuid:
    """GPUInfo.uuid, populated by the pynvml/torch probes, lets an operator
    co-locating node containers on one host verify each got a distinct physical GPU."""

    def test_pynvml_probe_reads_uuid(self):
        from modelship.preflight import base as preflight_base

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = object()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(free=1024, total=2048)
        mock_pynvml.nvmlDeviceGetName.return_value = "Test GPU"
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-abc123"

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            gpus = preflight_base._pynvml_node_discover()

        assert gpus == [GPUInfo(index=0, available_bytes=1024, name="Test GPU", uuid="GPU-abc123", total_bytes=2048)]

    def test_pynvml_probe_decodes_bytes_uuid(self):
        # Some pynvml builds return bytes for name/UUID rather than str.
        from modelship.preflight import base as preflight_base

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = object()
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(free=1024, total=2048)
        mock_pynvml.nvmlDeviceGetName.return_value = b"Test GPU"
        mock_pynvml.nvmlDeviceGetUUID.return_value = b"GPU-abc123"

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            gpus = preflight_base._pynvml_node_discover()

        assert gpus == [GPUInfo(index=0, available_bytes=1024, name="Test GPU", uuid="GPU-abc123", total_bytes=2048)]

    def test_torch_probe_reads_uuid(self):
        from modelship.preflight import base as preflight_base

        mock_props = SimpleNamespace(name="Test GPU", total_memory=2048, uuid="abc123")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (1024, 2048)
        mock_torch.version.hip = None

        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpus = preflight_base._torch_cuda_discover()

        # torch's uuid has no "GPU-" prefix (unlike pynvml's); the probe adds it so both
        # sources agree with nvidia-smi's own "GPU-<uuid>" format.
        assert gpus == [GPUInfo(index=0, available_bytes=1024, name="Test GPU", uuid="GPU-abc123", total_bytes=2048)]

    def test_torch_probe_uuid_none_when_attr_missing(self):
        # Older torch builds predate the `uuid` device-properties field.
        from modelship.preflight import base as preflight_base

        mock_props = SimpleNamespace(name="Test GPU", total_memory=2048)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (1024, 2048)
        mock_torch.version.hip = None

        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpus = preflight_base._torch_cuda_discover()

        assert gpus == [GPUInfo(index=0, available_bytes=1024, name="Test GPU", uuid=None, total_bytes=2048)]

    def test_torch_probe_derives_rocm_kind_from_hip_version(self):
        """ROCm PyTorch maps torch.cuda onto HIP; torch.version.hip is the only
        signal distinguishing an AMD device from a real CUDA one on this path."""
        from modelship.preflight import base as preflight_base

        mock_props = SimpleNamespace(name="MI300X", total_memory=2048)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (1024, 2048)
        mock_torch.version.hip = "5.7.1"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpus = preflight_base._torch_cuda_discover()

        assert gpus == [GPUInfo(index=0, available_bytes=1024, name="MI300X", uuid=None, kind="rocm", total_bytes=2048)]

    def test_torch_probe_defaults_to_cuda_kind_when_hip_unset(self):
        from modelship.preflight import base as preflight_base

        mock_props = SimpleNamespace(name="A100", total_memory=2048)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (1024, 2048)
        mock_torch.version.hip = None

        with patch.dict(sys.modules, {"torch": mock_torch}):
            gpus = preflight_base._torch_cuda_discover()

        assert gpus[0].kind == "cuda"


class TestRocmSmiDiscovery:
    """_rocm_smi_node_discover() — the non-torch ROCm probe for an
    llama_server-only install, which has no torch to key off of at all."""

    def test_no_binary_returns_empty(self):
        from modelship.preflight import base as preflight_base

        with patch("shutil.which", return_value=None):
            assert preflight_base._rocm_smi_node_discover() == []

    def test_parses_rocm_smi_json(self):
        from modelship.preflight import base as preflight_base

        payload = json.dumps(
            {
                "card0": {
                    "Card series": "MI300X",
                    "VRAM Total Memory (B)": "1000",
                    "VRAM Total Used Memory (B)": "400",
                }
            }
        )
        mock_result = SimpleNamespace(stdout=payload)
        with (
            patch("shutil.which", return_value="/opt/rocm/bin/rocm-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            gpus = preflight_base._rocm_smi_node_discover()

        assert gpus == [GPUInfo(index=0, available_bytes=600, name="MI300X", uuid=None, kind="rocm", total_bytes=1000)]

    def test_binary_failure_returns_empty(self):
        from modelship.preflight import base as preflight_base

        with (
            patch("shutil.which", return_value="/opt/rocm/bin/rocm-smi"),
            patch("subprocess.run", side_effect=OSError("boom")),
        ):
            assert preflight_base._rocm_smi_node_discover() == []

    def test_orders_cards_numerically_not_lexicographically(self):
        from modelship.preflight import base as preflight_base

        def card(n):
            return {
                f"card{n}": {
                    "Card series": f"MI300X-{n}",
                    "VRAM Total Memory (B)": "1000",
                    "VRAM Total Used Memory (B)": "0",
                }
            }

        payload = {}
        for n in (0, 1, 2, 10):
            payload.update(card(n))
        mock_result = SimpleNamespace(stdout=json.dumps(payload))
        with (
            patch("shutil.which", return_value="/opt/rocm/bin/rocm-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            gpus = preflight_base._rocm_smi_node_discover()

        assert [gpu.name for gpu in gpus] == ["MI300X-0", "MI300X-1", "MI300X-2", "MI300X-10"]
        assert [gpu.index for gpu in gpus] == [0, 1, 2, 3]


class TestAppleMetalDiscovery:
    """_apple_metal_discover() — darwin+arm64 only, torch.mps preferred over the
    sysctl/psutil heuristic, checked last in detect_gpus() so CUDA wins when present."""

    def _mock_psutil(self, total, available):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = SimpleNamespace(total=total, available=available)
        return mock_psutil

    def test_non_darwin_returns_empty(self):
        from modelship.preflight import base as preflight_base

        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert preflight_base._apple_metal_discover() == []

    def test_darwin_intel_returns_empty(self):
        from modelship.preflight import base as preflight_base

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert preflight_base._apple_metal_discover() == []

    def test_darwin_arm64_returns_single_mps_gpu(self):
        from modelship.preflight import base as preflight_base

        mock_psutil = self._mock_psutil(total=16 * 1024**3, available=8 * 1024**3)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.dict(sys.modules, {"psutil": mock_psutil, "torch": None}),
            patch.object(preflight_base, "_sysctl_int", return_value=None),
            patch.object(preflight_base, "_sysctl_str", return_value="Apple M1 Pro"),
        ):
            gpus = preflight_base._apple_metal_discover()

        assert len(gpus) == 1
        assert gpus[0].kind == "mps"
        assert gpus[0].index == 0
        assert gpus[0].name == "Apple M1 Pro"

    def test_torch_mps_recommendation_preferred_over_sysctl(self):
        from modelship.preflight import base as preflight_base

        mock_psutil = self._mock_psutil(total=16 * 1024**3, available=16 * 1024**3)
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.recommended_max_memory.return_value = 5 * 1024**3
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.dict(sys.modules, {"psutil": mock_psutil, "torch": mock_torch}),
            patch.object(preflight_base, "_sysctl_str", return_value="Apple M1 Pro"),
        ):
            gpus = preflight_base._apple_metal_discover()

        # cap (5 GiB, from torch) < available (16 GiB) -> cap wins.
        assert gpus[0].available_bytes == 5 * 1024**3

    def test_clamp_uses_available_when_smaller_than_cap(self):
        from modelship.preflight import base as preflight_base

        mock_psutil = self._mock_psutil(total=16 * 1024**3, available=2 * 1024**3)
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.recommended_max_memory.return_value = 10 * 1024**3
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.dict(sys.modules, {"psutil": mock_psutil, "torch": mock_torch}),
            patch.object(preflight_base, "_sysctl_str", return_value="Apple M1 Pro"),
        ):
            gpus = preflight_base._apple_metal_discover()

        # available (2 GiB) < cap (10 GiB) -> available wins.
        assert gpus[0].available_bytes == 2 * 1024**3

    def test_falls_back_to_wired_limit_sysctl_when_torch_unavailable(self):
        from modelship.preflight import base as preflight_base

        mock_psutil = self._mock_psutil(total=16 * 1024**3, available=16 * 1024**3)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.dict(sys.modules, {"psutil": mock_psutil, "torch": None}),
            patch.object(preflight_base, "_sysctl_int", return_value=4096),  # MB
            patch.object(preflight_base, "_sysctl_str", return_value="Apple M1 Pro"),
        ):
            gpus = preflight_base._apple_metal_discover()

        assert gpus[0].available_bytes == 4096 * 1024 * 1024

    def test_detect_gpus_prefers_cuda_over_metal(self):
        from modelship.preflight import base as preflight_base

        cuda_gpu = GPUInfo(index=0, available_bytes=1, name="cuda-gpu", uuid=None)
        with (
            patch.object(preflight_base, "_torch_cuda_discover", return_value=[cuda_gpu]),
            patch.object(preflight_base, "_apple_metal_discover") as mock_metal,
        ):
            gpus = preflight_base.detect_gpus()

        assert gpus == [cuda_gpu]
        mock_metal.assert_not_called()

    def test_detect_gpus_falls_back_to_metal_when_no_cuda(self):
        from modelship.preflight import base as preflight_base

        mps_gpu = GPUInfo(index=0, available_bytes=1, name="Apple GPU", uuid=None, kind="mps")
        with (
            patch.object(preflight_base, "_torch_cuda_discover", return_value=[]),
            patch.object(preflight_base, "_pynvml_node_discover", return_value=[]),
            patch.object(preflight_base, "_rocm_smi_node_discover", return_value=[]),
            patch.object(preflight_base, "_apple_metal_discover", return_value=[mps_gpu]),
        ):
            gpus = preflight_base.detect_gpus()

        assert gpus == [mps_gpu]

    def test_detect_gpus_tries_rocm_smi_before_metal(self):
        """pynvml is NVIDIA-only, so a torch-less ROCm install (llama_server-only)
        needs a chance to answer via rocm-smi before falling all the way to Metal."""
        from modelship.preflight import base as preflight_base

        rocm_gpu = GPUInfo(index=0, available_bytes=1, name="MI300X", uuid=None, kind="rocm")
        with (
            patch.object(preflight_base, "_torch_cuda_discover", return_value=[]),
            patch.object(preflight_base, "_pynvml_node_discover", return_value=[]),
            patch.object(preflight_base, "_rocm_smi_node_discover", return_value=[rocm_gpu]),
            patch.object(preflight_base, "_apple_metal_discover") as mock_metal,
        ):
            gpus = preflight_base.detect_gpus()

        assert gpus == [rocm_gpu]
        mock_metal.assert_not_called()


class TestUnifiedMemory:
    def test_true_when_any_gpu_is_mps(self):
        hw = HardwareProfile(gpus=[GPUInfo(index=0, available_bytes=1, name="Apple GPU", kind="mps")])
        assert hw.unified_memory is True

    def test_false_for_cuda_gpus(self):
        hw = HardwareProfile(gpus=[GPUInfo(index=0, available_bytes=1, name="cuda-gpu")])
        assert hw.unified_memory is False

    def test_false_when_no_gpus(self):
        assert HardwareProfile(gpus=[]).unified_memory is False


class TestMergeWithUserOverrides:
    def test_recommendation_fills_missing(self):
        result = merge_with_user_overrides({"max_model_len": 4096}, {}, model_name="m")
        assert result == {"max_model_len": 4096}

    def test_user_value_wins(self):
        result = merge_with_user_overrides(
            {"max_model_len": 4096},
            {"max_model_len": 32000},
            model_name="m",
        )
        assert result == {"max_model_len": 32000}

    def test_disjoint_keys_merge(self):
        result = merge_with_user_overrides(
            {"max_model_len": 4096},
            {"tensor_parallel_size": 2},
            model_name="m",
        )
        assert result == {"max_model_len": 4096, "tensor_parallel_size": 2}

    def test_warning_emitted_on_divergence(self):
        with patch("modelship.preflight.base.logger") as mock_logger:
            merge_with_user_overrides(
                {"max_model_len": 4096},
                {"max_model_len": 32000},
                model_name="gemma4-coder",
            )
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "gemma4-coder" in call_args.args
        assert "max_model_len" in call_args.args

    def test_matching_value_no_warning(self):
        with patch("modelship.preflight.base.logger") as mock_logger:
            merge_with_user_overrides(
                {"max_model_len": 4096},
                {"max_model_len": 4096},
                model_name="m",
            )
        mock_logger.warning.assert_not_called()


class TestRunPreflightDispatch:
    def test_returns_empty_for_unregistered_loader(self):
        cfg = _make_config()
        cfg.loader = ModelLoader.diffusers
        result = run_preflight(cfg, HardwareProfile())
        assert result == {}

    def test_swallows_estimator_exceptions(self):
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", side_effect=RuntimeError("boom")):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {}

    def test_disabled_via_env_returns_empty_even_with_recommendation(self, monkeypatch):
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", return_value={"max_model_len": 4096}):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {}

    def test_enabled_by_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MSHIP_PREFLIGHT", raising=False)
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", return_value={"max_model_len": 4096}):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {"max_model_len": 4096}


class TestEstimateWeightFootprint:
    def test_no_real_files_trusts_index_total(self, tmp_path):
        # _write_model_snapshot only ever writes the index JSON, never real
        # .safetensors files — this must keep returning the declared total_size unchanged.
        from modelship.preflight.vllm import _estimate_weight_footprint

        snapshot = _write_model_snapshot(tmp_path, config_json={}, weight_bytes=5 * 1024**3)
        assert _estimate_weight_footprint(str(snapshot)) == 5 * 1024**3

    def test_unindexed_safetensors_file_is_not_dropped(self, tmp_path):
        # An unindexed safetensors file (e.g. a vision tower not referenced by the
        # index) must still be counted — the directory sum then exceeds the index
        # total and wins the max().
        from modelship.preflight.vllm import _estimate_weight_footprint

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        indexed_shard_bytes = 8 * 1024
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": indexed_shard_bytes}, "weight_map": {}})
        )
        (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"\0" * indexed_shard_bytes)
        unindexed_bytes = 2 * 1024
        (snapshot / "vision_tower.safetensors").write_bytes(b"\0" * unindexed_bytes)
        assert _estimate_weight_footprint(str(snapshot)) == indexed_shard_bytes + unindexed_bytes

    def test_path_is_a_file_not_a_directory_returns_zero(self, tmp_path):
        # os.listdir() raises NotADirectoryError (an OSError subclass, not
        # FileNotFoundError) here — must degrade gracefully like the missing-path case.
        from modelship.preflight.vllm import _estimate_weight_footprint

        not_a_dir = tmp_path / "not_a_directory"
        not_a_dir.write_text("oops")
        assert _estimate_weight_footprint(str(not_a_dir)) == 0


_DENSE_CFG = {
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "hidden_size": 5120,
    "head_dim": 160,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 32768,
}


class TestVllmPreflight:
    # The recommendation keys that aren't VllmEngineConfig fields. vllm_infer pops
    # each one out of the merged kwargs before building the model, which forbids
    # extras; gpu_memory_utilization goes to resolve_gpu_memory_utilization instead.
    _ACTOR_POPPED = frozenset({"gpu_memory_utilization"})

    def test_recommendation_survives_the_actor_merge(self, tmp_path):
        """Every shape whose recommendation reaches VllmEngineConfig, including the
        two that recommend a derived gpu_memory_utilization (MLA on GPU, and CPU) —
        a dense GPU deploy is the only one that never emits it."""
        cases = (
            ("dense/40GiB", _DENSE_CFG, 1, HardwareProfile(gpus=[GPUInfo(0, 40 * 1024**3, "test")])),
            ("dense/80GiB", _DENSE_CFG, 1, HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])),
            ("mla/40GiB", _MLA_CFG, 1, HardwareProfile(gpus=[GPUInfo(0, 40 * 1024**3, "test")])),
            ("dense/cpu", _DENSE_CFG, 0, HardwareProfile(ram_bytes=64 * 1024**3)),
        )
        for label, config_json, num_gpus, hw in cases:
            case_dir = tmp_path / label.replace("/", "-")
            case_dir.mkdir()
            snapshot = _write_model_snapshot(case_dir, config_json=config_json, weight_bytes=19 * 1024**3)
            cfg = _make_config(resolved_path=str(snapshot), num_gpus=num_gpus)
            rec = VllmPreflight().recommend(cfg, hw)
            assert rec, label
            assert set(rec) - self._ACTOR_POPPED <= set(VllmEngineConfig.model_fields), label
            VllmEngineConfig(**{k: v for k, v in rec.items() if k not in self._ACTOR_POPPED})

    def test_popped_keys_are_the_ones_the_schema_rejects_by_name(self):
        """Popping a key the schema doesn't declare derived would hide a typo:
        forbid would have caught it, the pop swallows it."""
        from modelship.utils.config_schema import _VLLM_DERIVED_KEYS

        assert set(_VLLM_DERIVED_KEYS) >= self._ACTOR_POPPED

    def test_gpu_memory_utilization_is_recommended_where_it_matters(self, tmp_path):
        """Pins the asymmetry the merge test relies on: derived on CPU, absent on a
        dense GPU deploy. If this flips, the case list above is no longer covering
        both sides."""
        recs = {}
        for label, num_gpus, hw in (
            ("gpu", 1, HardwareProfile(gpus=[GPUInfo(0, 40 * 1024**3, "test")])),
            ("cpu", 0, HardwareProfile(ram_bytes=64 * 1024**3)),
        ):
            case_dir = tmp_path / label
            case_dir.mkdir()
            snapshot = _write_model_snapshot(case_dir, config_json=_DENSE_CFG, weight_bytes=19 * 1024**3)
            cfg = _make_config(resolved_path=str(snapshot), num_gpus=num_gpus)
            recs[label] = VllmPreflight().recommend(cfg, hw)
        assert "gpu_memory_utilization" not in recs["gpu"]
        assert "gpu_memory_utilization" in recs["cpu"]

    def test_no_gpus_returns_empty(self):
        cfg = _make_config(resolved_path="/nonexistent")
        assert VllmPreflight().recommend(cfg, HardwareProfile()) == {}

    def test_no_resolved_path_returns_empty(self):
        cfg = _make_config()
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_missing_config_json_returns_empty(self, tmp_path):
        cfg = _make_config(resolved_path=str(tmp_path))
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_constrained_budget_recommends_lower_max_model_len(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "hidden_size": 5120,
                "head_dim": 160,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=19 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 2},
            num_gpus=2,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test"), GPUInfo(1, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_model_len" in rec
        assert rec["max_model_len"] < 32768
        assert rec["max_model_len"] % 16 == 0

    def test_roomy_budget_caps_at_max_position_embeddings(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
            },
            weight_bytes=(15 * 1024**3),
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == 8192

    def test_missing_geometry_returns_empty(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={"torch_dtype": "bfloat16"},
            weight_bytes=1024,
        )
        cfg = _make_config(resolved_path=str(snapshot))
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_budget_below_zero_returns_empty(self, tmp_path):
        # Tiny GPU, huge model: budget goes negative.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "hidden_size": 8192,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=(140 * 1024**3),
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_fractional_num_gpus_sizes_max_model_len_to_share(self, tmp_path):
        # num_gpus=0.5 halves gpu_memory_utilization; when that budget is too small
        # for the full context, preflight sizes max_model_len down instead of bailing.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 8,
                "hidden_size": 3584,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=5 * 1024**3,
        )
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0.5)
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_model_len" in rec
        assert 0 < rec["max_model_len"] < 32768
        assert rec["max_model_len"] % 16 == 0

    def test_fractional_share_recommends_less_than_whole_gpu(self, tmp_path):
        # Same model, same card: a 0.5 share yields a smaller max_model_len than the whole GPU.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "hidden_size": 3584,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 131072,
            },
            weight_bytes=5 * 1024**3,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        shared = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot), num_gpus=0.5), hw)
        whole = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot), num_gpus=1), hw)
        assert shared["max_model_len"] < whole["max_model_len"]

    def test_fp8_kv_halves_per_token_bytes(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 1_000_000,
            },
            weight_bytes=15 * 1024**3,
        )
        cfg_fp16 = _make_config(
            resolved_path=str(snapshot),
        )
        cfg_fp8 = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"kv_cache_dtype": "fp8_e4m3"},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        rec_fp16 = VllmPreflight().recommend(cfg_fp16, hw)
        rec_fp8 = VllmPreflight().recommend(cfg_fp8, hw)
        # fp8 stores KV in half the bytes, so the suggested context roughly doubles.
        assert rec_fp8["max_model_len"] >= rec_fp16["max_model_len"]


class TestVllmPreflightFractionalGpu:
    """0 < num_gpus < 1 shares one physical GPU; budget derives from total capacity
    * gpu_memory_utilization (which equals the fraction), not from free VRAM."""

    def test_budget_derives_from_total_not_available(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
            },
            weight_bytes=4 * 1024**3,
        )
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0.5)
        hw_roomy_free = HardwareProfile(gpus=[GPUInfo(0, 79 * 1024**3, "test", total_bytes=80 * 1024**3)])
        hw_tight_free = HardwareProfile(gpus=[GPUInfo(0, 1 * 1024**3, "test", total_bytes=80 * 1024**3)])

        rec_roomy = VllmPreflight().recommend(cfg, hw_roomy_free)
        rec_tight = VllmPreflight().recommend(cfg, hw_tight_free)

        assert rec_roomy == rec_tight
        assert rec_roomy["max_model_len"] == 8192


class TestVllmPreflightCpu:
    """`config.num_gpus == 0` routes to `_recommend_cpu`, sized against system RAM;
    `_raw_host_ram_bytes` is patched so the math doesn't depend on the test machine's RAM."""

    _SMALL_MODEL_CFG: ClassVar[dict] = {
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "head_dim": 128,
        "torch_dtype": "float16",
        "max_position_embeddings": 2048,
    }

    def test_cpu_only_node_caps_at_mpe_and_clamps_gmu(self, tmp_path):
        # Roomy RAM: max_model_len caps at max_position_embeddings (2048), and
        # gpu_memory_utilization is sized to the clamped KV budget, not the raw headroom.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=256 * 1024**3):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == 2048
        # kv_per_token=32768B; 4 seqs * 2048 tokens = 256 MiB of KV, on top of
        # 1 GiB weights + 14% + the 2 GiB fixed overhead, / 256 GiB denom.
        assert rec["gpu_memory_utilization"] == 0.013

    def test_mixed_node_ignores_discoverable_gpus(self, tmp_path):
        # Same config, but the node-level pynvml view reports GPUs Ray didn't
        # actually assign to this num_gpus=0 deploy — must not affect sizing.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw_cpu_only = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        hw_mixed = HardwareProfile(
            gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3
        )
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=256 * 1024**3):
            rec_cpu_only = VllmPreflight().recommend(cfg, hw_cpu_only)
            rec_mixed = VllmPreflight().recommend(cfg, hw_mixed)
        assert rec_mixed == rec_cpu_only

    def test_unknown_context_length_falls_back_to_cap(self, tmp_path):
        cfg_json = {k: v for k, v in self._SMALL_MODEL_CFG.items() if k != "max_position_embeddings"}
        snapshot = _write_model_snapshot(tmp_path, config_json=cfg_json, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=1024 * 1024**3, available_ram_bytes=1024 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=1024 * 1024**3):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == 32768

    def test_weights_exceed_ram_returns_empty(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=64 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=32 * 1024**3, available_ram_bytes=32 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=32 * 1024**3):
            assert VllmPreflight().recommend(cfg, hw) == {}

    def test_undiscoverable_host_ram_returns_empty(self, tmp_path):
        # _raw_host_ram_bytes reads raw psutil total independently of hw.ram_bytes
        # (matches vLLM's own cgroup-blind denominator); a 0 return must not raise on divide.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=0):
            assert VllmPreflight().recommend(cfg, hw) == {}

    def test_explicit_gpu_memory_utilization_rejected_on_cpu_deploy(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        with pytest.raises(ValidationError, match="cannot be set"):
            _make_config(resolved_path=str(snapshot), num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.5})

    def test_sliding_window_uses_saturated_seq_bytes_for_gmu(self, tmp_path):
        """CPU sizing on a sliding-window model must clamp attention KV to the
        saturated per-sequence size, not `kv_per_token * suggested`."""
        from modelship.preflight.vllm import (
            _CPU_KV_SEQUENCES,
            _CPU_OVERHEAD_FIXED_BYTES,
            _OVERHEAD_WEIGHT_FRACTION,
            SlidingWindowInfo,
            _seq_kv_bytes,
        )

        cfg_json = {
            **self._SMALL_MODEL_CFG,
            "sliding_window": 64,
            "layer_types": ["sliding_attention"] * 6 + ["full_attention"] * 2,
        }
        weight_bytes = 1 * 1024**3
        snapshot = _write_model_snapshot(tmp_path, config_json=cfg_json, weight_bytes=weight_bytes)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)

        ram_bytes = 6 * 1024**3
        hw = HardwareProfile(ram_bytes=ram_bytes, available_ram_bytes=ram_bytes)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=ram_bytes):
            rec = VllmPreflight().recommend(cfg, hw)

        # Cap reached confirms _fit_len_with_sliding ran rather than the
        # uniform-attention path.
        assert rec["max_model_len"] == 2048

        kv_per_token = 2 * 8 * 128 * 2 * 8  # matches _SMALL_MODEL_CFG geometry
        sliding = SlidingWindowInfo(n_full_layers=2, n_sliding_layers=6, n_total_layers=8, window=64)
        saturated = _CPU_KV_SEQUENCES * _seq_kv_bytes(kv_per_token, sliding, 2048)
        linear = _CPU_KV_SEQUENCES * kv_per_token * 2048
        assert saturated < linear  # sanity: the two diverge in this scenario

        reserved = weight_bytes * (1 + _OVERHEAD_WEIGHT_FRACTION) + _CPU_OVERHEAD_FIXED_BYTES
        assert rec["gpu_memory_utilization"] == round((reserved + saturated) / ram_bytes, 3)
        wrong_gmu = round((reserved + linear) / ram_bytes, 3)
        assert rec["gpu_memory_utilization"] < wrong_gmu

    def test_non_hybrid_gmu_reserves_room_for_process_rss(self, tmp_path):
        """The KV pool vLLM ends up with is `gmu * total - process RSS`, so a gmu
        covering only the KV bytes leaves it short of the recommended
        max_model_len — silently, since preflight still returns a normal
        recommendation. Regression for the double-deducted weights."""
        # RSS / on-disk weight bytes, measured on vLLM 0.26.0+cpu with
        # Qwen/Qwen3-0.6B (1.4 GiB of weights, 2.12 GiB RSS after load).
        measured_rss_over_weights = 1.51

        cfg_json = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 40960,
        }
        weight_bytes = int(1.4 * 1024**3)
        snapshot = _write_model_snapshot(tmp_path, config_json=cfg_json, weight_bytes=weight_bytes)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)

        denom_ram = int(24.45 * 1024**3)
        ram_bytes = int(10.97 * 1024**3)
        hw = HardwareProfile(ram_bytes=ram_bytes, available_ram_bytes=ram_bytes)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=denom_ram):
            rec = VllmPreflight().recommend(cfg, hw)

        # A real recommendation, not a "no KV-cache budget" bailout.
        assert rec
        gmu = rec["gpu_memory_utilization"]
        suggested_len = rec["max_model_len"]

        kv_per_token = 2 * 8 * 128 * 2 * 28  # matches cfg_json geometry
        kv_bytes_needed = kv_per_token * suggested_len

        requested_pool = gmu * denom_ram
        estimated_rss = measured_rss_over_weights * weight_bytes
        actual_kv_pool = requested_pool - estimated_rss

        # vLLM's own _check_enough_kv_cache_memory raises below this.
        assert actual_kv_pool >= kv_bytes_needed


class TestDefaultGpuMemoryUtilization:
    """gpu_memory_utilization is derived, never a config field: a fractional
    num_gpus, else preflight's recommendation, else 0.9 GPU / 0.4 CPU."""

    def test_gpu_deploy_default(self):
        assert default_gpu_memory_utilization(_make_config(num_gpus=1)) == 0.9

    def test_cpu_deploy_default(self):
        assert default_gpu_memory_utilization(_make_config(num_gpus=0)) == 0.4

    def test_explicit_gmu_on_cpu_deploy_rejected(self):
        with pytest.raises(ValidationError, match="cannot be set"):
            _make_config(num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.6})

    def test_precedence_user_over_recommendation_over_default(self):
        # Mirrors vllm_infer: {**rec, **user_overrides}, then the derived gmu popped out.
        cfg = _make_config(num_gpus=0, vllm_kwargs={"max_model_len": 4096})
        user_overrides = cfg.vllm_engine_kwargs.model_dump(exclude_unset=True)
        recommendation = {"gpu_memory_utilization": 0.2, "max_model_len": 8192}
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=cfg.name)
        assert merged["max_model_len"] == 4096  # user wins over recommendation
        assert resolve_gpu_memory_utilization(cfg, merged.pop("gpu_memory_utilization")) == 0.2

    def test_default_survives_preflight_decline(self):
        assert resolve_gpu_memory_utilization(_make_config(num_gpus=0), None) == 0.4

    def test_fractional_share_wins_over_the_recommendation(self):
        assert resolve_gpu_memory_utilization(_make_config(num_gpus=0.5), 0.2) == 0.5


class TestMultimodal:
    @pytest.mark.parametrize(
        "model_cfg,expected",
        [
            ({"vision_config": {"image_size": 224}}, True),
            ({"audio_config": {}}, True),
            ({"architectures": ["LlavaForConditionalGeneration"]}, True),
            ({"architectures": ["Qwen2VLForConditionalGeneration"]}, True),
            ({"architectures": ["LlamaForCausalLM"]}, False),
            ({}, False),
        ],
    )
    def test_multimodal_detection(self, model_cfg, expected):
        from modelship.preflight.vllm import _is_multimodal

        assert _is_multimodal(model_cfg) == expected

    def test_mm_tokens_per_item_estimate(self):
        from modelship.preflight.vllm import _estimate_mm_tokens_per_item

        # 224 / 14 = 16 → 16² = 256 patches per image
        assert _estimate_mm_tokens_per_item({"vision_config": {"image_size": 224, "patch_size": 14}}) == 256
        # Missing geometry → None
        assert _estimate_mm_tokens_per_item({"vision_config": {}}) is None
        assert _estimate_mm_tokens_per_item({}) is None

    def test_multimodal_recommends_max_num_batched_tokens(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
                "architectures": ["LlavaForConditionalGeneration"],
                "vision_config": {"image_size": 336, "patch_size": 14},
            },
            weight_bytes=15 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_num_batched_tokens" in rec
        # 336/14 = 24 → 576 patches → 2x headroom = 1152, capped at the 8192 floor.
        # MNBT must match what the cudagraph budget was sized against; vLLM's chunked
        # prefill handles longer prompts, so it stays at the floor rather than scaling up.
        assert rec["max_num_batched_tokens"] == 8192

    def test_nested_text_config_is_unwrapped(self, tmp_path):
        # Geometry is read from a nested `text_config` (Gemma 3/4, LLaVA, Qwen2-VL,
        # etc.); GPUs are roomy so the budget produces a recommendation, not `{}`.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "architectures": ["Gemma3ForConditionalGeneration"],
                "torch_dtype": "bfloat16",
                "text_config": {
                    "num_hidden_layers": 48,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 16,
                    "hidden_size": 5120,
                    "head_dim": 160,
                    "max_position_embeddings": 32768,
                },
                "vision_config": {"image_size": 896, "patch_size": 14},
            },
            weight_bytes=19 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 2},
            num_gpus=2,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test"), GPUInfo(1, 24 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        # Produces a real recommendation, not `{}`.
        assert "max_model_len" in rec
        assert rec["max_model_len"] > 0
        # Also recognised as multimodal → max_num_batched_tokens is set.
        assert "max_num_batched_tokens" in rec

    @pytest.mark.parametrize(
        "nesting_key",
        ["text_config", "language_config", "llm_config", "language_model_config"],
    )
    def test_resolve_text_config_handles_known_nestings(self, nesting_key):
        from modelship.preflight.vllm import _resolve_text_config

        nested = {"num_hidden_layers": 32, "hidden_size": 4096}
        resolved = _resolve_text_config({nesting_key: nested, "architectures": ["X"]})
        assert resolved is nested

    def test_resolve_text_config_passes_through_top_level(self):
        from modelship.preflight.vllm import _resolve_text_config

        top = {"num_hidden_layers": 32, "hidden_size": 4096}
        assert _resolve_text_config(top) is top

    def test_text_only_no_max_num_batched_tokens(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
                "architectures": ["LlamaForCausalLM"],
            },
            weight_bytes=15 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_num_batched_tokens" not in rec


@pytest.mark.parametrize(
    "tp_size,num_kv_heads,expect_kv_shrinkage",
    [
        (1, 8, False),
        (2, 8, True),
        (4, 8, True),
        (3, 8, False),  # GQA edge case: 8 not divisible by 3, KV replicated
    ],
)
def test_kv_shrinks_per_gpu_only_when_tp_divides_heads(tp_size, num_kv_heads, expect_kv_shrinkage):
    """Unit-level: per-GPU KV bytes shrink by tp_size only when num_kv_heads is divisible."""
    from modelship.preflight.vllm import _divide_kv_by_tp

    kv_full = 100_000
    result = _divide_kv_by_tp(kv_full, {"num_key_value_heads": num_kv_heads}, tp_size)
    if expect_kv_shrinkage:
        assert result == kv_full / tp_size
    else:
        assert result == kv_full


class TestCudagraphNotModelled:
    """Cudagraph memory is deliberately not subtracted: preflight only sizes
    `max_model_len`, which doesn't bound what vLLM allocates."""

    def test_enforce_eager_does_not_change_max_model_len(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 1_000_000,
            },
            weight_bytes=15 * 1024**3,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        rec_graphs = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot)), hw)
        rec_eager = VllmPreflight().recommend(
            _make_config(resolved_path=str(snapshot), vllm_kwargs={"enforce_eager": True}), hw
        )
        assert rec_eager["max_model_len"] == rec_graphs["max_model_len"]

    @pytest.mark.parametrize("gpu_gib", [10.8, 40])
    def test_never_recommends_enforce_eager(self, tmp_path, gpu_gib):
        # 10.8 GiB is where the old cudagraph reclaim used to fire.
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        hw = HardwareProfile(gpus=[GPUInfo(0, int(gpu_gib * 1024**3), "test")])
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()):
            assert "enforce_eager" not in VllmPreflight().recommend(_make_config(resolved_path=str(snapshot)), hw)


# DeepSeek-V2-Lite's real config.json fields (deepseek-ai/DeepSeek-V2-Lite).
_MLA_CFG: dict = {
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "hidden_size": 2048,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 163840,
}


class TestMlaKvCache:
    """MLA stores one shared compressed latent per token, not per-head K/V —
    both the KV-bytes formula and TP sharding need a different rule."""

    def test_resolve_mla_detects_deepseek_v2_lite(self):
        from modelship.preflight.vllm import MLAInfo, _resolve_mla

        assert _resolve_mla(_MLA_CFG) == MLAInfo(
            kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128, num_heads=16
        )

    def test_resolve_mla_none_for_ordinary_gqa(self):
        from modelship.preflight.vllm import _resolve_mla

        assert _resolve_mla(_HYBRID_CFG) is None

    def test_kv_bytes_per_token_matches_deepseek_v2_lite(self):
        from modelship.preflight.vllm import _kv_bytes_per_token

        per_token, max_pos = _kv_bytes_per_token(_MLA_CFG, _MLA_CFG, _make_config())
        # (kv_lora_rank + qk_rope_head_dim) * dtype_bytes * num_layers = 576 * 2 * 27.
        assert per_token == 31104
        assert max_pos == 163840

    def test_kv_bytes_far_below_generic_per_head_formula(self):
        from modelship.preflight.vllm import _kv_bytes_per_token

        mla_per_token, _ = _kv_bytes_per_token(_MLA_CFG, _MLA_CFG, _make_config())
        generic_cfg = dict(_MLA_CFG)
        del generic_cfg["kv_lora_rank"]
        generic_per_token, _ = _kv_bytes_per_token(generic_cfg, generic_cfg, _make_config())
        assert mla_per_token is not None and generic_per_token is not None
        assert mla_per_token * 7 < generic_per_token

    def test_divide_kv_by_tp_does_not_shard_mla(self):
        # num_key_value_heads=16 divides tp_size=2 cleanly, but MLA's latent
        # is replicated per rank, not head-sharded — must not shrink.
        from modelship.preflight.vllm import _divide_kv_by_tp

        assert _divide_kv_by_tp(31104, _MLA_CFG, 2) == 31104

    def test_workspace_shrinks_with_tp(self):
        # The opposite rule to the latent cache above: the up-projection is
        # head-sharded, so the buffer scales down by tp.
        from modelship.preflight.vllm import _mla_chunked_prefill_workspace_bytes, _resolve_mla

        mla = _resolve_mla(_MLA_CFG)
        assert mla is not None
        at_1 = _mla_chunked_prefill_workspace_bytes(_MLA_CFG, _make_config(), 2, mla, 1)
        at_4 = _mla_chunked_prefill_workspace_bytes(_MLA_CFG, _make_config(), 2, mla, 4)
        assert at_4 == at_1 // 4

    def test_workspace_keeps_one_head_when_tp_exceeds_head_count(self):
        from modelship.preflight.vllm import _mla_chunked_prefill_workspace_bytes, _resolve_mla

        mla = _resolve_mla(_MLA_CFG)
        assert mla is not None
        assert _mla_chunked_prefill_workspace_bytes(_MLA_CFG, _make_config(), 2, mla, 64) > 0

    def test_workspace_floors_at_four_pages_per_seq(self):
        # Short context, many slots: the 4-pages-per-slot term beats 8 * max_model_len.
        from modelship.preflight.vllm import _mla_chunked_prefill_workspace_bytes, _resolve_mla

        mla = _resolve_mla(_MLA_CFG)
        assert mla is not None
        config = _make_config(vllm_kwargs={"max_model_len": 1024, "max_num_seqs": 1024})
        row_bytes = mla.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim) * 2
        assert _mla_chunked_prefill_workspace_bytes(_MLA_CFG, config, 2, mla, 1) == 65536 * row_bytes

    def test_workspace_floor_beats_the_row_cap(self):
        # The one-page-per-slot floor is applied after the cap, so it can exceed it.
        from modelship.preflight.vllm import _mla_chunked_prefill_workspace_bytes, _resolve_mla

        mla = _resolve_mla(_MLA_CFG)
        assert mla is not None
        config = _make_config(vllm_kwargs={"max_model_len": 1024, "max_num_seqs": 8192})
        row_bytes = mla.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim) * 2
        assert _mla_chunked_prefill_workspace_bytes(_MLA_CFG, config, 2, mla, 1) == 8192 * 16 * row_bytes

    def test_mla_chunked_prefill_workspace_matches_deepseek_v2_lite_oom(self):
        # Anchored against a real DeepSeek-V2-Lite-AWQ OOM: PyTorch reported
        # "Tried to allocate 512.00 MiB" for exactly this buffer.
        from modelship.preflight.vllm import _mla_chunked_prefill_workspace_bytes, _resolve_mla

        mla = _resolve_mla(_MLA_CFG)
        assert mla is not None
        workspace = _mla_chunked_prefill_workspace_bytes(_MLA_CFG, _make_config(), 2, mla, 1)
        assert workspace == 512 * 1024**2

    def test_recommend_lowers_gpu_memory_utilization_with_cudagraphs(self, tmp_path):
        # vLLM's own CUDA-graph memory profiler doesn't reserve room for this
        # workspace; preflight must shrink gpu_memory_utilization to compensate.
        snapshot = _write_model_snapshot(tmp_path, config_json=_MLA_CFG, weight_bytes=5 * 1024**3)
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot)), hw)
        assert "gpu_memory_utilization" in rec
        assert rec["gpu_memory_utilization"] < default_gpu_memory_utilization(_make_config())

    def test_recommend_reclaims_gpu_memory_utilization_at_higher_tp(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_MLA_CFG, weight_bytes=5 * 1024**3)
        gpus = [GPUInfo(i, 16 * 1024**3, "test") for i in range(4)]

        def _gmu(tp):
            cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"tensor_parallel_size": tp}, num_gpus=tp)
            return VllmPreflight().recommend(cfg, HardwareProfile(gpus=gpus[:tp]))["gpu_memory_utilization"]

        assert _gmu(4) > _gmu(1)

    def test_recommend_leaves_gpu_memory_utilization_unset_when_eager(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_MLA_CFG, weight_bytes=5 * 1024**3)
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"enforce_eager": True})
        assert "gpu_memory_utilization" not in VllmPreflight().recommend(cfg, hw)

    def test_recommend_leaves_gpu_memory_utilization_unset_for_ordinary_model(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=5 * 1024**3)
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot)), hw)
        assert "gpu_memory_utilization" not in rec


# Hybrid config mirroring Qwen3.5-4B: 32 layers, 8 full-attention + 24 linear.
_HYBRID_CFG: dict = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "head_dim": 128,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 262144,
}


def _mamba_info(**overrides):
    from modelship.preflight.vllm import MambaStateInfo

    base = dict(
        per_seq_state_bytes=49 * 1024**2,
        n_state_layers=24,
        n_full_attention_layers=8,
        n_total_layers=32,
        default_max_num_seqs=128,
    )
    base.update(overrides)
    return MambaStateInfo(**base)


class TestResolveMambaStateExecutorBackend:
    """The other hybrid tests patch `_resolve_mamba_state` out, so the real
    EngineArgs construction is only covered here."""

    def _captured_kwargs(self, num_gpus: int) -> dict:
        import vllm.engine.arg_utils

        from modelship.preflight.vllm import _resolve_mamba_state

        captured: dict = {}

        def fake_engine_args(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before config build")

        with patch.object(vllm.engine.arg_utils, "EngineArgs", fake_engine_args):
            assert _resolve_mamba_state(_make_config(num_gpus=num_gpus), "/nonexistent") is None
        return captured

    def test_multi_slot_selects_ray_backend(self):
        assert self._captured_kwargs(num_gpus=2)["distributed_executor_backend"] == "ray"

    def test_single_slot_leaves_backend_unset(self):
        assert self._captured_kwargs(num_gpus=1)["distributed_executor_backend"] is None


class TestApplyHybridFit:
    """Pure arithmetic of the device-agnostic fit ladder — no vLLM, no config
    building. `kv_pool` is the bytes available for KV cache + mamba state."""

    PER_SEQ = 50 * 1024**2  # 50 MiB per concurrent slot
    KV_PER_TOKEN = 32768  # 32 KiB/token (full-attention layers only)
    TARGET = 100_000

    def _fit(self, kv_pool, user_seqs=None):
        from modelship.preflight.vllm import _apply_hybrid_fit

        return _apply_hybrid_fit("test", kv_pool, self.PER_SEQ, self.KV_PER_TOKEN, self.TARGET, user_seqs, 128)

    def test_tight_pool_floors_seqs_and_trims_context(self):
        rec = self._fit(1 * 1024**3)  # 1 GiB: full 100k context can't fit
        assert rec["max_num_seqs"] == 8  # floor concurrency
        assert 0 < rec["max_model_len"] < self.TARGET  # context trimmed
        assert rec["max_model_len"] % 16 == 0

    def test_roomy_pool_keeps_context_and_climbs_seqs(self):
        rec = self._fit(10 * 1024**3)  # 10 GiB: full context fits with surplus
        assert rec["max_model_len"] == self.TARGET  # capability preserved
        assert rec["max_num_seqs"] == 128  # surplus spent, capped at vLLM default

    def test_user_pinned_seqs_sizes_context_and_omits_seq_recommendation(self):
        rec = self._fit(10 * 1024**3, user_seqs=64)
        assert rec["max_model_len"] == self.TARGET
        assert "max_num_seqs" not in rec  # honor the user's contract, don't recommend

    def test_pool_too_small_for_floor_state_returns_empty(self):
        # 0.3 GiB < mamba state at the floor of 8 seqs (8 * 50 MiB = 0.39 GiB).
        assert self._fit(int(0.3 * 1024**3)) == {}


class TestCorrectKvForHybrid:
    def test_scales_by_full_attention_fraction(self):
        from modelship.preflight.vllm import _correct_kv_for_hybrid

        # Only 8 of 32 layers hold a token-growing KV cache.
        assert _correct_kv_for_hybrid(32000, _mamba_info()) == 32000 * 8 / 32


class TestHybridIntegration:
    """End-to-end through recommend(), with `_resolve_mamba_state` patched to a
    synthetic MambaStateInfo so no real vLLM config gets built."""

    def test_gpu_hybrid_floors_seqs_and_trims_vs_dense_baseline(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(
            resolved_path=str(snapshot),
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, int(15.45 * 1024**3), "test")])
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()):
            hybrid = VllmPreflight().recommend(cfg, hw)
        # Same model/GPU but treated as a plain transformer (no state term).
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=None):
            dense = VllmPreflight().recommend(cfg, hw)
        assert hybrid["max_num_seqs"] == 8
        assert 0 < hybrid["max_model_len"] < _HYBRID_CFG["max_position_embeddings"]
        assert "max_num_seqs" not in dense  # non-hybrid never emits it

    def test_gpu_roomy_keeps_full_context_and_climbs(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot))
        hw = HardwareProfile(gpus=[GPUInfo(0, 40 * 1024**3, "test")])
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == _HYBRID_CFG["max_position_embeddings"]
        assert rec["max_num_seqs"] > 8

    def test_cpu_auto_gmu_hybrid_folds_state_into_gmu(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=16 * 1024**3, available_ram_bytes=16 * 1024**3)
        with (
            patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=16 * 1024**3),
            patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()),
        ):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_num_seqs"] == 8  # tight RAM → floor
        assert 0 < rec["max_model_len"] < _HYBRID_CFG["max_position_embeddings"]
        assert "gpu_memory_utilization" in rec  # auto path still sizes the fraction

    def test_explicit_gpu_memory_utilization_rejected_on_cpu_hybrid_deploy(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        with pytest.raises(ValidationError, match="cannot be set"):
            _make_config(resolved_path=str(snapshot), num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.5})


def _gemma4_shaped_config(*, layer_types: bool) -> dict:
    """Gemma-4-12B geometry: 40 sliding(1024) + 8 full out of 48 layers."""
    cfg = {
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 3840,
        "head_dim": 256,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 262144,
        "sliding_window": 1024,
    }
    if layer_types:
        cfg["layer_types"] = ["sliding_attention"] * 40 + ["full_attention"] * 8
    return cfg


class TestResolveSlidingWindow:
    """Layer split read from config.json alone."""

    def test_reads_layer_types(self):
        from modelship.preflight.vllm import _resolve_sliding_window

        sw = _resolve_sliding_window(_gemma4_shaped_config(layer_types=True))
        assert (sw.n_sliding_layers, sw.n_full_layers, sw.n_total_layers) == (40, 8, 48)
        assert sw.window == 1024

    def test_falls_back_to_sliding_window_pattern(self):
        """`sliding_window_pattern` means every Nth layer is full attention."""
        from modelship.preflight.vllm import _resolve_sliding_window

        sw = _resolve_sliding_window({"num_hidden_layers": 26, "sliding_window": 4096, "sliding_window_pattern": 2})
        assert (sw.n_sliding_layers, sw.n_full_layers) == (13, 13)

    def test_bare_sliding_window_means_every_layer_slides(self):
        from modelship.preflight.vllm import _resolve_sliding_window

        sw = _resolve_sliding_window({"num_hidden_layers": 32, "sliding_window": 4096})
        assert (sw.n_sliding_layers, sw.n_full_layers) == (32, 0)

    def test_uniform_full_attention_returns_none(self):
        from modelship.preflight.vllm import _resolve_sliding_window

        assert _resolve_sliding_window({"num_hidden_layers": 32}) is None
        assert _resolve_sliding_window({"num_hidden_layers": 32, "sliding_window": None}) is None

    def test_use_sliding_window_false_is_respected(self):
        """A `sliding_window` value is present but switched off."""
        from modelship.preflight.vllm import _resolve_sliding_window

        cfg = {"num_hidden_layers": 32, "sliding_window": 4096, "use_sliding_window": False}
        assert _resolve_sliding_window(cfg) is None

    def test_non_sliding_layer_types_count_as_full(self):
        """Non-sliding `layer_types` count as full, keeping the estimate conservative."""
        from modelship.preflight.vllm import _resolve_sliding_window

        cfg = {
            "num_hidden_layers": 4,
            "sliding_window": 8192,
            "layer_types": ["chunked_attention", "chunked_attention", "sliding_attention", "full_attention"],
        }
        sw = _resolve_sliding_window(cfg)
        assert (sw.n_sliding_layers, sw.n_full_layers) == (1, 3)


class TestFitLenWithSliding:
    def test_chosen_length_fits_and_one_more_token_does_not(self):
        from modelship.preflight.vllm import _fit_len_with_sliding, _resolve_sliding_window, _seq_kv_bytes

        sw = _resolve_sliding_window(_gemma4_shaped_config(layer_types=True))
        kv_per_token = 2 * 8 * 256 * 2 * 48
        budget = 8.0 * 1024**3
        length = _fit_len_with_sliding(budget, kv_per_token, sw, 262144)
        assert _seq_kv_bytes(kv_per_token, sw, length) <= budget
        assert _seq_kv_bytes(kv_per_token, sw, length + 1) > budget

    def test_all_sliding_is_bounded_only_by_the_context_cap(self):
        from modelship.preflight.vllm import SlidingWindowInfo, _fit_len_with_sliding

        sw = SlidingWindowInfo(n_full_layers=0, n_sliding_layers=32, n_total_layers=32, window=4096)
        assert _fit_len_with_sliding(64 * 1024**3, 2 * 8 * 128 * 2 * 32, sw, 32768) == 32768

    def test_budget_below_the_window_falls_back_to_per_token_growth(self):
        from modelship.preflight.vllm import SlidingWindowInfo, _fit_len_with_sliding

        sw = SlidingWindowInfo(n_full_layers=8, n_sliding_layers=40, n_total_layers=48, window=1_000_000)
        kv_per_token = 48 * 1024
        # Window unreachable, so every layer still grows: budget / kv_per_token.
        assert _fit_len_with_sliding(48 * 1024 * 500, kv_per_token, sw, 262144) == 500

    def test_mixed_layers_are_capped_even_with_a_huge_budget(self):
        """A generous budget must not push the result past ctx_cap — this is
        the case an unknown max_position_embeddings falls back to."""
        from modelship.preflight.vllm import SlidingWindowInfo, _fit_len_with_sliding

        sw = SlidingWindowInfo(n_full_layers=8, n_sliding_layers=40, n_total_layers=48, window=1024)
        kv_per_token = 2 * 8 * 256 * 2 * 48
        assert _fit_len_with_sliding(1024 * 1024**3, kv_per_token, sw, 32768) == 32768

    def test_below_window_result_is_still_capped(self):
        """A sliding window at or beyond ctx_cap must not let the below-window
        branch return a length past the cap."""
        from modelship.preflight.vllm import SlidingWindowInfo, _fit_len_with_sliding

        sw = SlidingWindowInfo(n_full_layers=8, n_sliding_layers=40, n_total_layers=48, window=32768)
        kv_per_token = 48 * 1024
        budget = 1_611_000_000  # below-window branch; pre-fix this returned 32775
        assert _fit_len_with_sliding(budget, kv_per_token, sw, 32768) == 32768


class TestSlidingWindowIntegration:
    """The same model with and without `layer_types` present."""

    def test_layer_types_unlocks_far_more_context(self, tmp_path):
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test"), GPUInfo(1, 16 * 1024**3, "test")])

        def _rec(config_json, subdir):
            snapshot = _write_model_snapshot(tmp_path / subdir, config_json=config_json, weight_bytes=10 * 1024**3)
            cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"tensor_parallel_size": 2}, num_gpus=2)
            return VllmPreflight().recommend(cfg, hw)["max_model_len"]

        (tmp_path / "with").mkdir()
        (tmp_path / "without").mkdir()
        with_sw = _rec(_gemma4_shaped_config(layer_types=True), "with")
        without_sw = _rec(
            {k: v for k, v in _gemma4_shaped_config(layer_types=False).items() if k != "sliding_window"}, "without"
        )

        assert with_sw > without_sw * 4
        assert with_sw % 16 == 0

    def test_all_full_layer_types_matches_no_sliding_keys(self, tmp_path):
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        base = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 32768,
        }

        def _rec(config_json, subdir):
            (tmp_path / subdir).mkdir()
            snapshot = _write_model_snapshot(tmp_path / subdir, config_json=config_json, weight_bytes=14 * 1024**3)
            cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"tensor_parallel_size": 1})
            return VllmPreflight().recommend(cfg, hw)["max_model_len"]

        uniform = _rec(base, "uniform")
        all_full = _rec({**base, "sliding_window": 4096, "layer_types": ["full_attention"] * 32}, "allfull")
        assert all_full == uniform
