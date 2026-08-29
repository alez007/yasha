"""Tests for the LlamaServerPreflight estimator, which shells out to `llama
fit-params` rather than modelling GGUF/VRAM math itself."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.preflight import GPUInfo, HardwareProfile, merge_with_user_overrides, run_preflight
from modelship.preflight.llama_cpp import LlamaServerPreflight


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


def _write_fake_binary(tmp_path: Path) -> str:
    path = tmp_path / "llama"
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _fit_result(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestLlamaServerPreflightDeclines:
    def test_no_resolved_path_skips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=None)
        with patch("subprocess.run") as run:
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        run.assert_not_called()
        assert rec == {}

    def test_non_gguf_resolved_path_skips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        non_gguf = tmp_path / "model.safetensors"
        non_gguf.write_bytes(b"\0" * 1024)
        cfg = _make_config(resolved_path=str(non_gguf))
        with patch("subprocess.run") as run:
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        run.assert_not_called()
        assert rec == {}

    def test_missing_binary_skips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", str(tmp_path / "nonexistent"))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        with patch("subprocess.run") as run:
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        run.assert_not_called()
        assert rec == {}

    def test_everything_pinned_skips_the_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_gpus=2,
            llama_server_kwargs={"n_ctx": 4096, "n_gpu_layers": 20, "tensor_split": [1.0, 1.0]},
        )
        with patch("subprocess.run") as run:
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        run.assert_not_called()
        assert rec == {}


class TestLlamaServerPreflightParsing:
    def test_full_fit_maps_ctx_and_layers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=2)
        with patch("subprocess.run", return_value=_fit_result("-c 190720 -ngl 65 -ts 31,34\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec["n_ctx"] == 190720
        assert rec["n_gpu_layers"] == 65
        assert rec["tensor_split"] == [31.0, 34.0]

    def test_zero_ctx_and_negative_ngl_round_trip_as_auto(self, tmp_path, monkeypatch):
        # A model that fits comfortably gets "-c 0 -ngl -1" back — 0 means "the
        # model's own maximum" and round-trips; -1 means "let llama decide", omitted.
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", return_value=_fit_result("-c 0 -ngl -1\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec["n_ctx"] == 0
        assert "n_gpu_layers" not in rec

    def test_ctx_divided_across_parallel_slots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1, llama_server_kwargs={"parallel": 4}
        )
        with patch("subprocess.run", return_value=_fit_result("-c 16384 -ngl 33\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec["n_ctx"] == 4096

    def test_per_slot_below_minimum_declines(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1, llama_server_kwargs={"parallel": 64}
        )
        with patch("subprocess.run", return_value=_fit_result("-c 16384 -ngl 33\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec == {}

    def test_nonzero_exit_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", return_value=_fit_result(returncode=1, stderr="error: invalid argument")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec == {}

    def test_timeout_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["llama"], timeout=30)):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec == {}

    def test_unparseable_output_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", return_value=_fit_result("not the expected format\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec == {}


class TestLlamaServerPreflightArgs:
    def test_cpu_only_passes_dev_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=0)
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl -1\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        args = run.call_args[0][0]
        assert args[args.index("-dev") + 1] == "none"

    def test_gpu_deploy_omits_dev_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 10\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert "-dev" not in run.call_args[0][0]

    def test_pinned_n_ctx_forwarded_as_total(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_gpus=1,
            llama_server_kwargs={"n_ctx": 8192, "parallel": 2},
        )
        with patch("subprocess.run", return_value=_fit_result("-c 16384 -ngl 18 -ts 10,8\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        args = run.call_args[0][0]
        assert args[args.index("-c") + 1] == "16384"

    def test_pinned_n_gpu_layers_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1, llama_server_kwargs={"n_gpu_layers": 20}
        )
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 20\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        args = run.call_args[0][0]
        assert args[args.index("-ngl") + 1] == "20"

    def test_pinned_tensor_split_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_gpus=2,
            llama_server_kwargs={"tensor_split": [3.0, 1.0]},
        )
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 18\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        args = run.call_args[0][0]
        assert args[args.index("-ts") + 1] == "3.0,1.0"

    def test_engine_tunables_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_gpus=1,
            llama_server_kwargs={"ubatch_size": 2048, "flash_attn": "off", "cache_type_k": "q8_0"},
        )
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 10\n")) as run:
            LlamaServerPreflight().recommend(cfg, HardwareProfile())
        args = run.call_args[0][0]
        assert args[args.index("-ub") + 1] == "2048"
        assert args[args.index("-fa") + 1] == "off"
        assert args[args.index("-ctk") + 1] == "q8_0"

    def test_fractional_num_gpus_converts_share_to_margin(self, tmp_path, monkeypatch):
        # Declared share is half of a 16 GiB GPU with 10 GiB currently free:
        # margin = free - share + 1024 MiB.
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=0.5)
        hw = HardwareProfile(gpus=[GPUInfo(0, 10 * 1024**3, "test", total_bytes=16 * 1024**3)])
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 10\n")) as run:
            LlamaServerPreflight().recommend(cfg, hw)
        args = run.call_args[0][0]
        expected = (10 * 1024) - (8 * 1024) + 1024
        assert args[args.index("-fitt") + 1] == str(expected)

    def test_whole_gpu_uses_flat_margin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=2)
        hw = HardwareProfile(gpus=[GPUInfo(0, 10 * 1024**3, "a"), GPUInfo(1, 8 * 1024**3, "b")])
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 10\n")) as run:
            LlamaServerPreflight().recommend(cfg, hw)
        args = run.call_args[0][0]
        assert args[args.index("-fitt") + 1] == "1024"


class TestLlamaServerPreflightThreads:
    def test_threads_recommended_even_when_fit_declines(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", str(tmp_path / "nonexistent"))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=8)
        rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec == {"threads": 8}

    def test_default_fractional_num_cpus_has_no_threads_recommendation(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))  # num_cpus defaults to 0.1
        rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert "threads" not in rec

    def test_user_set_threads_wins_at_merge_level(self):
        merged = merge_with_user_overrides({"threads": 4}, {"threads": 16}, model_name="m")
        assert merged["threads"] == 16

    def test_threads_declined_when_it_would_undercut_parallel_slots(self, tmp_path, monkeypatch):
        # num_cpus=2 with parallel=4: capping to 2 threads would starve the 4 concurrent
        # slots of compute — decline and let llama-server keep all cores.
        monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=2, llama_server_kwargs={"parallel": 4}
        )
        rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert "threads" not in rec

    def test_threads_recommended_when_it_covers_parallel_slots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=4, llama_server_kwargs={"parallel": 4}
        )
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl -1\n")):
            rec = LlamaServerPreflight().recommend(cfg, HardwareProfile())
        assert rec["threads"] == 4


class TestRunPreflightDispatch:
    def test_run_preflight_dispatches_to_llama_server(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MSHIP_LLAMA_SERVER_BIN", _write_fake_binary(tmp_path))
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        with patch("subprocess.run", return_value=_fit_result("-c 4096 -ngl 10\n")):
            rec = run_preflight(cfg, HardwareProfile())
        assert rec["n_ctx"] == 4096
