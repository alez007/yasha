"""Tests for start.py CLI argument parsing and helpers."""

import os
from pathlib import Path
from unittest.mock import patch

from start import _rand_suffix, build_actor_options, parse_args, resolve_plugin_wheel

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.redeploy is False
        assert args.gateway_name is None
        assert args.ray_cluster_address is None
        assert args.ray_redis_port is None
        assert args.use_existing_ray_cluster is None

    def test_redeploy_flag(self):
        args = parse_args(["--redeploy"])
        assert args.redeploy is True

    def test_config_path(self):
        args = parse_args(["--config", "/some/path/models.yaml"])
        assert args.config == "/some/path/models.yaml"

    def test_gateway_name(self):
        args = parse_args(["--gateway-name", "my-gateway"])
        assert args.gateway_name == "my-gateway"

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--config",
                "llm.yaml",
                "--gateway-name",
                "llm-api",
                "--redeploy",
                "--ray-cluster-address",
                "10.0.0.1",
                "--ray-redis-port",
                "6379",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.redeploy is True
        assert args.ray_cluster_address == "10.0.0.1"
        assert args.ray_redis_port == "6379"
        assert args.use_existing_ray_cluster is True


class TestRandSuffix:
    def test_default_length(self):
        suffix = _rand_suffix()
        assert len(suffix) == 5

    def test_custom_length(self):
        suffix = _rand_suffix(10)
        assert len(suffix) == 10

    def test_chars_are_alphanumeric_lowercase(self):
        for _ in range(50):
            suffix = _rand_suffix()
            assert all(c.islower() or c.isdigit() for c in suffix)


class TestBuildActorOptions:
    def test_basic_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            num_cpus=2,
        )
        opts = build_actor_options(config)
        assert opts["num_gpus"] == 1
        assert opts["num_cpus"] == 2
        assert "env_vars" in opts["runtime_env"]
        assert "pip" not in opts["runtime_env"]

    def test_with_plugin_wheel(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.custom,
            plugin="myplugin",
        )
        wheel_path = Path("/tmp/myplugin-0.1.0-py3-none-any.whl")
        opts = build_actor_options(config, plugin_wheel=wheel_path)
        assert opts["runtime_env"]["pip"] == [str(wheel_path)]

    def test_llama_cpp_force_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_cpp,
            num_gpus=1,
        )
        opts = build_actor_options(config)
        assert opts["num_gpus"] == 0


class TestResolvePluginWheel:
    def test_resolves_latest_wheel(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "myplugin-0.1.0-py3-none-any.whl").touch()
        (wheel_dir / "myplugin-0.1.1-py3-none-any.whl").touch()

        with patch.dict(os.environ, {"MSHIP_PLUGIN_WHEEL_DIR": str(wheel_dir)}):
            wheel = resolve_plugin_wheel("myplugin")
            assert wheel.name == "myplugin-0.1.1-py3-none-any.whl"
            assert wheel.is_absolute()

    def test_raises_if_no_wheel(self, tmp_path):
        import pytest

        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()

        with (
            patch.dict(os.environ, {"MSHIP_PLUGIN_WHEEL_DIR": str(wheel_dir)}),
            pytest.raises(RuntimeError, match="No wheel found for plugin 'myplugin'"),
        ):
            resolve_plugin_wheel("myplugin")
