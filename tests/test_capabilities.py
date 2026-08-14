import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from modelship.deploy.capabilities import (
    LOADER_MODULES,
    deployment_capability_resources,
    node_capability_resources,
)
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase


def _config(loader: ModelLoader) -> ModelshipModelConfig:
    image_loaders = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
    usecase = ModelUsecase.image if loader in image_loaders else ModelUsecase.generate
    return ModelshipModelConfig(name="m", model="org/m", usecase=usecase, loader=loader)


class TestNodeCapabilityResources:
    def test_all_loader_modules_importable(self):
        with patch("importlib.util.find_spec", return_value=object()), patch.dict(os.environ, {}, clear=True):
            resources = node_capability_resources()
        for loader in LOADER_MODULES:
            assert resources[f"mship_{loader}"] == 1

    def test_diffusers_missing_on_cpu_image(self):
        """The `cpu` extra has no diffusers — find_spec returns None for it only."""

        def fake_find_spec(name):
            return None if name == "diffusers" else object()

        with patch("importlib.util.find_spec", side_effect=fake_find_spec), patch.dict(os.environ, {}, clear=True):
            resources = node_capability_resources()
        assert "mship_diffusers" not in resources
        assert resources["mship_vllm"] == 1
        assert resources["mship_stable_diffusion_cpp"] == 1

    def test_no_modules_importable_and_no_llama_server(self):
        with patch("importlib.util.find_spec", return_value=None), patch.dict(os.environ, {}, clear=True):
            resources = node_capability_resources()
        assert resources == {}

    def test_env_override_replaces_probe_wholesale(self):
        override = json.dumps({"mship_nemo": 1})
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch.dict(os.environ, {"MSHIP_NODE_CAPABILITIES": override}, clear=True),
        ):
            resources = node_capability_resources()
        assert resources == {"mship_nemo": 1.0}

    def test_llama_server_available_when_binary_present(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\n")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": str(binary)}, clear=True),
        ):
            resources = node_capability_resources()
        assert resources == {"mship_llama_server": 1}

    def test_llama_server_wrapper_with_missing_target_is_unavailable(self, tmp_path):
        """The `thin` image bakes the wrapper script unconditionally but ships no
        real binary behind it — the probe must resolve the exec target, not just
        check the wrapper file's own existence."""
        missing_target = tmp_path / "nonexistent" / "llama-server"
        wrapper = tmp_path / "llama-server.sh"
        wrapper.write_text(f'#!/bin/sh\nexport LD_LIBRARY_PATH="{tmp_path}"\nexec "{missing_target}" "$@"\n')
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": str(wrapper)}, clear=True),
        ):
            resources = node_capability_resources()
        assert "mship_llama_server" not in resources

    def test_llama_server_wrapper_with_present_target_is_available(self, tmp_path):
        real_bin = tmp_path / "llama-server"
        real_bin.write_text("binary")
        wrapper = tmp_path / "llama-server.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{real_bin}" "$@"\n')
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": str(wrapper)}, clear=True),
        ):
            resources = node_capability_resources()
        assert resources == {"mship_llama_server": 1}

    def test_llama_server_unquoted_wrapper_with_missing_target_is_unavailable(self, tmp_path):
        """The Dockerfile's actual llama-server.sh emits an unquoted exec target
        (unlike the bootstrapper's quoted wrapper form) — the regex must handle both."""
        missing_target = tmp_path / "nonexistent" / "llama-server"
        wrapper = tmp_path / "llama-server.sh"
        wrapper.write_text(f'#!/bin/sh\nexport LD_LIBRARY_PATH="{tmp_path}"\nexec {missing_target} "$@"\n')
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": str(wrapper)}, clear=True),
        ):
            resources = node_capability_resources()
        assert "mship_llama_server" not in resources

    def test_llama_server_unquoted_wrapper_with_present_target_is_available(self, tmp_path):
        real_bin = tmp_path / "llama-server"
        real_bin.write_text("binary")
        wrapper = tmp_path / "llama-server.sh"
        wrapper.write_text(f'#!/bin/sh\nexec {real_bin} "$@"\n')
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": str(wrapper)}, clear=True),
        ):
            resources = node_capability_resources()
        assert resources == {"mship_llama_server": 1}

    def test_llama_server_unset_env_is_unavailable(self):
        with patch("importlib.util.find_spec", return_value=None), patch.dict(os.environ, {}, clear=True):
            resources = node_capability_resources()
        assert "mship_llama_server" not in resources

    def test_llama_server_nonexistent_path_is_unavailable(self):
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": "/does/not/exist"}, clear=True),
        ):
            resources = node_capability_resources()
        assert "mship_llama_server" not in resources


class TestDeploymentCapabilityResources:
    @pytest.mark.parametrize(
        "loader",
        [ModelLoader.vllm, ModelLoader.diffusers, ModelLoader.llama_server, ModelLoader.stable_diffusion_cpp],
    )
    def test_loader_requests_its_capability(self, loader):
        resources = deployment_capability_resources(_config(loader))
        assert resources == {f"mship_{loader}": 0.001}


class TestCapabilitiesModuleIsRayFree:
    def test_import_does_not_pull_in_ray(self):
        """modelship/launcher.py imports LOADER_MODULES from this module before
        resolve_ray_auth_env runs — a top-level `import ray` here (even
        transitively, e.g. via modelship.infer.infer_config) would silently
        break that discipline. Run in a subprocess so this test's own import of
        `modelship.infer.infer_config` above can't taint the check."""
        result = subprocess.run(
            [sys.executable, "-c", "import modelship.deploy.capabilities, sys; assert 'ray' not in sys.modules"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
