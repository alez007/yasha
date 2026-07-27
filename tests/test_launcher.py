import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from modelship import launcher


class TestGuardPythonVersion:
    def test_matching_version_passes(self):
        with patch.object(launcher.sys, "version_info", (3, 12, 10, "final", 0)):
            launcher._guard_python_version()  # no raise

    def test_mismatched_version_exits(self):
        with (
            patch.object(launcher.sys, "version_info", (3, 11, 4, "final", 0)),
            pytest.raises(SystemExit) as exc,
        ):
            launcher._guard_python_version()
        assert exc.value.code == 1


class TestCheckLoaderCapabilities:
    def _write_config(self, tmp_path, loader):
        path = tmp_path / "models.yaml"
        path.write_text(f"models:\n  - name: m\n    loader: {loader}\n    model: x\n")
        return str(path)

    def test_no_config_path_is_noop(self):
        launcher._check_loader_capabilities(None)  # no raise

    def test_missing_file_is_noop(self, tmp_path):
        launcher._check_loader_capabilities(str(tmp_path / "nope.yaml"))  # no raise

    def test_llama_server_loader_never_gated(self, tmp_path):
        config = self._write_config(tmp_path, "llama_server")
        with patch("modelship.launcher.importlib.util.find_spec") as mock_find:
            launcher._check_loader_capabilities(config)
        mock_find.assert_not_called()

    def test_passes_when_module_importable(self, tmp_path):
        config = self._write_config(tmp_path, "vllm")
        with patch("modelship.launcher.importlib.util.find_spec", return_value=MagicMock()):
            launcher._check_loader_capabilities(config)  # no raise

    def test_exits_when_module_missing(self, tmp_path):
        config = self._write_config(tmp_path, "vllm")
        with (
            patch("modelship.launcher.importlib.util.find_spec", return_value=None),
            pytest.raises(SystemExit) as exc,
        ):
            launcher._check_loader_capabilities(config)
        assert exc.value.code == 1


class TestProvisionMacosLlamaServer:
    def test_short_circuits_when_env_set(self):
        with patch.dict(os.environ, {"MSHIP_LLAMA_SERVER_BIN": "/existing/bin"}, clear=True):
            assert launcher._provision_macos_llama_server() == "/existing/bin"

    def test_sets_env_on_success(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.resolve_llama_server_bin", return_value="/resolved/bin"),
        ):
            path = launcher._provision_macos_llama_server()
            assert path == "/resolved/bin"
            assert os.environ["MSHIP_LLAMA_SERVER_BIN"] == "/resolved/bin"

    def test_warns_and_returns_none_on_failure(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("modelship.provision.llama_server.resolve_llama_server_bin", side_effect=RuntimeError("boom")),
        ):
            assert launcher._provision_macos_llama_server() is None
        assert "MSHIP_LLAMA_SERVER_BIN" not in os.environ


class TestCmdDeploy:
    def test_forwards_argv_to_driver_after_gates(self):
        argv = ["--config", "models.yaml", "--reconcile"]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_check_loader_capabilities") as mock_gate,
            patch.object(launcher, "_guard_python_version") as mock_guard,
            patch("modelship.driver.main") as mock_driver_main,
        ):
            launcher._cmd_deploy(argv)

        mock_guard.assert_called_once()
        mock_gate.assert_called_once()
        mock_driver_main.assert_called_once_with(argv)

    def test_metal_accelerator_triggers_provisioning(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="metal"),
            patch.object(launcher, "_provision_macos_llama_server") as mock_provision,
            patch.object(launcher, "_check_loader_capabilities"),
            patch.object(launcher, "_guard_python_version"),
            patch("modelship.driver.main"),
        ):
            launcher._cmd_deploy([])
        mock_provision.assert_called_once()

    def test_cpu_accelerator_skips_provisioning(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "_provision_macos_llama_server") as mock_provision,
            patch.object(launcher, "_check_loader_capabilities"),
            patch.object(launcher, "_guard_python_version"),
            patch("modelship.driver.main"),
        ):
            launcher._cmd_deploy([])
        mock_provision.assert_not_called()

    def test_driver_not_imported_when_guard_exits(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/mship-test-cache"),
            patch.object(launcher, "_guard_python_version", side_effect=SystemExit(1)),
            patch("modelship.driver.main") as mock_driver_main,
            pytest.raises(SystemExit),
        ):
            launcher._cmd_deploy([])
        mock_driver_main.assert_not_called()


class TestMain:
    def test_no_args_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            launcher.main([])
        assert exc.value.code == 2

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            launcher.main(["bogus"])
        assert exc.value.code == 2

    def test_deploy_dispatches_to_cmd_deploy(self):
        with patch.object(launcher, "_cmd_deploy") as mock_cmd:
            launcher.main(["deploy", "--reconcile"])
        mock_cmd.assert_called_once_with(["--reconcile"])

    def test_info_dispatches_to_cmd_info(self):
        with patch.object(launcher, "_cmd_info") as mock_cmd:
            launcher.main(["info"])
        mock_cmd.assert_called_once()

    def test_defaults_to_sys_argv(self):
        with (
            patch.object(sys, "argv", ["mship", "info"]),
            patch.object(launcher, "_cmd_info") as mock_cmd,
        ):
            launcher.main()
        mock_cmd.assert_called_once()


class TestCmdInfo:
    def test_prints_cpu_details(self, capsys):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "detect_accelerator", return_value="cpu"),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/cache"),
        ):
            launcher._cmd_info()
        out = capsys.readouterr().out
        assert "accelerator: cpu" in out
        assert "cache: /tmp/cache" in out
        assert "llama-server: unset" in out

    def test_prints_metal_provisioned_path(self, capsys):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(launcher, "detect_accelerator", return_value="metal"),
            patch.object(launcher, "resolve_cache_root", return_value="/tmp/cache"),
            patch.object(launcher, "_provision_macos_llama_server", return_value="/tmp/cache/llama-server.sh"),
        ):
            launcher._cmd_info()
        out = capsys.readouterr().out
        assert "llama-server: /tmp/cache/llama-server.sh" in out
