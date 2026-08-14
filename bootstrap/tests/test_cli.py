import os
from unittest.mock import patch

import pytest

from mship_bootstrap import cli, paths
from mship_bootstrap.variants import VARIANTS


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    """Everything past the gates stubbed, so tests exercise argv and env only."""
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_VARIANT", raising=False)
    monkeypatch.delenv("MSHIP_LLAMA_SERVER_BIN", raising=False)
    with (
        patch.object(cli.gates, "check_platform"),
        patch.object(cli.gates, "check_hardware"),
        patch.object(cli.uv_binary, "ensure_uv", return_value="/usr/bin/uv"),
        patch.object(cli.engine, "provision", return_value="/env/bin/python"),
        patch.object(cli.llama_cpp, "provision", return_value="/builds/llama-server.sh"),
        patch.object(cli.llama_cpp, "warn_if_no_cuda_device"),
        patch("os.execve") as execve,
    ):
        yield execve


class TestUsage:
    def test_no_args_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main([])
        assert exc.value.code == 2

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["serve"])
        assert exc.value.code == 2


class TestVariantRequired:
    def test_deploy_without_a_variant_lists_the_options(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--config", "models.yaml"])
        assert "no variant selected" in str(exc.value)

    def test_two_variants_is_refused(self):
        with pytest.raises(SystemExit) as exc:
            cli.main(["deploy", "--cpu", "--cuda"])
        assert "pick one variant" in str(exc.value)


class TestExec:
    def test_execs_the_engine_via_module(self, provisioned):
        cli.main(["deploy", "--cpu", "--config", "models.yaml"])
        python, args, _env = provisioned.call_args[0]
        assert python == "/env/bin/python"
        assert args == ["/env/bin/python", "-m", "modelship.launcher", "deploy", "--config", "models.yaml"]

    def test_variant_flag_is_not_passed_to_the_engine(self, provisioned):
        cli.main(["deploy", "--cuda", "--reconcile"])
        assert "--cuda" not in provisioned.call_args[0][1]

    def test_env_var_variant_needs_no_flag(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_VARIANT", "cpu")
        cli.main(["deploy", "--config", "x"])
        assert provisioned.called


class TestEngineEnvironment:
    def test_llama_server_bin_is_set_before_exec(self, provisioned):
        cli.main(["deploy", "--cpu"])
        env = provisioned.call_args[0][2]
        assert env["MSHIP_LLAMA_SERVER_BIN"] == "/builds/llama-server.sh"

    def test_thin_advertises_no_capacity(self, provisioned):
        cli.main(["deploy", "--thin"])
        env = provisioned.call_args[0][2]
        assert env["MSHIP_NODE_NUM_CPUS"] == "0"
        assert env["MSHIP_NODE_NUM_GPUS"] == "0"

    def test_other_variants_do_not_pin_capacity(self, provisioned):
        cli.main(["deploy", "--cpu"])
        env = provisioned.call_args[0][2]
        assert "MSHIP_NODE_NUM_CPUS" not in env

    def test_explicit_node_sizing_wins_over_thin_defaults(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_NODE_NUM_CPUS", "4")
        cli.main(["deploy", "--thin"])
        assert provisioned.call_args[0][2]["MSHIP_NODE_NUM_CPUS"] == "4"

    def test_cache_dir_defaults_under_mship_home(self, provisioned, tmp_path):
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_CACHE_DIR"] == os.path.join(str(tmp_path), "cache")

    def test_explicit_cache_dir_is_preserved(self, provisioned, monkeypatch):
        monkeypatch.setenv("MSHIP_CACHE_DIR", "/mnt/shared/models")
        cli.main(["deploy", "--cpu"])
        assert provisioned.call_args[0][2]["MSHIP_CACHE_DIR"] == "/mnt/shared/models"

    def test_cuda_checks_for_a_live_device(self, provisioned):
        cli.main(["deploy", "--cuda"])
        cli.llama_cpp.warn_if_no_cuda_device.assert_called_once()

    def test_non_cuda_skips_that_check(self, provisioned):
        cli.main(["deploy", "--cpu"])
        cli.llama_cpp.warn_if_no_cuda_device.assert_not_called()


class TestInfo:
    def test_info_without_a_variant_answers_locally(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        monkeypatch.delenv("MSHIP_VARIANT", raising=False)
        cli.main(["info"])
        out = capsys.readouterr().out
        assert "MSHIP_HOME" in out
        assert "none provisioned" in out

    def test_info_with_a_variant_execs_the_engine(self, provisioned):
        cli.main(["info", "--cpu"])
        assert provisioned.call_args[0][1][-1] == "info"


class TestPaths:
    def test_mship_home_relocates_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        assert paths.env_dir("cpu").startswith(str(tmp_path))
        assert paths.builds_dir("cpu").startswith(str(tmp_path))
        assert paths.bin_dir().startswith(str(tmp_path))

    def test_builds_are_scoped_per_variant(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        assert paths.builds_dir("cpu") != paths.builds_dir("cuda")

    def test_cache_dir_is_not_the_bootstrappers_concern(self, monkeypatch, tmp_path):
        """MSHIP_CACHE_DIR may point at shared storage; MSHIP_HOME must not."""
        monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
        monkeypatch.setenv("MSHIP_CACHE_DIR", "/mnt/shared")
        for name in VARIANTS:
            assert not paths.env_dir(name).startswith("/mnt/shared")
            assert not paths.builds_dir(name).startswith("/mnt/shared")


class TestThinSkipsLlamaServer:
    def test_thin_does_not_provision_a_binary_it_cannot_use(self, provisioned):
        cli.main(["deploy", "--thin"])
        cli.llama_cpp.provision.assert_not_called()
        assert "MSHIP_LLAMA_SERVER_BIN" not in provisioned.call_args[0][2]

    def test_serving_variants_still_provision(self, provisioned):
        cli.main(["deploy", "--cpu"])
        cli.llama_cpp.provision.assert_called_once()
