import os
from unittest.mock import patch

import pytest

from mship_bootstrap import engine, paths
from mship_bootstrap.variants import VARIANTS


def _pins_by_package() -> dict[str, dict[str, str]]:
    pinned: dict[str, dict[str, str]] = {}
    for name in sorted(VARIANTS):
        for line in engine.read_pins(VARIANTS[name]).splitlines():
            if "==" not in line or line.startswith((" ", "#")):
                continue
            pkg, _, rest = line.partition("==")
            version = rest.split()[0].rstrip("\\").strip().split(";")[0].strip()
            pinned.setdefault(pkg.strip(), {})[name] = version
    return pinned


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
    monkeypatch.delenv("MSHIP_ENGINE_WHEEL", raising=False)
    return tmp_path


class TestPins:
    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_every_variant_ships_pins(self, name):
        body = engine.read_pins(VARIANTS[name])
        assert "--hash=sha256:" in body

    def test_pins_do_not_contain_the_engine_itself(self):
        """`uv export --no-emit-project` omits it; _write_requirements appends it."""
        assert "mship-engine==" not in engine.read_pins(VARIANTS["thin"])

    def test_cluster_critical_packages_match_across_variants(self):
        """Ray refuses to form a cluster across mismatched versions."""
        pinned = _pins_by_package()
        for pkg in ("ray", "pydantic", "protobuf", "numpy", "fastapi"):
            versions = pinned[pkg]
            assert len(set(versions.values())) == 1, f"{pkg} differs across variants: {versions}"

    def test_only_accelerator_packages_differ_across_variants(self):
        """uv resolves conflicting extras independently, so variants may diverge."""
        # torch/torchvision/vllm differ by local version (+cpu vs +cu130).
        # setuptools is build-time only.
        expected = {"torch", "torchvision", "vllm", "setuptools"}
        differing = {pkg for pkg, versions in _pins_by_package().items() if len(set(versions.values())) > 1}
        assert differing == expected


class TestWriteRequirements:
    def test_appends_the_engine_requirement(self, home):
        os.makedirs(paths.env_dir("thin"), exist_ok=True)
        target = engine._write_requirements(VARIANTS["thin"], "0.8.0")
        assert open(target).read().splitlines()[-1] == "mship-engine[thin]==0.8.0"

    def test_writes_where_an_operator_will_find_it(self, home):
        os.makedirs(paths.env_dir("cpu"), exist_ok=True)
        assert engine._write_requirements(VARIANTS["cpu"], "0.8.0") == str(home / "envs" / "cpu" / "pins.txt")

    def test_engine_wheel_override(self, home, monkeypatch, tmp_path):
        wheel = tmp_path / "mship_engine-0.8.0-py3-none-any.whl"
        wheel.write_text("")
        monkeypatch.setenv("MSHIP_ENGINE_WHEEL", str(wheel))
        os.makedirs(paths.env_dir("cuda"), exist_ok=True)
        target = engine._write_requirements(VARIANTS["cuda"], "0.8.0")
        assert open(target).read().splitlines()[-1] == f"{wheel}[cuda]"

    def test_missing_engine_wheel_is_an_error(self, home, monkeypatch):
        monkeypatch.setenv("MSHIP_ENGINE_WHEEL", "/does/not/exist.whl")
        os.makedirs(paths.env_dir("cpu"), exist_ok=True)
        with pytest.raises(engine.EngineError, match="not a file"):
            engine._write_requirements(VARIANTS["cpu"], "0.8.0")


class TestProvision:
    def test_creates_the_venv_then_syncs(self, home):
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["thin"], "/usr/bin/uv", "0.8.0")
        venv_cmd, sync_cmd = run.call_args_list[0][0][0], run.call_args_list[1][0][0]
        assert venv_cmd[:4] == ["/usr/bin/uv", "venv", "--python", engine.PYTHON_VERSION]
        assert sync_cmd[:3] == ["/usr/bin/uv", "pip", "sync"]

    def test_cpu_passes_its_indexes_to_sync(self, home):
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["cpu"], "/usr/bin/uv", "0.8.0")
        sync_cmd = run.call_args_list[-1][0][0]
        assert "https://download.pytorch.org/whl/cpu" in sync_cmd
        assert "unsafe-best-match" in sync_cmd

    def test_existing_venv_is_not_recreated(self, home):
        python = paths.venv_python("cpu")
        os.makedirs(os.path.dirname(python), exist_ok=True)
        open(python, "w").close()
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["cpu"], "/usr/bin/uv", "0.8.0")
        assert all(call[0][0][1] != "venv" for call in run.call_args_list)

    def test_missing_venv_is_recreated(self, home):
        """Self-heal: `rm -rf` on the env directory recovers on the next run."""
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["cpu"], "/usr/bin/uv", "0.8.0")
        assert any(call[0][0][1] == "venv" for call in run.call_args_list)


class TestDescribeEnvs:
    def test_empty_when_nothing_provisioned(self, home):
        assert engine.describe_envs() == []

    def test_lists_only_complete_envs(self, home):
        for name in ("cpu", "cuda"):
            os.makedirs(paths.env_dir(name), exist_ok=True)
        python = paths.venv_python("cpu")
        os.makedirs(os.path.dirname(python), exist_ok=True)
        open(python, "w").close()
        assert [name for name, _ in engine.describe_envs()] == ["cpu"]
