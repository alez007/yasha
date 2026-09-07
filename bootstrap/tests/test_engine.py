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


def _local_version_pins(variant) -> list[str]:
    return [
        line
        for line in engine.read_pins(variant).splitlines()
        if "==" in line and not line.startswith((" ", "#")) and "+" in line.split("==")[1]
    ]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path))
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
        # torch/torchvision/torchaudio/vllm differ by local version (+cpu vs +cu130).
        # setuptools is build-time only.
        expected = {"torch", "torchvision", "torchaudio", "vllm", "setuptools"}
        differing = {pkg for pkg, versions in _pins_by_package().items() if len(set(versions.values())) > 1}
        assert differing == expected


class TestWriteRequirements:
    def test_appends_the_engine_requirement(self, home):
        os.makedirs(paths.env_dir("thin"), exist_ok=True)
        target = engine._write_requirements(VARIANTS["thin"], "0.8.0")
        assert open(target).read().splitlines()[-1] == "mship-engine[thin]==0.8.0"

    def test_stages_rather_than_overwriting_the_success_stamp(self, home):
        os.makedirs(paths.env_dir("cpu"), exist_ok=True)
        assert engine._write_requirements(VARIANTS["cpu"], "0.8.0") == paths.pins_staging("cpu")
        assert not os.path.exists(paths.pins_copy("cpu"))


class TestProvision:
    def test_creates_the_venv_then_syncs(self, home):
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["thin"], "/usr/bin/uv", "0.8.0")
        venv_cmd, sync_cmd = run.call_args_list[0][0][0], run.call_args_list[1][0][0]
        assert venv_cmd[:4] == ["/usr/bin/uv", "venv", "--python", engine.PYTHON_VERSION]
        assert sync_cmd[:3] == ["/usr/bin/uv", "pip", "sync"]

    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_local_version_pins_have_an_index_to_come_from(self, name, home):
        """A `+cu130`/`+cpu` pin is absent from PyPI, so its index must be passed."""
        variant = VARIANTS[name]
        tags = {line.split("+")[1].split()[0].rstrip("\\") for line in _local_version_pins(variant)}
        with patch.object(engine, "_run") as run:
            engine.provision(variant, "/usr/bin/uv", "0.8.0")
        sync_cmd = run.call_args_list[-1][0][0]
        for tag in tags:
            assert any(arg.rstrip("/").endswith(f"/{tag}") for arg in sync_cmd), f"no index serves {tag} for {name}"
        if tags:
            assert "unsafe-first-match" in sync_cmd

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

    def test_syncs_the_staged_file_then_promotes_it(self, home):
        with patch.object(engine, "_run") as run:
            engine.provision(VARIANTS["thin"], "/usr/bin/uv", "0.8.0")
        assert run.call_args_list[-1][0][0][-1] == paths.pins_staging("thin")
        assert open(paths.pins_copy("thin")).read().splitlines()[-1] == "mship-engine[thin]==0.8.0"
        assert not os.path.exists(paths.pins_staging("thin"))

    def test_a_failed_sync_leaves_no_success_stamp(self, home):
        with patch.object(engine, "_run", side_effect=[None, engine.EngineError("boom")]):
            with pytest.raises(engine.EngineError):
                engine.provision(VARIANTS["thin"], "/usr/bin/uv", "0.8.0")
        assert not os.path.exists(paths.pins_copy("thin"))


def _provision_env(name: str, version: str | None = "0.8.0") -> None:
    python = paths.venv_python(name)
    os.makedirs(os.path.dirname(python), exist_ok=True)
    open(python, "w").close()
    if version is not None:
        with open(paths.pins_copy(name), "w") as f:
            f.write(f"idna==3.10\n{engine.engine_requirement(VARIANTS[name], version)}\n")


class TestIsCurrent:
    def test_matching_version(self, home):
        _provision_env("cpu")
        assert engine.is_current(VARIANTS["cpu"], "0.8.0")

    def test_version_skew(self, home):
        _provision_env("cpu", "0.7.12")
        assert not engine.is_current(VARIANTS["cpu"], "0.8.0")

    def test_missing_pins(self, home):
        _provision_env("cpu", version=None)
        assert not engine.is_current(VARIANTS["cpu"], "0.8.0")

    def test_empty_pins(self, home):
        _provision_env("cpu", version=None)
        open(paths.pins_copy("cpu"), "w").close()
        assert not engine.is_current(VARIANTS["cpu"], "0.8.0")

    def test_missing_venv(self, home):
        assert not engine.is_current(VARIANTS["cpu"], "0.8.0")


class TestDescribeStaleness:
    def test_not_bootstrapped_names_the_command(self, home):
        message = engine.describe_staleness(VARIANTS["cuda"], "0.8.0")
        assert "has not been bootstrapped" in message
        assert "mship bootstrap --cuda" in message

    def test_not_bootstrapped_points_at_what_is(self, home):
        _provision_env("cpu")
        assert "Provisioned: cpu" in engine.describe_staleness(VARIANTS["cuda"], "0.8.0")

    def test_skew_names_both_versions(self, home):
        _provision_env("cpu", "0.7.12")
        message = engine.describe_staleness(VARIANTS["cpu"], "0.8.0")
        assert "built for mship 0.7.12" in message
        assert "but this is 0.8.0" in message


class TestProvisionedVariants:
    def test_empty_when_nothing_provisioned(self, home):
        assert engine.provisioned_variants() == []

    def test_lists_only_complete_envs(self, home):
        os.makedirs(paths.env_dir("cuda"), exist_ok=True)
        _provision_env("cpu")
        assert engine.provisioned_variants() == ["cpu"]

    def test_ignores_directories_that_are_not_variants(self, home):
        _provision_env("cpu")
        stray = paths.venv_python("scratch")
        os.makedirs(os.path.dirname(stray), exist_ok=True)
        open(stray, "w").close()
        assert engine.provisioned_variants() == ["cpu"]


class TestDescribeEnvs:
    def test_empty_when_nothing_provisioned(self, home):
        assert engine.describe_envs() == []

    def test_lists_only_complete_envs(self, home):
        os.makedirs(paths.env_dir("cuda"), exist_ok=True)
        _provision_env("cpu")
        assert [name for name, _ in engine.describe_envs()] == ["cpu"]
