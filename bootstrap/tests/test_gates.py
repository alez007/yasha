from unittest.mock import patch

import pytest

from mship_bootstrap import gates


class TestCheckPlatform:
    def test_windows_is_refused(self):
        with patch("platform.system", return_value="Windows"), pytest.raises(gates.GateError, match="Windows"):
            gates.check_platform()

    def test_musl_is_refused_by_libc_ver(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("musl", "1.2.5")),
            pytest.raises(gates.GateError, match="musllinux"),
        ):
            gates.check_platform()

    def test_musl_is_refused_by_loader_probe(self):
        """A glibc-built uv-managed CPython reports glibc even on Alpine."""
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.39")),
            patch("os.listdir", return_value=["ld-musl-x86_64.so.1"]),
            pytest.raises(gates.GateError, match="Alpine"),
        ):
            gates.check_platform()

    def test_glibc_linux_passes(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.libc_ver", return_value=("glibc", "2.39")),
            patch("os.listdir", return_value=["libc.so.6"]),
        ):
            gates.check_platform()

    def test_macos_passes(self):
        with patch("platform.system", return_value="Darwin"):
            gates.check_platform()


class TestDetectAccelerator:
    def test_cuda_when_nvidia_smi_lists_a_gpu(self):
        with patch.object(gates, "_nvidia_devices_present", return_value=True):
            assert gates.detect_accelerator() == "cuda"

    def test_metal_on_apple_silicon(self):
        with (
            patch.object(gates, "_nvidia_devices_present", return_value=False),
            patch("os.path.exists", return_value=False),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert gates.detect_accelerator() == "metal"

    def test_cpu_otherwise(self):
        with (
            patch.object(gates, "_nvidia_devices_present", return_value=False),
            patch("os.path.exists", return_value=False),
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert gates.detect_accelerator() == "cpu"

    def test_nvidia_smi_present_but_listing_no_devices(self):
        """A driverless container can still have the binary on PATH."""
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            assert gates._nvidia_devices_present() is False

    def test_nvidia_smi_absent(self):
        with patch("shutil.which", return_value=None):
            assert gates._nvidia_devices_present() is False


class TestCheckHardware:
    def test_none_required_always_passes(self):
        gates.check_hardware(None)

    def test_cuda_without_a_gpu_is_refused(self):
        with (
            patch.object(gates, "detect_accelerator", return_value="cpu"),
            pytest.raises(gates.GateError, match="nvidia-smi"),
        ):
            gates.check_hardware("cuda")

    def test_metal_on_linux_is_refused(self):
        with (
            patch.object(gates, "detect_accelerator", return_value="cpu"),
            pytest.raises(gates.GateError, match="Apple Silicon"),
        ):
            gates.check_hardware("metal")

    def test_match_passes(self):
        with patch.object(gates, "detect_accelerator", return_value="cuda"):
            gates.check_hardware("cuda")

    def test_the_toolchain_is_a_separate_check(self):
        with (
            patch.object(gates, "detect_accelerator", return_value="cuda"),
            patch.object(gates, "cuda_toolkit_gaps", return_value=["ninja"]),
        ):
            gates.check_hardware("cuda")


class TestCheckToolchain:
    def test_cuda_without_nvcc_is_refused(self):
        with (
            patch.object(gates, "cuda_toolkit_gaps", return_value=["nvcc (CUDA toolkit)"]),
            pytest.raises(gates.GateError, match="nvcc"),
        ):
            gates.check_toolchain("cuda")

    def test_both_missing_are_named(self):
        with (
            patch.object(gates, "cuda_toolkit_gaps", return_value=["nvcc (CUDA toolkit)", "ninja"]),
            pytest.raises(gates.GateError, match="nvcc \\(CUDA toolkit\\) and ninja"),
        ):
            gates.check_toolchain("cuda")

    def test_cuda_with_the_toolkit_passes(self):
        with patch.object(gates, "cuda_toolkit_gaps", return_value=[]):
            gates.check_toolchain("cuda")

    def test_other_variants_are_unchecked(self):
        with patch.object(gates, "cuda_toolkit_gaps", return_value=["ninja"]):
            for name in ("cpu", "metal", "thin"):
                gates.check_toolchain(name)


class TestFindNvcc:
    def test_env_var_prefix_wins(self, tmp_path):
        nvcc = tmp_path / "bin" / "nvcc"
        nvcc.parent.mkdir()
        nvcc.touch()
        with patch.dict("os.environ", {"CUDA_HOME": str(tmp_path)}):
            assert gates.find_nvcc() == str(nvcc)

    def test_env_var_pointing_nowhere_falls_through_to_path(self):
        with (
            patch.dict("os.environ", {"CUDA_HOME": "/nonexistent"}),
            patch("shutil.which", return_value="/usr/bin/nvcc"),
        ):
            assert gates.find_nvcc() == "/usr/bin/nvcc"

    def test_conventional_symlink_when_off_path(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", lambda p: p == "/usr/local/cuda/bin/nvcc"),
        ):
            assert gates.find_nvcc() == "/usr/local/cuda/bin/nvcc"

    def test_none_when_absent_everywhere(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
        ):
            assert gates.find_nvcc() is None


class TestCudaToolkitGaps:
    def test_no_gaps_when_both_present(self):
        with (
            patch.object(gates, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
            patch("shutil.which", return_value="/usr/bin/ninja"),
        ):
            assert gates.cuda_toolkit_gaps() == []

    def test_reports_each_missing_piece(self):
        with (
            patch.object(gates, "find_nvcc", return_value=None),
            patch("shutil.which", return_value=None),
        ):
            assert gates.cuda_toolkit_gaps() == ["nvcc (CUDA toolkit)", "ninja"]

    def test_ninja_alone(self):
        with (
            patch.object(gates, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"),
            patch("shutil.which", return_value=None),
        ):
            assert gates.cuda_toolkit_gaps() == ["ninja"]
