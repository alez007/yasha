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
