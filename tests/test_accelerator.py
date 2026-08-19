from unittest.mock import MagicMock, patch

from modelship.utils.accelerator import detect_accelerator


def _fake_torch(*, cuda_version=None, hip_version=None, device_count=0, has_xpu=False, xpu_available=False):
    """A MagicMock standing in for `torch`, shaped for `detect_accelerator`'s probes.
    `has_xpu=False` deletes the auto-vivified `.xpu` attribute so `hasattr(torch, "xpu")` is False."""
    mock_torch = MagicMock()
    mock_torch.version.cuda = cuda_version
    mock_torch.version.hip = hip_version
    mock_torch.cuda.device_count.return_value = device_count
    if has_xpu:
        mock_torch.xpu.is_available.return_value = xpu_available
    else:
        del mock_torch.xpu
    return mock_torch


class TestDetectAcceleratorTorchBuild:
    """Keys on the installed torch build, not on what hardware happens to be visible."""

    def test_cuda_build_with_device_returns_cuda(self):
        mock_torch = _fake_torch(cuda_version="12.4", device_count=1)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert detect_accelerator() == "cuda"

    def test_rocm_build_with_device_returns_rocm(self):
        mock_torch = _fake_torch(hip_version="5.7", device_count=1)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert detect_accelerator() == "rocm"

    def test_xpu_build_available_returns_xpu(self):
        mock_torch = _fake_torch(has_xpu=True, xpu_available=True)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert detect_accelerator() == "xpu"

    def test_cpu_build_no_device_returns_cpu(self):
        mock_torch = _fake_torch()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("platform.system", return_value="Linux"),
        ):
            assert detect_accelerator() == "cpu"

    def test_cpu_image_with_nvidia_smi_on_path_must_return_cpu(self):
        """torch's own build wins over nvidia-smi merely being on PATH."""
        mock_torch = _fake_torch()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("platform.system", return_value="Linux"),
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        ):
            assert detect_accelerator() == "cpu"

    def test_cuda_build_but_no_gpus_visible_returns_cpu(self):
        mock_torch = _fake_torch(cuda_version="12.4", device_count=0)
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("platform.system", return_value="Linux"),
        ):
            assert detect_accelerator() == "cpu"

    def test_darwin_arm64_with_torch_installed_returns_metal(self):
        mock_torch = _fake_torch()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert detect_accelerator() == "metal"

    def test_darwin_intel_with_torch_installed_returns_cpu(self):
        mock_torch = _fake_torch()
        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert detect_accelerator() == "cpu"


class TestDetectAcceleratorNoTorch:
    """torch entirely absent (the `thin` coordinator image) — nvidia-smi/`/dev/kfd`
    are the only signals left, and only as a last resort."""

    def test_nvidia_smi_on_path_returns_cuda(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        ):
            assert detect_accelerator() == "cuda"

    def test_dev_kfd_present_returns_rocm(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=True),
        ):
            assert detect_accelerator() == "rocm"

    def test_darwin_arm64_returns_metal(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert detect_accelerator() == "metal"

    def test_darwin_intel_returns_cpu(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert detect_accelerator() == "cpu"

    def test_linux_no_signals_returns_cpu(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
            patch("platform.system", return_value="Linux"),
        ):
            assert detect_accelerator() == "cpu"
