from unittest.mock import MagicMock, patch

from modelship.utils.accelerator import detect_accelerator


class TestDetectAccelerator:
    def test_nvml_gpu_present_returns_cuda(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            assert detect_accelerator() == "cuda"

    def test_nvidia_smi_on_path_returns_cuda(self):
        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        ):
            assert detect_accelerator() == "cuda"

    def test_darwin_arm64_returns_metal(self):
        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("shutil.which", return_value=None),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            assert detect_accelerator() == "metal"

    def test_darwin_intel_returns_cpu(self):
        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("shutil.which", return_value=None),
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert detect_accelerator() == "cpu"

    def test_linux_no_gpu_returns_cpu(self):
        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("shutil.which", return_value=None),
            patch("platform.system", return_value="Linux"),
        ):
            assert detect_accelerator() == "cpu"
