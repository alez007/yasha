"""Detects which accelerator this machine can use. Ray/torch-free."""

from __future__ import annotations

import platform
import shutil


def detect_accelerator() -> str:
    """Returns "cuda", "metal", or "cpu"."""
    if _nvidia_gpu_present():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    return "cpu"


def _nvidia_gpu_present() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return shutil.which("nvidia-smi") is not None
