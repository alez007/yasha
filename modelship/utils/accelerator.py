"""Detects which accelerator this machine can use. Ray-free, torch-optional."""

from __future__ import annotations

import os
import platform
import shutil


def detect_accelerator() -> str:
    """Returns "cuda", "rocm", "xpu", "metal", or "cpu". Keys on the installed
    torch build, not on what hardware happens to be visible."""
    kind = _torch_accelerator()
    return kind if kind is not None else _no_torch_fallback()


def _torch_accelerator() -> str | None:
    """None only when torch itself isn't importable — never a "no accelerator"
    signal, which is "cpu"/"metal" below."""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.version.hip is not None and torch.cuda.device_count() > 0:
            return "rocm"
        if torch.version.cuda is not None and torch.cuda.device_count() > 0:
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    return "cpu"


def _no_torch_fallback() -> str:
    """Last resort when torch is entirely absent (e.g. the `thin` image)."""
    if shutil.which("nvidia-smi") is not None:
        return "cuda"
    if os.path.exists("/dev/kfd"):
        return "rocm"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    return "cpu"
