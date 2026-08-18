"""Checks that run before anything is downloaded."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess

CUDA_TOOLKIT_DOCS = "https://docs.model-ship.ai/install-native/#platform-prerequisites"


class GateError(Exception):
    pass


def check_platform() -> None:
    if platform.system() == "Windows":
        raise GateError("error: Windows is not supported.\nRun the Docker images, or use WSL2 with a Linux install.")
    if _is_musl():
        raise GateError(
            "error: musl-based distros (Alpine) are not supported.\n"
            "ray publishes no musllinux wheels, so the engine cannot be installed.\n"
            "Use a glibc distro, or run the Docker images, which are Debian-based."
        )


def _is_musl() -> bool:
    if platform.system() != "Linux":
        return False
    if "musl" in (platform.libc_ver()[0] or "").lower():
        return True
    # libc_ver() reports the Python binary's libc, not the host's.
    return bool(_ld_musl_present())


def _ld_musl_present() -> bool:
    for directory in ("/lib", "/usr/lib"):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        if any(e.startswith("ld-musl-") for e in entries):
            return True
    return False


def detect_accelerator() -> str:
    """Hardware-visible accelerator: "cuda", "rocm", "metal" or "cpu".

    Mirrors `modelship.utils.accelerator._no_torch_fallback`, not
    `detect_accelerator`, which keys on a torch build that doesn't exist yet.
    """
    if _nvidia_devices_present():
        return "cuda"
    if os.path.exists("/dev/kfd"):
        return "rocm"
    if (platform.system(), platform.machine()) == ("Darwin", "arm64"):
        return "metal"
    return "cpu"


def _nvidia_devices_present() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and any(line.startswith("GPU ") for line in result.stdout.splitlines())


def check_hardware(required: str | None) -> None:
    if required is None:
        return
    found = detect_accelerator()
    if found == required:
        return
    if required == "cuda":
        raise GateError(
            "error: --cuda needs an NVIDIA GPU, and none was found.\n"
            "`nvidia-smi -L` must list at least one device. Install the NVIDIA driver first,\n"
            "or use --cpu for a CPU-only node."
        )
    if required == "metal":
        raise GateError(
            f"error: --metal needs Apple Silicon, found {platform.system()}/{platform.machine()}.\n"
            "Use --cuda for an NVIDIA host or --cpu for a CPU-only node."
        )
    raise GateError(f"error: --{required} is unavailable on this machine (detected {found}).")


def find_nvcc() -> str | None:
    """CUDA_HOME/CUDA_PATH, then PATH, then /usr/local/cuda — torch's own order in
    `cpp_extension._find_cuda_home`. The CUDA apt packages leave nvcc off PATH."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = (os.environ.get(var) or "").strip()
        if root and os.path.exists(candidate := os.path.join(root, "bin", "nvcc")):
            return candidate
    if found := shutil.which("nvcc"):
        return found
    return "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None


def cuda_toolkit_gaps() -> list[str]:
    """What flashinfer's JIT needs beyond the driver."""
    gaps = []
    if find_nvcc() is None:
        gaps.append("nvcc (CUDA toolkit)")
    if shutil.which("ninja") is None:
        gaps.append("ninja")
    return gaps


def check_toolchain(variant_name: str) -> None:
    if variant_name != "cuda" or not (gaps := cuda_toolkit_gaps()):
        return
    raise GateError(
        f"error: --cuda is missing {' and '.join(gaps)}.\n"
        "vLLM and Diffusers JIT-compile kernels at model load and fail without them.\n"
        f"See {CUDA_TOOLKIT_DOCS}"
    )
