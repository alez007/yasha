"""Checks that run before anything is downloaded."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


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
    # Provisioning for hardware the builder doesn't have: the images for a GPU
    # node are built on CPU-only runners. Deploy still gates on the real thing.
    if (os.environ.get("MSHIP_SKIP_HARDWARE_CHECK") or "").strip().lower() in ("1", "true"):
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
