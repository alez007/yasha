"""Locating uv, and fetching a pinned one when the host has none."""

from __future__ import annotations

import os
import platform
import shutil
import stat

from . import paths
from .fetch import FetchError, fetch_and_extract_archive

_UV_VERSION = "0.12.4"
_UV_RELEASE_URL = f"https://github.com/astral-sh/uv/releases/download/{_UV_VERSION}"

# Line-anchored to match the llama.cpp pins; update all four together with the version.
_SHA256_UV_LINUX_X64 = "c8c60f47e6f88d18dbf6f33d7279fb1fbf7ae76631768152cf5578c3d65729b4"
_SHA256_UV_LINUX_ARM64 = "49d881b3403187e1f1789720881e77e4251ad4259d86c4844862657d2a35d13f"
_SHA256_UV_MACOS_ARM64 = "99a913b606194867b43086404412c1afe079547fee72ecfb6af7e7b0dd54b0c6"
_SHA256_UV_MACOS_X64 = "e603f1eb634ca97a2a125539b983891f53235e901511ed10c32c08c86e253ecd"

_UV_TARGETS: dict[tuple[str, str], tuple[str, str]] = {
    ("Linux", "x86_64"): ("x86_64-unknown-linux-gnu", _SHA256_UV_LINUX_X64),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-gnu", _SHA256_UV_LINUX_ARM64),
    ("Darwin", "arm64"): ("aarch64-apple-darwin", _SHA256_UV_MACOS_ARM64),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin", _SHA256_UV_MACOS_X64),
}


class UvError(RuntimeError):
    pass


def ensure_uv() -> str:
    """Prefer the host's uv; otherwise download the pinned build into MSHIP_HOME."""
    if found := shutil.which("uv"):
        return found

    managed = os.path.join(paths.bin_dir(), "uv")
    if os.path.isfile(managed):
        return managed

    key = (platform.system(), platform.machine())
    target = _UV_TARGETS.get(key)
    if target is None:
        raise UvError(
            f"error: no pinned uv build for {key[0]}/{key[1]}.\n"
            "Install uv yourself (https://docs.astral.sh/uv/getting-started/installation/) and re-run."
        )
    slug, sha256 = target

    extract_dir = os.path.join(paths.bin_dir(), f"uv-{_UV_VERSION}")
    print(f"mship: fetching uv {_UV_VERSION}", flush=True)
    try:
        fetch_and_extract_archive(
            f"{_UV_RELEASE_URL}/uv-{slug}.tar.gz",
            sha256,
            os.path.join(paths.bin_dir(), f"uv-{slug}.tar.gz"),
            extract_dir,
            flatten=True,
        )
    except FetchError as e:
        raise UvError(f"error: could not fetch uv: {e}") from e

    extracted = os.path.join(extract_dir, "uv")
    if not os.path.isfile(extracted):
        raise UvError(f"error: uv archive did not contain a uv binary at {extracted}")
    os.chmod(extracted, os.stat(extracted).st_mode | stat.S_IEXEC)
    os.replace(extracted, managed)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return managed
