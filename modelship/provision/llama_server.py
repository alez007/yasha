"""Resolves the llama-server binary for the llama_server loader on macOS.

Order: MSHIP_LLAMA_SERVER_BIN -> cached extract -> download+verify+extract.
Linux stops after step 1 (see CLAUDE.md) — the images already set
MSHIP_LLAMA_SERVER_BIN themselves.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import tarfile
import urllib.request

from modelship._pins import LLAMA_CPP_METAL_ASSET_URL, LLAMA_CPP_METAL_SHA256, LLAMA_CPP_TAG
from modelship.utils.cache import resolve_cache_root


class LlamaServerProvisionError(RuntimeError):
    pass


def resolve_llama_server_bin() -> str:
    if explicit := os.environ.get("MSHIP_LLAMA_SERVER_BIN"):
        return explicit

    if platform.system() != "Darwin":
        raise LlamaServerProvisionError(
            "loader: llama_server needs MSHIP_LLAMA_SERVER_BIN set to a llama-server binary "
            "(auto-provisioning only runs on macOS). See https://github.com/ggml-org/llama.cpp/releases."
        )

    tag_dir = os.path.join(resolve_cache_root(), "llama.cpp", LLAMA_CPP_TAG)
    archive_path = os.path.join(tag_dir, "archive.tar.gz")
    extract_dir = os.path.join(tag_dir, "extracted")
    wrapper_path = os.path.join(tag_dir, "llama-server.sh")

    os.makedirs(tag_dir, exist_ok=True)
    if not os.path.isfile(archive_path):
        _download(LLAMA_CPP_METAL_ASSET_URL, archive_path)
    _verify_sha256(archive_path, LLAMA_CPP_METAL_SHA256)

    if not os.path.isfile(os.path.join(extract_dir, "llama-server")):
        _extract(archive_path, extract_dir)

    _write_wrapper(wrapper_path, extract_dir)
    return wrapper_path


def _download(url: str, dest: str) -> None:
    tmp = f"{dest}.tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _verify_sha256(path: str, expected: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise LlamaServerProvisionError(
            f"llama-server archive at {path} failed sha256 verification: expected {expected}, got {actual}"
        )


def _extract(archive_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if not name:
                continue
            member.name = name
            tar.extract(member, path=extract_dir, filter="data")
    binary = os.path.join(extract_dir, "llama-server")
    if os.path.isfile(binary):
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)


def _write_wrapper(wrapper_path: str, extract_dir: str) -> None:
    content = (
        "#!/bin/sh\n"
        f'export DYLD_LIBRARY_PATH="{extract_dir}${{DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}}"\n'
        f'exec "{extract_dir}/llama-server" "$@"\n'
    )
    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)
