"""`mship` console-script entry point. Ray/torch-free until it hands off to
modelship.driver — see modelship/utils/cli.py and modelship/utils/ray_auth.py
for the same discipline.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import stat
import sys
from typing import NamedTuple

from modelship.deploy.capabilities import LOADER_MODULES
from modelship.utils import fetch_and_extract_archive
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.cache import resolve_cache_root

_REQUIRED_PYTHON = (3, 12, 10)

# Own-CI builds for every platform; see .github/workflows/llama-cpp-build.yml.
_LLAMA_CPP_TAG = "b10375"
_LLAMA_CPP_BUILDS_REPO = "modelship-ai/llama-cpp-builds"
_LLAMA_CPP_RELEASE_URL = f"https://github.com/{_LLAMA_CPP_BUILDS_REPO}/releases/download/llamacpp-{_LLAMA_CPP_TAG}"

# Line-anchored: llama-cpp-build.yml's pin job rewrites these by sed.
_SHA256_LINUX_X64 = "64625921d1257485a82cc7eee6de58075d5f81a1b588e3e2817cf9632ffc8090"
_SHA256_LINUX_ARM64 = "122186a168c10c9510b6e43c670515206d3a4ca7f5c10ef9fa4708fbea77a9de"
_SHA256_MACOS_ARM64_METAL = "b3f66fc4f82fbaaa70a3d18c37d1e9cbddc65133cf226b1695dc8c2cd20b4545"


class _LlamaCppAsset(NamedTuple):
    """`lib_env` is the loader-path env var the wrapper script exports."""

    url: str
    sha256: str
    lib_env: str


_LLAMA_CPP_ASSETS: dict[tuple[str, str], _LlamaCppAsset] = {
    ("Darwin", "arm64"): _LlamaCppAsset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-macos-arm64-metal.tar.gz",
        sha256=_SHA256_MACOS_ARM64_METAL,
        lib_env="DYLD_LIBRARY_PATH",
    ),
    ("Linux", "x86_64"): _LlamaCppAsset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-x64.tar.gz",
        sha256=_SHA256_LINUX_X64,
        lib_env="LD_LIBRARY_PATH",
    ),
    ("Linux", "aarch64"): _LlamaCppAsset(
        url=f"{_LLAMA_CPP_RELEASE_URL}/llama-server-{_LLAMA_CPP_TAG}-linux-arm64.tar.gz",
        sha256=_SHA256_LINUX_ARM64,
        lib_env="LD_LIBRARY_PATH",
    ),
}


class LlamaServerProvisionError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("deploy", "info"):
        print("usage: mship {deploy,info} [args]", file=sys.stderr)
        sys.exit(2)

    command, rest = argv[0], argv[1:]
    if command == "info":
        _cmd_info()
    else:
        _cmd_deploy(rest)


def _cmd_deploy(argv: list[str]) -> None:
    from modelship.utils.cli import apply_args_to_env, parse_args

    args = parse_args(argv)
    apply_args_to_env(args)
    os.environ.setdefault("MSHIP_CACHE_DIR", resolve_cache_root())
    _guard_python_version()

    if _has_llama_cpp_asset():
        _provision_llama_server()

    if _is_own_head_deploy():
        _check_loader_capabilities(args.config)

    from modelship.driver import main as driver_main

    driver_main(argv)


def _cmd_info() -> None:
    accelerator = detect_accelerator()
    print(f"accelerator: {accelerator}")
    print(f"python: {platform.python_version()}")
    print(f"cache: {resolve_cache_root()}")
    try:
        import ray

        print(f"ray: {ray.__version__}")
    except Exception:
        print("ray: not installed")

    if _has_llama_cpp_asset():
        path = _provision_llama_server()
        print(f"llama-server: {path or 'not provisioned'}")
    else:
        print(f"llama-server: {os.environ.get('MSHIP_LLAMA_SERVER_BIN') or 'unset'}")


def _guard_python_version() -> None:
    if sys.version_info[:3] != _REQUIRED_PYTHON:
        got = ".".join(map(str, sys.version_info[:3]))
        print(f"mship requires Python {'.'.join(map(str, _REQUIRED_PYTHON))} exactly, found {got}.", file=sys.stderr)
        sys.exit(1)


def _current_llama_cpp_asset() -> _LlamaCppAsset | None:
    return _LLAMA_CPP_ASSETS.get((platform.system(), platform.machine()))


def _has_llama_cpp_asset() -> bool:
    return _current_llama_cpp_asset() is not None


def _provision_llama_server() -> str | None:
    if explicit := os.environ.get("MSHIP_LLAMA_SERVER_BIN"):
        return explicit
    try:
        path = _resolve_llama_server_bin()
        os.environ["MSHIP_LLAMA_SERVER_BIN"] = path
        return path
    except Exception as e:
        print(f"warning: llama-server provisioning failed: {e}", file=sys.stderr)
        return None


def _resolve_llama_server_bin() -> str:
    """Downloads, sha256-verifies, and extracts the pinned llama.cpp build for this
    (system, machine), caching under resolve_cache_root()/llama.cpp/<tag>/. Native-install
    equivalent of the Docker images baking a prebuilt binary at build time."""
    asset = _current_llama_cpp_asset()
    if asset is None:
        raise LlamaServerProvisionError(
            "loader: llama_server needs MSHIP_LLAMA_SERVER_BIN set to a llama-server binary "
            f"(no prebuilt asset for {platform.system()}/{platform.machine()}). "
            "See https://github.com/ggml-org/llama.cpp/releases."
        )

    tag_dir = os.path.join(resolve_cache_root(), "llama.cpp", _LLAMA_CPP_TAG)
    archive_path = os.path.join(tag_dir, "archive.tar.gz")
    extract_dir = os.path.join(tag_dir, "extracted")
    wrapper_path = os.path.join(tag_dir, "llama-server.sh")

    binary = os.path.join(extract_dir, "llama-server")
    if not os.path.isfile(binary):
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
        fetch_and_extract_archive(
            asset.url,
            asset.sha256,
            archive_path,
            extract_dir,
            flatten=True,
            keep_archive=True,
        )
        os.chmod(binary, os.stat(binary).st_mode | stat.S_IEXEC)

    _write_wrapper(wrapper_path, extract_dir, asset.lib_env)
    return wrapper_path


def _write_wrapper(wrapper_path: str, extract_dir: str, lib_env: str) -> None:
    content = (
        "#!/bin/sh\n"
        f'export {lib_env}="{extract_dir}${{{lib_env}:+:${lib_env}}}"\n'
        f'exec "{extract_dir}/llama-server" "$@"\n'
    )
    with open(wrapper_path, "w") as f:
        f.write(content)
    os.chmod(wrapper_path, os.stat(wrapper_path).st_mode | stat.S_IEXEC)


def _is_own_head_deploy() -> bool:
    """True only when this driver IS the node the model will run on (no --address
    join, no --use-existing-ray-cluster) — the only topology where this local
    module-presence check is meaningful."""
    if os.environ.get("MSHIP_ADDRESS"):
        return False
    return os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() != "true"


def _check_loader_capabilities(config_path: str | None) -> None:
    if not config_path or not os.path.isfile(config_path):
        return
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    loaders = {str(m.get("loader")) for m in raw.get("models", []) if isinstance(m, dict) and m.get("loader")}
    for loader in loaders:
        module = LOADER_MODULES.get(loader)
        if module and importlib.util.find_spec(module) is None:
            print(
                f"error: models.yaml uses loader: {loader}, but '{module}' isn't installed in this "
                "environment. Use loader: llama_server for GGUF models on this hardware.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
