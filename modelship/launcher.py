"""Engine entry point: `python -m modelship.launcher`, no console script.

Ray-free until it hands off to modelship.driver.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import TYPE_CHECKING

import yaml

from modelship.deploy.capabilities import LOADER_MODULES
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.cache import resolve_cache_root

if TYPE_CHECKING:
    from modelship.utils.config_schema import ModelshipConfig

_REQUIRED_PYTHON = (3, 12, 10)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("deploy", "info"):
        print("usage: python -m modelship.launcher {deploy,info} [args]", file=sys.stderr)
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

    config = _validate_config(args.config)
    if config is not None and _is_own_head_deploy():
        _check_loader_capabilities({m.loader.value for m in config.models})

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

    print(f"llama-server: {os.environ.get('MSHIP_LLAMA_SERVER_BIN') or 'unset'}")


def _guard_python_version() -> None:
    if sys.version_info[:3] != _REQUIRED_PYTHON:
        got = ".".join(map(str, sys.version_info[:3]))
        print(f"mship requires Python {'.'.join(map(str, _REQUIRED_PYTHON))} exactly, found {got}.", file=sys.stderr)
        sys.exit(1)


def _is_own_head_deploy() -> bool:
    """True only when this driver IS the node the model will run on (no --address
    join, no --use-existing-ray-cluster, some capacity of its own) — the only
    topology where this local module-presence check is meaningful."""
    if os.environ.get("MSHIP_ADDRESS"):
        return False
    if os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true":
        return False
    return not _advertises_no_capacity()


def _advertises_no_capacity() -> bool:
    """A thin coordinator reserves 0 CPUs and 0 GPUs, so every model in its config is
    bound for a node that joins later."""
    reserved = []
    for var in ("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"):
        try:
            reserved.append(float(os.environ[var]))
        except (KeyError, ValueError):
            return False
    return not any(reserved)


def _validate_config(config_path: str | None) -> ModelshipConfig | None:
    """Validate models.yaml before the driver imports ray. None when there is
    no config to load."""
    from pydantic import ValidationError

    from modelship.deploy.config import config_absent, load_yaml_config

    if config_absent(config_path):
        return None
    try:
        return load_yaml_config(config_path)
    except (FileNotFoundError, ValidationError, yaml.YAMLError, ValueError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        sys.exit(1)


def _check_loader_capabilities(loaders: set[str]) -> None:
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
