"""Engine entry point, run as `python -m modelship.launcher` by the `mship`
bootstrapper and by scripts/entrypoint.sh. Ray-free until it hands off to
modelship.driver — see modelship/utils/cli.py and modelship/utils/ray_auth.py
for the same discipline.

llama-server provisioning lives in the bootstrapper: it needs the CUDA lib path
of an environment that doesn't exist yet when the engine is being installed.
MSHIP_LLAMA_SERVER_BIN arrives here already set, by the bootstrapper natively or
by the image in Docker.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

from modelship.deploy.capabilities import LOADER_MODULES
from modelship.utils.accelerator import detect_accelerator
from modelship.utils.cache import resolve_cache_root

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

    print(f"llama-server: {os.environ.get('MSHIP_LLAMA_SERVER_BIN') or 'unset'}")


def _guard_python_version() -> None:
    if sys.version_info[:3] != _REQUIRED_PYTHON:
        got = ".".join(map(str, sys.version_info[:3]))
        print(f"mship requires Python {'.'.join(map(str, _REQUIRED_PYTHON))} exactly, found {got}.", file=sys.stderr)
        sys.exit(1)


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
