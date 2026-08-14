"""Filesystem layout owned by the bootstrapper.

Deliberately smaller than the engine's `modelship.utils.cache.resolve_cache_root`:
no `/.cache` container branch and no MSHIP_CACHE_DIR handling, because the Docker
images bypass the bootstrapper entirely. The two rules must not overlap.

MSHIP_HOME must stay node-local. MSHIP_CACHE_DIR (engine-owned) may point at
shared storage — model weights are identical on every node, while venvs and
llama.cpp binaries are platform-, ABI- and variant-specific.
"""

from __future__ import annotations

import os

_DEFAULT_HOME = "~/.modelship"


def home() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("MSHIP_HOME") or _DEFAULT_HOME))


def bin_dir() -> str:
    return os.path.join(home(), "bin")


def env_dir(variant: str) -> str:
    return os.path.join(home(), "envs", variant)


def venv_dir(variant: str) -> str:
    return os.path.join(env_dir(variant), ".venv")


def venv_python(variant: str) -> str:
    return os.path.join(venv_dir(variant), "bin", "python")


def pins_copy(variant: str) -> str:
    """Operator-visible copy of the pins that built this environment."""
    return os.path.join(env_dir(variant), "pins.txt")


def builds_dir(variant: str) -> str:
    """Per-variant: the CUDA addon mutates the extracted tree and the wrapper
    bakes in that variant's venv library path, so variants cannot share one."""
    return os.path.join(home(), "builds", variant)
