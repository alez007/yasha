"""Provisioning the engine environment.

One environment per variant, no version in the path. `uv pip sync` is declarative
— it removes packages the pins no longer list — so an upgrade converges in place
and a missing or partial environment repairs itself on the next run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources

from . import paths
from .variants import Variant, engine_requirement

PYTHON_VERSION = "3.12.10"


class EngineError(RuntimeError):
    pass


def read_pins(variant: Variant) -> str:
    try:
        return resources.files(f"{__package__}.pins").joinpath(f"{variant.name}.txt").read_text()
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise EngineError(f"error: no pins file shipped for variant {variant.name!r}") from e


def provision(variant: Variant, uv: str, version: str) -> str:
    """Returns the interpreter to exec. Safe to call on every invocation."""
    env_dir = paths.env_dir(variant.name)
    venv = paths.venv_dir(variant.name)
    os.makedirs(env_dir, exist_ok=True)

    requirements = _write_requirements(variant, version)

    if not os.path.isfile(paths.venv_python(variant.name)):
        _run([uv, "venv", "--python", PYTHON_VERSION, venv], "create the engine environment")

    _run(
        [uv, "pip", "sync", "--python", venv, *variant.index_args, requirements],
        f"install the {variant.name} engine environment",
    )
    return paths.venv_python(variant.name)


def _write_requirements(variant: Variant, version: str) -> str:
    """The shipped pins plus the engine itself, which `uv export --no-emit-project`
    omits. Written where an operator debugging the node will find it."""
    body = read_pins(variant)
    if wheel := os.environ.get("MSHIP_ENGINE_WHEEL"):
        if not os.path.isfile(wheel):
            raise EngineError(f"error: MSHIP_ENGINE_WHEEL={wheel!r} is not a file")
        engine = f"{os.path.abspath(wheel)}[{','.join(variant.extras)}]"
    else:
        engine = engine_requirement(variant, version)

    target = paths.pins_copy(variant.name)
    with open(target, "w") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
        f.write(f"{engine}\n")
    return target


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise EngineError(f"error: {cmd[0]} not found") from e
    except subprocess.CalledProcessError as e:
        raise EngineError(f"error: failed to {what} (uv exited {e.returncode})") from e


def describe_envs() -> list[tuple[str, str]]:
    """(variant, human-readable size) for every provisioned environment."""
    root = os.path.join(paths.home(), "envs")
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if os.path.isfile(paths.venv_python(name)):
            out.append((name, _human_size(_tree_size(paths.env_dir(name)))))
    return out


def _tree_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def print_info(version: str) -> None:
    print(f"mship (bootstrapper): {version}")
    print(f"python:               {sys.version.split()[0]} ({sys.executable})")
    print(f"MSHIP_HOME:           {paths.home()}")
    print(f"uv:                   {shutil.which('uv') or os.path.join(paths.bin_dir(), 'uv')}")
    envs = describe_envs()
    if not envs:
        print("environments:         none provisioned")
        return
    print("environments:")
    for name, size in envs:
        print(f"  {name:<6} {size:>8}  {paths.env_dir(name)}")
