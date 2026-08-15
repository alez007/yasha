"""Entry point. Resolves a variant, provisions its environment, execs the engine."""

from __future__ import annotations

import os
import sys

from . import __version__, engine, gates, llama_cpp, paths, uv_binary
from .variants import VariantError, split_variant_flag
from .variants import resolve as resolve_variant

_COMMANDS = ("deploy", "info")

_USAGE = """usage: mship {deploy,info} [--cuda|--cpu|--metal|--thin] [args]

  deploy   provision the node and serve models (requires a variant)
  info     report bootstrapper state, or the engine's own report with a variant
"""


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in _COMMANDS:
        sys.stderr.write(_USAGE)
        sys.exit(2)

    command, rest = argv[0], argv[1:]
    try:
        flag, rest = split_variant_flag(rest)
    except VariantError as e:
        sys.exit(str(e))

    # Answerable before any variant exists.
    if command == "info" and flag is None and not os.environ.get("MSHIP_VARIANT"):
        engine.print_info(__version__)
        return

    try:
        variant = resolve_variant(flag)
        gates.check_platform()
        gates.check_hardware(variant.requires_accelerator)
        uv = uv_binary.ensure_uv()
        python = engine.provision(variant, uv, __version__)
    except (VariantError, gates.GateError, uv_binary.UvError, engine.EngineError) as e:
        sys.exit(str(e))

    env = _engine_env(variant)
    os.execve(python, [python, "-m", "modelship.launcher", command, *rest], env)


def _engine_env(variant) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MSHIP_CACHE_DIR", os.path.join(paths.home(), "cache"))

    if variant.serves_models and (wrapper := llama_cpp.provision(variant)):
        env["MSHIP_LLAMA_SERVER_BIN"] = wrapper
        if variant.name == "cuda":
            llama_cpp.warn_if_no_cuda_device(wrapper)

    if variant.name == "thin":
        # Same contract as the thin image.
        env.setdefault("MSHIP_NODE_NUM_CPUS", "0")
        env.setdefault("MSHIP_NODE_NUM_GPUS", "0")

    return env
