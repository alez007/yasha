"""Variant definitions and resolution. Chosen at runtime, not at install time;
no default and no auto-detection."""

from __future__ import annotations

import os
from typing import NamedTuple

_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# Embeds the vLLM version; bump alongside the `vllm` pin in the engine's pyproject.
_VLLM_CPU_INDEX = "https://wheels.vllm.ai/0.26.0/cpu"


class Variant(NamedTuple):
    name: str
    extras: tuple[str, ...]
    index_args: tuple[str, ...]
    # None means runs anywhere.
    requires_accelerator: str | None
    serves_models: bool
    summary: str


VARIANTS: dict[str, Variant] = {
    "cuda": Variant(
        name="cuda",
        extras=("cuda",),
        index_args=(),
        requires_accelerator="cuda",
        serves_models=True,
        summary="NVIDIA GPU node (vLLM, Diffusers, llama.cpp GPU offload)",
    ),
    "cpu": Variant(
        name="cpu",
        extras=("cpu", "vllm-cpu"),
        index_args=(
            "--index",
            _PYTORCH_CPU_INDEX,
            "--index",
            _VLLM_CPU_INDEX,
            "--index-strategy",
            "unsafe-best-match",
        ),
        requires_accelerator=None,
        serves_models=True,
        summary="CPU node (vLLM CPU, llama.cpp, whisper.cpp, sherpa-onnx, SD)",
    ),
    "metal": Variant(
        name="metal",
        extras=("metal",),
        index_args=(),
        requires_accelerator="metal",
        serves_models=True,
        summary="Apple Silicon (Metal offload)",
    ),
    "thin": Variant(
        name="thin",
        extras=("thin",),
        index_args=(),
        serves_models=False,
        requires_accelerator=None,
        summary="coordinator/head only — joins capacity from other nodes",
    ),
}

VARIANT_ORDER = ("cuda", "cpu", "metal", "thin")

NO_VARIANT_ERROR = (
    "error: no variant selected\n\n"
    + "".join(f"  mship deploy --{n:<8} {VARIANTS[n].summary}\n" for n in VARIANT_ORDER)
    + ("\nFirst use of a variant provisions a pinned Python 3.12.10 environment for it\n(several GB for --cuda).\n")
)


class VariantError(Exception):
    pass


def split_variant_flag(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pull the variant flag out of argv, leaving everything else for the engine."""
    found: list[str] = []
    rest: list[str] = []
    for arg in argv:
        name = arg[2:] if arg.startswith("--") else None
        if name in VARIANTS:
            found.append(name)
        else:
            rest.append(arg)
    if len(found) > 1:
        raise VariantError(f"error: pick one variant, got {', '.join('--' + f for f in sorted(set(found)))}")
    return (found[0] if found else None), rest


def resolve(flag: str | None, env: dict[str, str] | None = None) -> Variant:
    """Flag wins over MSHIP_VARIANT; disagreement is an error."""
    env = os.environ if env is None else env
    from_env = (env.get("MSHIP_VARIANT") or "").strip() or None

    if from_env is not None and from_env not in VARIANTS:
        raise VariantError(
            f"error: MSHIP_VARIANT={from_env!r} is not a variant; expected one of {', '.join(VARIANT_ORDER)}"
        )
    if flag is not None and from_env is not None and flag != from_env:
        raise VariantError(f"error: --{flag} conflicts with MSHIP_VARIANT={from_env}")

    chosen = flag or from_env
    if chosen is None:
        raise VariantError(NO_VARIANT_ERROR)
    return VARIANTS[chosen]


def engine_requirement(variant: Variant, version: str) -> str:
    return f"mship-engine[{','.join(variant.extras)}]=={version}"
