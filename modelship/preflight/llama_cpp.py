from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from modelship.infer.infer_config import LlamaServerConfig, ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile, gpu_share_bytes

logger = get_logger("preflight.llama_cpp")

# n_ctx alignment; llama.cpp has no hard requirement, powers of 256 are
# convention.
_NCTX_ALIGNMENT = 256

# Below this n_ctx, decline the recommendation instead of shipping it. Doubles
# as fit-params' own `-fitc` floor, so it never solves below what we'd accept.
_MIN_NCTX = 512

_FIT_TIMEOUT_S = 30

# Per-device MiB left free by `-fitt`. Broadcasts to every device on a whole-GPU
# deploy; a fractional deploy replaces this with its declared share of the GPU.
_FIT_MARGIN_MIB = 1024

# `llama fit-params`' one stdout line: `-c N -ngl M [-ts a,b,...]`.
_FIT_ARGS_RE = re.compile(r"^-c\s+(-?\d+)\s+-ngl\s+(-?\d+)(?:\s+-ts\s+([\d.,]+))?\s*$")


class LlamaServerPreflight:
    """Sizes the `llama_server` loader's launch args via `llama fit-params`,
    which builds the real KV cache and compute buffers without loading weights."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Thread alignment is independent of context/offload sizing — recommend
        # it even when the fit below declines.
        threads_rec = _recommend_threads(config)

        model_path = config._resolved_path
        if not model_path or not os.path.isfile(model_path):
            logger.info("preflight '%s': skipping — resolved path is not a GGUF file: %s", config.name, model_path)
            return threads_rec

        binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
        if not binary or not os.path.isfile(binary):
            logger.info("preflight '%s': skipping — MSHIP_LLAMA_SERVER_BIN not set", config.name)
            return threads_rec

        server_config = config.llama_server_config or LlamaServerConfig()
        fields_set = server_config.model_fields_set
        pinned_ctx = "n_ctx" in fields_set
        pinned_ngl = "n_gpu_layers" in fields_set
        pinned_ts = "tensor_split" in fields_set

        if pinned_ctx and pinned_ngl and pinned_ts:
            logger.info("preflight '%s': n_ctx, n_gpu_layers and tensor_split all pinned — nothing to fit", config.name)
            return threads_rec

        args = [
            binary,
            "fit-params",
            "-m",
            model_path,
            "--parallel",
            str(server_config.parallel),
            "-b",
            str(server_config.n_batch),
            "-ub",
            str(server_config.ubatch_size),
            "-fa",
            server_config.flash_attn,
            "-ctk",
            server_config.cache_type_k,
            "-ctv",
            server_config.cache_type_v,
            "-fitc",
            str(_MIN_NCTX),
            "-fitt",
            str(_fit_margin_mib(config, hw)),
        ]
        if config.num_gpus == 0:
            args += ["-dev", "none"]
        if pinned_ctx:
            args += ["-c", str(server_config.n_ctx * server_config.parallel)]
        if pinned_ngl:
            args += ["-ngl", str(server_config.n_gpu_layers)]
        if pinned_ts and server_config.tensor_split:
            args += ["-ts", ",".join(str(v) for v in server_config.tensor_split)]

        rec = _run_fit(config, args, server_config.parallel)
        return {**threads_rec, **rec}


def _fit_margin_mib(config: ModelshipModelConfig, hw: HardwareProfile) -> int:
    """`-fitt` reads from *free* VRAM; convert a fractional num_gpus' declared
    share of *total* capacity into a margin against that free figure."""
    if not (0 < config.num_gpus < 1) or not hw.gpus:
        return _FIT_MARGIN_MIB
    gpu = hw.gpus[0] if len(hw.gpus) == 1 else min(hw.gpus, key=lambda g: g.available_bytes)
    share_mib = gpu_share_bytes(config, gpu) / 1024**2
    free_mib = gpu.available_bytes / 1024**2
    return max(_FIT_MARGIN_MIB, int(free_mib - share_mib + _FIT_MARGIN_MIB))


def _run_fit(config: ModelshipModelConfig, args: list[str], parallel: int) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_FIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("preflight '%s': fit-params invocation failed: %s", config.name, e)
        return {}

    if result.returncode != 0:
        logger.warning(
            "preflight '%s': fit-params exited %d: %s",
            config.name,
            result.returncode,
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "",
        )
        return {}

    stdout = result.stdout.strip()
    line = stdout.splitlines()[-1] if stdout else ""
    match = _FIT_ARGS_RE.match(line)
    if match is None:
        logger.warning("preflight '%s': could not parse fit-params output: %r", config.name, line)
        return {}

    ctx_total_raw, ngl_raw, ts_raw = match.groups()
    ctx_total = int(ctx_total_raw)
    ngl = int(ngl_raw)

    rec: dict[str, Any] = {}
    if ctx_total == 0:
        # 0 means "model's own maximum, unconstrained"; round-trips as-is since
        # `_launch` sends `n_ctx * parallel` and llama-server resolves 0 itself.
        rec["n_ctx"] = 0
    else:
        per_slot = (ctx_total // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if per_slot < _MIN_NCTX:
            logger.warning(
                "preflight '%s': fit-params context %d across parallel=%d yields n_ctx=%d (< %d); "
                "skipping recommendation",
                config.name,
                ctx_total,
                parallel,
                per_slot,
                _MIN_NCTX,
            )
            return {}
        rec["n_ctx"] = per_slot
    if ngl >= 0:
        rec["n_gpu_layers"] = ngl
    if ts_raw:
        rec["tensor_split"] = [float(v) for v in ts_raw.split(",")]

    logger.info(
        "preflight llama_server '%s': fit-params -> n_ctx=%s n_gpu_layers=%s tensor_split=%s",
        config.name,
        rec.get("n_ctx"),
        rec.get("n_gpu_layers"),
        rec.get("tensor_split"),
    )
    return rec


def _recommend_threads(config: ModelshipModelConfig) -> dict[str, Any]:
    """Aligns llama-server's threads to `config.num_cpus` (>= 1 only; the 0.1
    default isn't a real budget). Declines rather than undercut `parallel`."""
    if config.num_cpus < 1:
        return {}
    threads = int(config.num_cpus)
    parallel = config.llama_server_config.parallel if config.llama_server_config else 1
    if threads < parallel:
        logger.info(
            "preflight '%s': skipping thread alignment — num_cpus=%d would undercut parallel=%d slots",
            config.name,
            threads,
            parallel,
        )
        return {}
    logger.info("preflight '%s': aligning llama-server threads to num_cpus=%d", config.name, threads)
    return {"threads": threads}
