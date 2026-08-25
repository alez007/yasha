from __future__ import annotations

import os
import re
from typing import Any

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight._mla import MLAInfo
from modelship.preflight._mla import kv_bytes_per_token as mla_kv_bytes_per_token
from modelship.preflight._sliding_window import SlidingWindowInfo, fit_len_with_sliding
from modelship.preflight.base import HardwareProfile, gpu_share_bytes

logger = get_logger("preflight.llama_cpp")

# No equivalent of vLLM's `gpu_memory_utilization` for this loader; 0.8
# leaves room for the OS, page cache, and other actors on the node.
_RAM_UTILIZATION = 0.8

# Fixed overhead for llama.cpp runtime state (compute buffers, sampler
# stacks, tokenizer vocab) — smaller than vLLM's since there's no CUDA-graph
# capture, NCCL, or fused-kernel workspace.
_OVERHEAD_FIXED_BYTES = 512 * 1024**2

# n_ctx alignment; llama.cpp has no hard requirement, powers of 256 are
# convention.
_NCTX_ALIGNMENT = 256

# Below this n_ctx, decline the recommendation instead of shipping it.
_MIN_NCTX = 512

# Fallback cap when the GGUF omits `{arch}.context_length`.
_UNKNOWN_CONTEXT_LENGTH_CAP = 32768

# Default KV-cache element size when neither `type_k` nor `type_v` is set in
# `model_kwargs`. llama.cpp defaults to fp16 (2 bytes).
_DEFAULT_KV_DTYPE_BYTES = 2

# GPU-offload constants (num_gpus >= 1).
_VRAM_UTILIZATION = 0.9
# CUDA context + compute buffers, per GPU.
_GPU_OVERHEAD_FIXED_BYTES = 1 * 1024**3
# GGUF loads near-verbatim (no repack like AWQ/Marlin), so the runtime
# footprint tracks on-disk size closely; still leave a small margin.
_GGUF_WEIGHT_OVERHEAD_FRACTION = 0.05
# llama.cpp counts the output layer as one extra offloadable "layer" beyond
# the transformer blocks — full offload means block_count + 1.
_NON_BLOCK_LAYER_EQUIV = 1
# Default context to size partial-offload ngl against when the user hasn't
# pinned n_ctx themselves.
_PARTIAL_OFFLOAD_NCTX_TARGET = 8192

# Sharded GGUF filenames (e.g. model-00001-of-00003.gguf); the resolver
# keeps only the first shard's path, so weight-bytes sums all of them.
_SHARD_SUFFIX_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")

# Per-block tensors are named `blk.<index>.<role>` across every llama.cpp arch.
_BLOCK_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
# Role prefixes that mark a block as recurrent (SSM/Mamba, RWKV time-mixing).
_RECURRENT_TENSOR_PREFIXES = ("ssm_", "time_mix_")


class LlamaServerPreflight:
    """Sizes the `llama_server` loader's launch args. Branches on
    `config.num_gpus`, not hardware discoverability."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Thread alignment is independent of context/offload sizing — recommend
        # it even when the GGUF-based math below declines.
        threads_rec = _recommend_threads(config)

        model_path = config._resolved_path
        if not model_path or not os.path.isfile(model_path):
            logger.info("preflight '%s': skipping — resolved path is not a GGUF file: %s", config.name, model_path)
            return threads_rec

        meta = _read_gguf_metadata(model_path)
        if meta is None:
            logger.info("preflight '%s': skipping — GGUF metadata unreadable at %s", config.name, model_path)
            return threads_rec

        kv_per_token = _kv_bytes_per_token(meta)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — GGUF metadata missing KV-cache geometry "
                "(block_count/head_count_kv/head_dim)",
                config.name,
            )
            return threads_rec

        weight_bytes = _weight_bytes(model_path)

        if config.num_gpus > 0:
            rec = self._recommend_gpu(config, hw, meta, kv_per_token, weight_bytes)
        else:
            rec = self._recommend_cpu(config, hw, meta, kv_per_token, weight_bytes)

        rec = self._apply_parallel_division(config, rec)
        return {**threads_rec, **rec}

    def _recommend_cpu(
        self, config: ModelshipModelConfig, hw: HardwareProfile, meta: _GGUFMeta, kv_per_token: int, weight_bytes: int
    ) -> dict[str, Any]:
        if hw.ram_bytes <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        ram_basis = hw.sizing_ram_bytes
        fallback = " [total fallback]" if not hw.available_ram_bytes else ""
        budget = ram_basis * _RAM_UTILIZATION - weight_bytes - _OVERHEAD_FIXED_BYTES
        if budget <= 0:
            logger.warning(
                "preflight '%s': no n_ctx budget (ram_avail=%.2f GiB%s, util=%.2f, weights=%.2f GiB, "
                "overhead=%.2f GiB). Model likely won't fit; deploy will be attempted anyway.",
                config.name,
                ram_basis / 1024**3,
                fallback,
                _RAM_UTILIZATION,
                weight_bytes / 1024**3,
                _OVERHEAD_FIXED_BYTES / 1024**3,
            )
            return {}

        cap = meta.context_length if meta.context_length else _UNKNOWN_CONTEXT_LENGTH_CAP
        if not meta.context_length:
            logger.info(
                "preflight '%s': GGUF metadata missing context_length; capping n_ctx at %d",
                config.name,
                _UNKNOWN_CONTEXT_LENGTH_CAP,
            )
        if meta.sliding is not None:
            max_tokens = fit_len_with_sliding(budget, kv_per_token, meta.sliding, cap)
        else:
            max_tokens = int(budget // kv_per_token)
        suggested = (max_tokens // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        suggested = min(suggested, cap)
        if suggested < _MIN_NCTX:
            logger.warning(
                "preflight '%s': budget yields n_ctx=%d (< %d); skipping recommendation",
                config.name,
                suggested,
                _MIN_NCTX,
            )
            return {}

        logger.info(
            "preflight llama_server cpu '%s': ram_avail=%.2f GiB%s util=%.2f weights=%.2f GiB kv/token=%d B%s "
            "→ suggested n_ctx=%d",
            config.name,
            ram_basis / 1024**3,
            fallback,
            _RAM_UTILIZATION,
            weight_bytes / 1024**3,
            int(kv_per_token),
            _sliding_log_suffix(meta.sliding),
            suggested,
        )

        return {"n_ctx": suggested}

    def _recommend_gpu(
        self, config: ModelshipModelConfig, hw: HardwareProfile, meta: _GGUFMeta, kv_per_token: int, weight_bytes: int
    ) -> dict[str, Any]:
        if not hw.gpus:
            logger.info(
                "preflight '%s': skipping — GPU offload requested but no GPUs discoverable on this node",
                config.name,
            )
            return {}

        fractional = 0 < config.num_gpus < 1
        num_gpus = 1 if fractional else int(config.num_gpus)
        # --split-mode layer splits proportionally to free VRAM; take the
        # num_gpus smallest-free GPUs as a lower bound on real capacity.
        picked = sorted(hw.gpus, key=lambda g: g.available_bytes)[:num_gpus]
        if len(picked) < num_gpus:
            logger.info(
                "preflight '%s': skipping — %d GPU(s) requested but only %d discoverable",
                config.name,
                num_gpus,
                len(picked),
            )
            return {}

        total_layers = meta.block_count + _NON_BLOCK_LAYER_EQUIV
        layer_bytes = weight_bytes * (1 + _GGUF_WEIGHT_OVERHEAD_FRACTION) / total_layers
        kv_per_layer = kv_per_token / meta.block_count
        ctx_cap = meta.context_length or _UNKNOWN_CONTEXT_LENGTH_CAP
        # Fractional: budget from declared share of total capacity, not free VRAM.
        if fractional:
            vram_budget = gpu_share_bytes(config, picked[0]) * _VRAM_UTILIZATION - _GPU_OVERHEAD_FIXED_BYTES
        else:
            vram_budget = (
                sum(g.available_bytes for g in picked) * _VRAM_UTILIZATION - len(picked) * _GPU_OVERHEAD_FIXED_BYTES
            )

        budget = vram_budget - layer_bytes * total_layers
        if meta.sliding is not None:
            ctx_full = fit_len_with_sliding(budget, kv_per_token, meta.sliding, ctx_cap)
        else:
            ctx_full = int(budget // kv_per_token)
        if ctx_full >= _MIN_NCTX:
            suggested = min(ctx_full, ctx_cap)
            suggested = (suggested // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
            if suggested >= _MIN_NCTX:
                logger.info(
                    "preflight llama_server gpu '%s': vram_budget=%.2f GiB across %d GPU(s), full offload%s "
                    "→ n_ctx=%d n_gpu_layers=%d",
                    config.name,
                    vram_budget / 1024**3,
                    len(picked),
                    _sliding_log_suffix(meta.sliding),
                    suggested,
                    total_layers,
                )
                return {"n_ctx": suggested, "n_gpu_layers": total_layers}

        if hw.unified_memory:
            # Partial offload would double-count VRAM/RAM as separate pools; they're the same bytes here.
            logger.warning(
                "preflight '%s': unified-memory budget only supports n_ctx=%d (< %d) at full "
                "offload; skipping recommendation",
                config.name,
                max(ctx_full, 0),
                _MIN_NCTX,
            )
            return {}

        return self._recommend_gpu_partial(
            config, hw, meta, kv_per_layer, layer_bytes, vram_budget, total_layers, ctx_cap
        )

    def _recommend_gpu_partial(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        meta: _GGUFMeta,
        kv_per_layer: float,
        layer_bytes: float,
        vram_budget: float,
        total_layers: int,
        ctx_cap: int,
    ) -> dict[str, Any]:
        # kv_per_layer carries the corrected magnitude but stays flat/linear
        # here — unlike full offload, which layers land on GPU isn't known.
        server_config = config.llama_server_config
        if server_config is not None and "n_ctx" in server_config.model_fields_set:
            target_ctx = server_config.n_ctx * server_config.parallel
        else:
            target_ctx = min(ctx_cap, _PARTIAL_OFFLOAD_NCTX_TARGET)

        def fit_ngl(ctx: int) -> int:
            denom = layer_bytes + kv_per_layer * ctx
            if denom <= 0:
                return total_layers
            return max(0, min(total_layers, int(vram_budget // denom)))

        ngl = fit_ngl(target_ctx)
        # cpu_blocks: KV-bearing transformer blocks left on CPU. cpu_layers:
        # all CPU-resident weight layers, including the output layer.
        cpu_blocks = meta.block_count - min(ngl, meta.block_count)
        cpu_layers = total_layers - ngl

        if cpu_layers > 0:
            ram_budget = hw.sizing_ram_bytes * _RAM_UTILIZATION - _OVERHEAD_FIXED_BYTES
            weight_ram = layer_bytes * cpu_layers
            kv_ram_per_ctx = kv_per_layer * cpu_blocks
            if kv_ram_per_ctx > 0:
                ctx_ram = int((ram_budget - weight_ram) // kv_ram_per_ctx)
            else:
                # No CPU-resident blocks (only the output layer is), so context
                # size doesn't affect this budget — it's a pure weight-fit check.
                ctx_ram = target_ctx if ram_budget >= weight_ram else 0
            if ctx_ram < target_ctx:
                target_ctx = ctx_ram
                if target_ctx < _MIN_NCTX:
                    logger.warning(
                        "preflight '%s': RAM budget for %d CPU-resident layers yields n_ctx=%d "
                        "(< %d); skipping recommendation",
                        config.name,
                        cpu_layers,
                        target_ctx,
                        _MIN_NCTX,
                    )
                    return {}
                # Refit once against the shrunk context — a smaller context
                # needs less VRAM per layer, so more layers may now fit.
                ngl = fit_ngl(target_ctx)

        suggested = (target_ctx // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if suggested < _MIN_NCTX:
            logger.warning(
                "preflight '%s': partial-offload budget yields n_ctx=%d (< %d); skipping recommendation",
                config.name,
                suggested,
                _MIN_NCTX,
            )
            return {}

        logger.info(
            "preflight llama_server gpu '%s': vram_budget=%.2f GiB, partial offload → n_ctx=%d n_gpu_layers=%d/%d",
            config.name,
            vram_budget / 1024**3,
            suggested,
            ngl,
            total_layers,
        )
        return {"n_ctx": suggested, "n_gpu_layers": ngl}

    def _apply_parallel_division(self, config: ModelshipModelConfig, rec: dict[str, Any]) -> dict[str, Any]:
        """llama-server splits total context (`-c`) across `parallel` slots;
        `n_gpu_layers` is per-process, so it's untouched by the division."""
        if "n_ctx" not in rec:
            return rec

        server_config = config.llama_server_config
        parallel = server_config.parallel if server_config else 1
        if parallel <= 1:
            return rec

        per_slot = (rec["n_ctx"] // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if per_slot < _MIN_NCTX:
            logger.warning(
                "preflight '%s': RAM/VRAM budget yields n_ctx=%d across %d parallel slots (< %d per slot); "
                "skipping recommendation",
                config.name,
                per_slot,
                parallel,
                _MIN_NCTX,
            )
            return {}
        logger.info(
            "preflight llama_server '%s': dividing total n_ctx budget %d across parallel=%d -> n_ctx=%d",
            config.name,
            rec["n_ctx"],
            parallel,
            per_slot,
        )
        return {**rec, "n_ctx": per_slot}


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


class _GGUFMeta:
    __slots__ = (
        "attn_block_count",
        "block_count",
        "context_length",
        "head_count_kv",
        "head_dim",
        "mla",
        "sliding",
        "v_head_dim",
    )

    def __init__(
        self,
        block_count: int,
        head_count_kv: int,
        head_dim: int,
        context_length: int | None,
        sliding: SlidingWindowInfo | None = None,
        v_head_dim: int | None = None,
        mla: MLAInfo | None = None,
        attn_block_count: int | None = None,
    ) -> None:
        self.block_count = block_count
        # On a hybrid, only the attention blocks hold a token-growing KV cache.
        self.attn_block_count = block_count if attn_block_count is None else attn_block_count
        self.head_count_kv = head_count_kv
        self.head_dim = head_dim
        self.context_length = context_length
        self.sliding = sliding
        self.v_head_dim = head_dim if v_head_dim is None else v_head_dim
        self.mla = mla


def _weight_bytes(path: str) -> int:
    """On-disk size of the GGUF file, summed across shards for a sharded
    model — the resolver only keeps the first shard's path."""
    match = _SHARD_SUFFIX_RE.search(path)
    if match is None:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    prefix = path[: match.start()]
    total_shards = match.group(2)
    total = 0
    for shard_num in range(1, int(total_shards) + 1):
        shard_path = f"{prefix}-{shard_num:0{len(total_shards)}d}-of-{total_shards}.gguf"
        try:
            total += os.path.getsize(shard_path)
        except OSError:
            logger.debug("preflight: sharded GGUF sibling missing: %s", shard_path)
    return total


def _read_gguf_metadata(path: str) -> _GGUFMeta | None:
    """Reads the architecture-relevant header fields from a GGUF file.
    `GGUFReader` mmaps and parses metadata only, no tensor data."""
    try:
        from gguf import GGUFReader
    except Exception:
        logger.debug("preflight: gguf package not available", exc_info=True)
        return None

    try:
        reader = GGUFReader(path)
    except Exception:
        logger.debug("preflight: GGUFReader failed to open %s", path, exc_info=True)
        return None

    arch = _read_string(reader, "general.architecture")
    if not arch:
        logger.debug("preflight: GGUF missing general.architecture at %s", path)
        return None

    block_count = _read_int(reader, f"{arch}.block_count")
    head_count = _read_int(reader, f"{arch}.attention.head_count")
    embedding_length = _read_int(reader, f"{arch}.embedding_length")
    key_length = _read_int(reader, f"{arch}.attention.key_length")
    context_length = _read_int(reader, f"{arch}.context_length")

    sliding = _resolve_sliding_window_gguf(reader, arch, block_count)

    # head_count_kv is per-layer on hybrid archs; sample a sliding-layer index
    # and prefer the sliding head_dim, matching vLLM's config.json estimator.
    if sliding is not None:
        sample_index = _first_sliding_layer_index(reader, arch) or 0
        head_dim = _read_int(reader, f"{arch}.attention.key_length_swa") or key_length
    else:
        sample_index = 0
        head_dim = key_length
    head_count_kv = _read_int_at(reader, f"{arch}.attention.head_count_kv", sample_index, head_count)

    # Older GGUFs without key_length fall back to embedding_length/head_count.
    if head_dim is None and embedding_length and head_count:
        head_dim = embedding_length // head_count

    if not (block_count and head_count_kv and head_dim):
        return None

    # No value_length_swa exists to pair with key_length_swa, so only trust a
    # distinct V width when there's no SWA.
    v_head_dim = head_dim
    if sliding is None:
        v_head_dim = _read_int(reader, f"{arch}.attention.value_length") or head_dim

    mla = _resolve_mla_gguf(reader, arch, head_count)

    return _GGUFMeta(
        block_count=int(block_count),
        head_count_kv=int(head_count_kv),
        head_dim=int(head_dim),
        context_length=int(context_length) if context_length else None,
        sliding=sliding,
        v_head_dim=int(v_head_dim),
        mla=mla,
        attn_block_count=_attention_block_count(reader),
    )


def _resolve_mla_gguf(reader: Any, arch: str, num_heads: int | None) -> MLAInfo | None:
    """GGUF equivalent of vllm.py's `_resolve_mla`, gated on the split
    `attn_k_b`/`attn_v_b` tensors: llama.cpp caches the compressed latent only
    when they're present, else full per-head K/V."""
    if not _has_split_mla_tensors(reader):
        return None
    kv_lora_rank = _read_int(reader, f"{arch}.attention.kv_lora_rank")
    qk_rope_head_dim = _read_int(reader, f"{arch}.rope.dimension_count")
    if not (kv_lora_rank and qk_rope_head_dim and num_heads):
        return None
    # Per-head dims stay 0: a split conversion reports key/value_length as the
    # compressed dims, and only the latent size feeds this loader's estimate.
    return MLAInfo(kv_lora_rank, qk_rope_head_dim, 0, 0, num_heads)


def _attention_block_count(reader: Any) -> int | None:
    """Blocks holding a token-growing KV cache, i.e. every block that isn't
    recurrent. Keyed on tensor names rather than the arch string, so it covers
    any hybrid llama.cpp supports. None when it can't be determined."""
    try:
        names = [t.name for t in reader.tensors]
    except (AttributeError, TypeError):
        logger.debug("preflight: GGUF reader exposes no tensor list", exc_info=True)
        return None

    recurrent: dict[int, bool] = {}
    for name in names:
        match = _BLOCK_TENSOR_RE.match(name)
        if match is None:
            continue
        index = int(match.group(1))
        recurrent[index] = recurrent.get(index, False) or match.group(2).startswith(_RECURRENT_TENSOR_PREFIXES)

    if not recurrent:
        return None
    count = sum(1 for is_recurrent in recurrent.values() if not is_recurrent)
    # Callers divide by kv_per_token, so never hand back a zero.
    return count or None


def _has_split_mla_tensors(reader: Any) -> bool:
    try:
        return any(t.name.endswith("attn_k_b.weight") for t in reader.tensors)
    except (AttributeError, TypeError):
        logger.debug("preflight: GGUF reader exposes no tensor list", exc_info=True)
        return False


def _as_list(val: Any) -> list | None:
    """Normalizes a gguf field's raw contents into a plain list when it's
    array-like (a per-layer field); None for a true scalar."""
    if val is None:
        return None
    if hasattr(val, "ndim") and getattr(val, "ndim", 0) > 0:
        try:
            return list(val.tolist()) if val.size else []
        except Exception:
            return None
    if isinstance(val, list | tuple):
        return list(val)
    return None


def _resolve_sliding_window_gguf(reader: Any, arch: str, block_count: int | None) -> SlidingWindowInfo | None:
    """GGUF equivalent of vllm.py's `_resolve_sliding_window`, reading
    `{arch}.attention.sliding_window_pattern` instead of `layer_types`."""
    window = _read_int(reader, f"{arch}.attention.sliding_window")
    if not window or not block_count:
        return None

    pattern_val = _read_field_value(reader, f"{arch}.attention.sliding_window_pattern")
    pattern_list = _as_list(pattern_val)
    if pattern_list is not None:
        n_sliding = sum(1 for v in pattern_list if bool(v))
    elif pattern_val is not None:
        try:
            period = int(pattern_val)
        except (TypeError, ValueError):
            period = 0
        n_sliding = block_count - block_count // period if period > 0 else block_count
    else:
        n_sliding = block_count

    if n_sliding <= 0:
        return None
    return SlidingWindowInfo(block_count - n_sliding, n_sliding, block_count, int(window))


def _first_sliding_layer_index(reader: Any, arch: str) -> int | None:
    pattern_list = _as_list(_read_field_value(reader, f"{arch}.attention.sliding_window_pattern"))
    if pattern_list is None:
        return None
    for i, v in enumerate(pattern_list):
        if bool(v):
            return i
    return None


def _read_int_at(reader: Any, key: str, index: int, default: int | None) -> int | None:
    """Reads a field that may be a per-layer array or a plain scalar; the
    value at `index` for an array, the scalar itself otherwise."""
    val = _read_field_value(reader, key)
    lst = _as_list(val)
    if lst is not None:
        try:
            return int(lst[index])
        except (IndexError, TypeError, ValueError):
            return default
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _read_field_value(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    # Modern gguf (>=0.10) exposes `.contents()` returning a Python primitive.
    contents = getattr(field, "contents", None)
    if callable(contents):
        try:
            return contents()
        except Exception:
            logger.debug("preflight: field.contents() raised for %s", key, exc_info=True)
    # Fallback: pull the first part out of the raw numpy data array.
    try:
        if field.data and field.parts:
            return field.parts[field.data[0]][0]
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _unwrap_scalar(val: Any) -> Any:
    """Extracts a scalar from a gguf field's raw contents — numpy array,
    Python sequence, or true scalar — taking the first element of any."""
    if val is None:
        return None
    # numpy array (1-d or higher): take the first element.
    if hasattr(val, "ndim") and getattr(val, "ndim", 0) > 0:
        try:
            return val.item(0) if val.size else None
        except (AttributeError, IndexError, ValueError):
            return None
    # Python sequence.
    if isinstance(val, list | tuple):
        return val[0] if val else None
    return val


def _read_int(reader: Any, key: str) -> int | None:
    val = _unwrap_scalar(_read_field_value(reader, key))
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _read_string(reader: Any, key: str) -> str | None:
    val = _unwrap_scalar(_read_field_value(reader, key))
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _sliding_log_suffix(sliding: SlidingWindowInfo | None) -> str:
    if sliding is None:
        return ""
    return f" swa({sliding.n_sliding_layers}/{sliding.n_total_layers} layers, window {sliding.window})"


def _kv_bytes_per_token(meta: _GGUFMeta) -> int | None:
    """Bytes of KV cache per token across all layers; element size fixed at
    fp16 (no `type_k`/`type_v` override exists)."""
    if meta.mla is not None:
        return mla_kv_bytes_per_token(meta.mla, _DEFAULT_KV_DTYPE_BYTES, meta.attn_block_count)
    # Summed separately, not `2 *`: MLA geometry caches a narrower V than K.
    return (meta.head_dim + meta.v_head_dim) * meta.attn_block_count * meta.head_count_kv * _DEFAULT_KV_DTYPE_BYTES
