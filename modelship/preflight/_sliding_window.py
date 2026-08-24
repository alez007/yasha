"""Sliding-window KV-cache sizing math shared by the vllm and llama_server
preflight estimators."""

from __future__ import annotations

from typing import NamedTuple

# Safety margin added to `window` before computing the saturation point —
# covers rounding to whole KV pages/batches at each loader's granularity.
_WINDOW_SLACK_TOKENS = 16


class SlidingWindowInfo(NamedTuple):
    """Layer split for interleaved sliding-window/full attention. A sliding
    layer's KV stops growing at `window` tokens."""

    n_full_layers: int
    n_sliding_layers: int
    n_total_layers: int
    window: int


def seq_kv_bytes(kv_per_token: float, sliding: SlidingWindowInfo | None, length: int) -> float:
    """KV bytes one sequence of `length` tokens occupies."""
    if sliding is None:
        return kv_per_token * length
    per_layer = kv_per_token / sliding.n_total_layers
    window_tokens = min(sliding.window + _WINDOW_SLACK_TOKENS, length)
    return per_layer * (sliding.n_full_layers * length + sliding.n_sliding_layers * window_tokens)


def fit_len_with_sliding(budget: float, kv_per_token: float, sliding: SlidingWindowInfo, ctx_cap: int) -> int:
    """Largest single-sequence max length whose KV fits `budget`, capped at
    `ctx_cap`. Apportions `kv_per_token` evenly across layers."""
    per_layer = kv_per_token / sliding.n_total_layers
    full_per_token = per_layer * sliding.n_full_layers
    sliding_per_token = per_layer * sliding.n_sliding_layers
    window_tokens = sliding.window + _WINDOW_SLACK_TOKENS

    # Below the window nothing has saturated: every layer still grows per token.
    if budget < (full_per_token + sliding_per_token) * window_tokens:
        return min(int(budget // (full_per_token + sliding_per_token)), ctx_cap)
    if full_per_token <= 0:
        # No full-attention layer: KV stops growing, only the context cap binds.
        return ctx_cap
    return min(int((budget - sliding_per_token * window_tokens) // full_per_token), ctx_cap)
