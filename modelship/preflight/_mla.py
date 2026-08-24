"""MLA (Multi-head Latent Attention) KV-cache sizing shared by the vllm and
llama_server preflight estimators."""

from __future__ import annotations

from typing import NamedTuple


class MLAInfo(NamedTuple):
    """DeepSeek-style MLA geometry: one shared compressed latent per token,
    not per-head K/V."""

    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    num_heads: int


def kv_bytes_per_token(mla: MLAInfo, dtype_bytes: int, num_layers: int) -> int:
    """One shared latent per token per layer: no per-head factor, no x2 for K/V."""
    return (mla.kv_lora_rank + mla.qk_rope_head_dim) * dtype_bytes * num_layers
