"""BDH custom attention kernels (Triton + pure-PyTorch fallbacks)."""

from .attention_dispatch import bdh_attn, resolve_attn_impl

__all__ = ["bdh_attn", "resolve_attn_impl"]
