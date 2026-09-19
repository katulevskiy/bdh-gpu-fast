# Copyright 2025 Pathway Technology, Inc.
# Optimized fork (private): packed KR/V cache — see OPT_NOTES.md
"""Packed KR/V cache: preallocate max_seq, write slices (no cat per step).

Optional ``storage_dtype=torch.float16`` stores tensors in half precision and
casts back to ``compute_dtype`` (default fp32) before attention matmuls so RoPE
score GEMMs stay in fp32. See OPT_NOTES.md for numerical tolerance notes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class CacheManager:
    """Per-layer packed KR/V buffers with a shared sequence length.

    Layout
    ------
    kr[level]: (B, n_head, max_seq, N)
    v[level]:  (B, 1, max_seq, D)

    During a multi-layer forward, each layer ``append``s into the same write
    window ``[seq_len : seq_len + T]``. Call ``commit()`` once after the layer
    loop so ``seq_len`` advances only when every layer has written.
    """

    def __init__(
        self,
        n_layer: int,
        max_seq: int,
        batch_size: int,
        n_head: int,
        n_latent: int,
        n_embd: int,
        device: torch.device | str,
        *,
        compute_dtype: torch.dtype = torch.float32,
        storage_dtype: Optional[torch.dtype] = None,
    ):
        if max_seq < 1:
            raise ValueError("max_seq must be >= 1")
        self.n_layer = n_layer
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.n_head = n_head
        self.n_latent = n_latent
        self.n_embd = n_embd
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype or compute_dtype
        self.seq_len = 0
        self._pending_t: Optional[int] = None

        self._kr = [
            torch.zeros(
                batch_size,
                n_head,
                max_seq,
                n_latent,
                dtype=self.storage_dtype,
                device=self.device,
            )
            for _ in range(n_layer)
        ]
        self._v = [
            torch.zeros(
                batch_size,
                1,
                max_seq,
                n_embd,
                dtype=self.storage_dtype,
                device=self.device,
            )
            for _ in range(n_layer)
        ]

    @classmethod
    def from_config(
        cls,
        config,
        batch_size: int,
        max_seq: int,
        device: torch.device | str,
        *,
        compute_dtype: torch.dtype = torch.float32,
        storage_dtype: Optional[torch.dtype] = None,
    ) -> "CacheManager":
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        return cls(
            n_layer=config.n_layer,
            max_seq=max_seq,
            batch_size=batch_size,
            n_head=nh,
            n_latent=N,
            n_embd=D,
            device=device,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
        )

    def get_past(
        self, level: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Valid past prefix for ``level``, cast to compute_dtype if needed."""
        if self.seq_len == 0:
            return None, None
        kr = self._kr[level][:, :, : self.seq_len]
        v = self._v[level][:, :, : self.seq_len]
        if self.storage_dtype != self.compute_dtype:
            # fp16 storage → fp32 for RoPE score GEMMs / V multiply
            kr = kr.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        return kr, v

    def append(self, level: int, new_kr: torch.Tensor, new_v: torch.Tensor) -> None:
        """Write this layer's new block at the current seq_len offset."""
        T = new_kr.size(2)
        if T != new_v.size(2):
            raise ValueError(f"kr T={T} != v T={new_v.size(2)}")
        end = self.seq_len + T
        if end > self.max_seq:
            raise RuntimeError(
                f"cache overflow: need {end} slots but max_seq={self.max_seq}"
            )
        if self._pending_t is None:
            self._pending_t = T
        elif self._pending_t != T:
            raise RuntimeError(
                f"inconsistent append T: layer wrote {T}, pending={self._pending_t}"
            )
        # copy_ avoids dtype surprises; cast if storage is narrower
        self._kr[level][:, :, self.seq_len : end].copy_(
            new_kr.to(self.storage_dtype)
        )
        self._v[level][:, :, self.seq_len : end].copy_(
            new_v.to(self.storage_dtype)
        )

    def commit(self) -> None:
        """Advance seq_len after all layers have appended for this step."""
        if self._pending_t is None:
            return
        self.seq_len += self._pending_t
        self._pending_t = None

    def reset(self) -> None:
        self.seq_len = 0
        self._pending_t = None

    @property
    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._kr) + sum(
            t.numel() * t.element_size() for t in self._v
        )
