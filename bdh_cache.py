# Copyright 2025 Pathway Technology, Inc.
# Optimized fork (private): packed KR/V cache — see OPT_NOTES.md
"""Packed KR/V cache: layer-contiguous buffers, slice writes, optional page growth.

v1 (cache-pack): per-layer tensors, fixed ``max_seq``, ``append``/``commit``.
v2 (cache-v2):
  - Layer-contiguous packing: one ``(n_layer, B, nh, capacity, N)`` KR buffer
    and one ``(n_layer, B, 1, capacity, D)`` V buffer (views per layer).
  - Optional ``page_size``: grow capacity in pages instead of failing early
    (still hard-capped by ``max_seq``).
  - ``stage`` writes the new block and returns a contiguous past+new view so
    multi-token+past attention can avoid ``torch.cat`` on KR/V.
v3 (decode-copy):
  - ``reserve`` returns writable slot views so RoPE can write KR in-place
    (skips a host ``copy_`` when storage dtype matches compute).
  - ``get_past`` / ``stage`` return ``narrow`` views — never ``.contiguous()``.
  - ``copy_`` into packed slots casts without an intermediate ``.to()`` alloc.
v4 (cache-page):
  - Geometric page growth (``≥2×`` capacity, page-aligned, ≤``max_seq``) so
    long generate realloc copies stay O(S) instead of O(S²) linear ``+page``.
  - ``_grow`` uses ``torch.empty`` + prefix ``copy_`` (no zero-fill of free slots).
  - ``ensure_capacity(need)`` public hint; ``n_grows`` / ``bytes_copied_on_grow``.
  - ``reserve`` / ``stage`` still grow-before-write; prior views are invalidated
    by a grow (forward already reserves before ``get_past``).

Optional ``storage_dtype=torch.float16`` stores half and casts back to
``compute_dtype`` (default fp32) before attention matmuls so RoPE score GEMMs
stay in fp32. See OPT_NOTES.md for numerical tolerance notes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class CacheManager:
    """Per-layer packed KR/V in contiguous layer-major buffers.

    Layout
    ------
    ``_kr_buf``: (n_layer, B, n_head, capacity, N)
    ``_v_buf``:  (n_layer, B, 1, capacity, D)

    During a multi-layer forward, each layer ``append``s (or ``stage``s) into
    the same write window ``[seq_len : seq_len + T]``. Call ``commit()`` once
    after the layer loop so ``seq_len`` advances only when every layer has
    written.
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
        page_size: Optional[int] = None,
    ):
        if max_seq < 1:
            raise ValueError("max_seq must be >= 1")
        if page_size is not None and page_size < 1:
            raise ValueError("page_size must be >= 1 when set")
        self.n_layer = n_layer
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.n_head = n_head
        self.n_latent = n_latent
        self.n_embd = n_embd
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype or compute_dtype
        # Hoisted equality — generate/forward checks this every layer/step.
        self.storage_matches_compute = self.storage_dtype == self.compute_dtype
        self.page_size = page_size
        self.seq_len = 0
        self._pending_t: Optional[int] = None
        # Paging stats (cache-page): realloc count + bytes of live prefix recopied.
        self.n_grows = 0
        self.bytes_copied_on_grow = 0

        # Initial capacity: full max_seq unless paging (start with one page).
        if page_size is None:
            self.capacity = max_seq
        else:
            self.capacity = min(max_seq, page_size)

        self._kr_buf = torch.zeros(
            n_layer,
            batch_size,
            n_head,
            self.capacity,
            n_latent,
            dtype=self.storage_dtype,
            device=self.device,
        )
        self._v_buf = torch.zeros(
            n_layer,
            batch_size,
            1,
            self.capacity,
            n_embd,
            dtype=self.storage_dtype,
            device=self.device,
        )

    # --- backward-compatible layer views (tests / debug) ---------------------
    @property
    def _kr(self) -> List[torch.Tensor]:
        """Per-layer views into the contiguous KR buffer."""
        return [self._kr_buf[i] for i in range(self.n_layer)]

    @property
    def _v(self) -> List[torch.Tensor]:
        """Per-layer views into the contiguous V buffer."""
        return [self._v_buf[i] for i in range(self.n_layer)]

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
        page_size: Optional[int] = None,
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
            page_size=page_size,
        )

    def ensure_capacity(self, need: int) -> None:
        """Public grow hint so ``need`` slots fit (geometric when paging).

        No-op when already large enough. Useful for long generate when the
        caller knows final S and set ``page_size`` — one realloc instead of
        mid-stream grows. Default ``generate`` still preallocates full
        ``max_seq`` when ``cache_page_size is None`` (unchanged).
        """
        self._ensure_capacity(need)

    def _next_capacity(self, need: int) -> int:
        """Geometric page growth: ``max(need, 2*capacity)``, page-aligned, ≤max_seq.

        Linear ``capacity + page_size`` recopied the live prefix every page on
        long S (≈ O(S²) bytes). Doubling keeps total grow-copy bytes ≈ O(S).
        """
        ps = self.page_size
        assert ps is not None
        target = max(int(need), self.capacity * 2)
        # Round up to a whole number of pages.
        target = ((target + ps - 1) // ps) * ps
        return min(self.max_seq, target)

    def _ensure_capacity(self, need: int) -> None:
        """Grow until ``need`` fits (geometric pages), hard-capped at ``max_seq``."""
        if need <= self.capacity:
            return
        if need > self.max_seq:
            raise RuntimeError(
                f"cache overflow: need {need} slots but max_seq={self.max_seq}"
            )
        if self.page_size is None:
            raise RuntimeError(
                f"cache overflow: need {need} slots but max_seq={self.max_seq}"
            )
        self._grow(self._next_capacity(need))

    def _grow(self, new_cap: int) -> None:
        if new_cap <= self.capacity:
            return
        # empty: free slots are written before read (reserve/append/stage).
        new_kr = torch.empty(
            self.n_layer,
            self.batch_size,
            self.n_head,
            new_cap,
            self.n_latent,
            dtype=self.storage_dtype,
            device=self.device,
        )
        new_v = torch.empty(
            self.n_layer,
            self.batch_size,
            1,
            new_cap,
            self.n_embd,
            dtype=self.storage_dtype,
            device=self.device,
        )
        # Preserve committed (+ any staged) prefix already written.
        used = self.seq_len + (self._pending_t or 0)
        if used > 0:
            new_kr[:, :, :, :used].copy_(self._kr_buf[:, :, :, :used])
            new_v[:, :, :, :used].copy_(self._v_buf[:, :, :, :used])
            elem = self._kr_buf.element_size()
            # KR: (L,B,nh,used,N) + V: (L,B,1,used,D)
            self.bytes_copied_on_grow += used * self.n_layer * self.batch_size * (
                self.n_head * self.n_latent + self.n_embd
            ) * elem
        self._kr_buf = new_kr
        self._v_buf = new_v
        self.capacity = new_cap
        self.n_grows += 1

    def get_past(
        self, level: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Valid past prefix for ``level``, cast to compute_dtype if needed.

        Returns ``narrow`` views into packed storage (no ``.contiguous()``).
        When ``seq_len < capacity`` the KR view may be non-contiguous across
        heads; last-dim stride stays 1 so decode GEMMs keep a dense (S, N).
        """
        if self.seq_len == 0:
            return None, None
        # narrow avoids an extra Python slice chain on the layer index.
        kr = self._kr_buf[level].narrow(2, 0, self.seq_len)
        v = self._v_buf[level].narrow(2, 0, self.seq_len)
        if not self.storage_matches_compute:
            # fp16 storage → fp32 for RoPE score GEMMs / V multiply
            kr = kr.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        return kr, v

    def _track_pending(self, T: int) -> int:
        """Ensure capacity for ``seq_len+T`` and record pending length."""
        end = self.seq_len + T
        self._ensure_capacity(end)
        if self._pending_t is None:
            self._pending_t = T
        elif self._pending_t != T:
            raise RuntimeError(
                f"inconsistent append T: layer wrote {T}, pending={self._pending_t}"
            )
        return end

    def reserve(
        self, level: int, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return writable views for the next block at ``[seq_len : seq_len+T]``.

        Does not copy. Caller may RoPE / write directly into the returned
        tensors (storage dtype). Does not advance ``seq_len`` — call
        ``commit`` after all layers. When ``T`` mismatches other layers'
        pending width, raises like ``append``.

        May geometrically grow the packed buffers first; any previously
        returned ``get_past`` / ``reserve`` / ``stage`` views are then
        invalid (``BDH.forward`` reserves before ``get_past`` for this reason).
        """
        if T < 1:
            raise ValueError("reserve T must be >= 1")
        end = self._track_pending(T)
        dst_kr = self._kr_buf[level].narrow(2, self.seq_len, T)
        dst_v = self._v_buf[level].narrow(2, self.seq_len, T)
        return dst_kr, dst_v

    def _write_block(
        self, level: int, new_kr: torch.Tensor, new_v: torch.Tensor
    ) -> int:
        """Write ``new_*`` at ``seq_len``; return end index. Does not commit."""
        T = new_kr.size(2)
        if T != new_v.size(2):
            raise ValueError(f"kr T={T} != v T={new_v.size(2)}")
        end = self._track_pending(T)
        # copy_ casts when dtypes differ — no intermediate .to() alloc.
        dst_kr = self._kr_buf[level].narrow(2, self.seq_len, T)
        dst_v = self._v_buf[level].narrow(2, self.seq_len, T)
        dst_kr.copy_(new_kr)
        dst_v.copy_(new_v)
        return end

    def append(self, level: int, new_kr: torch.Tensor, new_v: torch.Tensor) -> None:
        """Write this layer's new block at the current seq_len offset."""
        self._write_block(level, new_kr, new_v)

    def stage(
        self, level: int, new_kr: torch.Tensor, new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new block and return past+new views (no ``cat``).

        Used for multi-token continuation under a packed cache so attention can
        run tril over a single slice. Does not advance ``seq_len`` (call
        ``commit`` after all layers). Returned tensors are cast to
        ``compute_dtype`` when storage differs. Never calls ``.contiguous()``.

        Grows geometrically when paging and ``seq_len+T`` exceeds capacity;
        returned views always reference post-grow storage.
        """
        end = self._write_block(level, new_kr, new_v)
        kr = self._kr_buf[level].narrow(2, 0, end)
        v = self._v_buf[level].narrow(2, 0, end)
        if not self.storage_matches_compute:
            kr = kr.to(self.compute_dtype)
            v = v.to(self.compute_dtype)
        return kr, v

    def commit(self) -> None:
        """Advance seq_len after all layers have appended for this step."""
        if self._pending_t is None:
            return
        self.seq_len += self._pending_t
        self._pending_t = None

    def reset(self) -> None:
        self.seq_len = 0
        self._pending_t = None
        self.n_grows = 0
        self.bytes_copied_on_grow = 0

    @property
    def bytes_allocated(self) -> int:
        return self._kr_buf.numel() * self._kr_buf.element_size() + (
            self._v_buf.numel() * self._v_buf.element_size()
        )
