# Copyright 2025 Pathway Technology, Inc.
# Optimized fork (private): see OPT_NOTES.md

import dataclasses
import os
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from bdh_cache import CacheManager

# Eager imports so Attention.forward / rope have no lazy-import tax under
# Dynamo or the generate decode loop (n_layer × steps).
from kernels.attention_dispatch import bdh_attn, bdh_attn_decode, resolve_attn_impl
from kernels.attention_bwd import _env_autograd_enabled
from kernels.rope_dispatch import bdh_rope_rotate, resolve_rope_impl
from kernels.rope import rope_rotate_paired


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    # Published / baseline BDH does NOT tie embed ↔ lm_head (separate
    # Parameters: embed (V,D), lm_head (D,V)). Default False keeps checkpoints
    # and vs-baseline bit-identical. Opt-in True shares embed for the vocab
    # projection (GPT-style F.linear); changes trainable param count.
    tie_weights: bool = False


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )



def _try_extend_seq(past: Optional[torch.Tensor], new: Optional[torch.Tensor]):
    """If ``new`` is the next seq block after ``past`` in the same storage, return a past||new view.

    Used to skip empty+copy_ when CacheManager.reserve already wrote the new
    block adjacent to the committed prefix. Returns None when not adjacent.
    """
    if past is None or new is None:
        return None
    if past.untyped_storage().data_ptr() != new.untyped_storage().data_ptr():
        return None
    if past.stride() != new.stride():
        return None
    if past.shape[:-2] != new.shape[:-2] or past.size(-1) != new.size(-1):
        return None
    S = past.size(-2)
    T = new.size(-2)
    expected_off = past.storage_offset() + S * past.stride(-2)
    if new.storage_offset() != expected_off:
        return None
    return past.as_strided((*past.shape[:-2], S + T, past.size(-1)), past.stride())


class Attention(torch.nn.Module):
    """BDH attention: raw (unnormalized) scores with strict causal mask.

    Mask is tril(diagonal=-1): position i attends only to j < i (diagonal
    EXCLUDED). No softmax and no 1/sqrt(d) scale — matching the original paper
    code. Do NOT substitute F.scaled_dot_product_attention(is_causal=True);
    that would both include the diagonal and apply softmax+scale.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )
        # Cached (cos, sin) for rope_start=0, keyed by (T, head_dim, device, dtype).
        # Reused across layers (caller) and across batches while T is unchanged.
        self._rope_cis_key = None
        self._rope_cis = None
        # Full-sequence table for generate: positions [0, max_seq). Decode slices
        # (rope_start!=0) hit this instead of redoing arange+trig every step.
        self._rope_table_key = None
        self._rope_table = None
        # Pair views of the generate table: (cos,sin) reshaped (..., N/2, 2).
        # Built once in ensure_rope_table; T=1 decode apply can reuse without
        # per-step full-N reshape of the cold table.
        self._rope_table_pairs = None
        # Last T=1 decode cis narrow: reuse across layers when rope_start matches
        # (defensive if caller does not share cos_sin; zero-cost when shared).
        self._rope_t1_cis_key = None
        self._rope_t1_cis = None
        # Paired narrow of `_rope_table_pairs` for the same T=1 key — apply path
        # skips per-step full-N→pair reshape of cis (opt/rope-fuse-v2).
        self._rope_t1_cis_pairs = None
        # Optional: generate hoists resolve_attn_impl() once per call.
        self._attn_impl_override = None

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v, out: Optional[torch.Tensor] = None, cos_sin=None):
        """Rotate adjacent pairs into ``out`` (or a fresh empty_like).

        When ``cos_sin`` is provided, skips recomputing cos/sin (shared across
        layers / ``rope_cos_sin`` cache). Dispatch via ``BDH_ROPE_IMPL``:

        - ``eager`` (default): strided even/odd path (historical; bit-identical)
        - ``fused``: pair-contiguous pure PyTorch; Triton on CUDA when usable

        Skips redundant casts when dtypes already match ``v``.
        """
        if cos_sin is None:
            phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        else:
            phases_cos, phases_sin = cos_sin
        return bdh_rope_rotate(v, phases_cos, phases_sin, out=out)

    def _rope_phases(self, T: int, rope_start: int, device):
        assert self.freqs.dtype == torch.float32
        positions = torch.arange(
            rope_start,
            rope_start + T,
            device=device,
            dtype=self.freqs.dtype,
        ).view(1, 1, -1, 1)
        return positions * self.freqs

    def _rope_cis_cache_key(self, T: int, device):
        """Cache key: (T, head_dim, device, dtype) — training/prefill hot path."""
        head_dim = self.freqs.shape[-1]
        return (T, int(head_dim), device.type, device.index, self.freqs.dtype)

    def ensure_rope_table(self, max_T: int, device) -> None:
        """Precompute cos/sin for positions ``[0, max_T)`` (generate warm path).

        Decode steps then ``narrow`` into this table instead of ``arange`` +
        trig every token. No-op when an identical table is already resident.
        Skips Python attribute writes while Dynamo is tracing/compiling.
        """
        if max_T < 1:
            raise ValueError("max_T must be >= 1")
        key = (int(max_T), device.type, device.index, self.freqs.dtype)
        hit = self._rope_table
        if hit is not None and self._rope_table_key == key:
            cos, sin = hit
            if cos.device == device and cos.shape[-2] == max_T:
                return
        cos, sin = self.phases_cos_sin(self._rope_phases(max_T, 0, device))
        cos, sin = cos.detach(), sin.detach()
        if not torch.compiler.is_compiling():
            self._rope_table_key = key
            self._rope_table = (cos, sin)
            # Pair layout once — T=1 decode apply / tests can reuse without
            # reshaping the full table every token.
            self._rope_table_pairs = (
                cos.reshape(*cos.shape[:-1], -1, 2),
                sin.reshape(*sin.shape[:-1], -1, 2),
            )
            # Invalidate T=1 cis narrow cache (table identity changed).
            self._rope_t1_cis_key = None
            self._rope_t1_cis = None
            self._rope_t1_cis_pairs = None
            # Also warm the rope_start=0 single-T cache for max_T.
            self._rope_cis_key = self._rope_cis_cache_key(max_T, device)
            self._rope_cis = (cos, sin)

    def rope_cos_sin(self, T: int, rope_start: int, device):
        """Precompute (cos, sin); cache by (T, head_dim, device, dtype) when rope_start=0.

        Training / full prefill (rope_start=0) regenerates phases only when T,
        head_dim, device, or dtype change — reused across layers and batches.
        When ``ensure_rope_table`` has warmed a full sequence, both prefill and
        decode (``rope_start!=0``) ``narrow`` into that table (no fresh trig).
        Without a covering table, decode still computes fresh so absolute
        positions stay correct without polluting the fixed-T training table.
        """
        # Prefer full generate table when it covers [rope_start, rope_start+T).
        table = self._rope_table
        tkey = self._rope_table_key
        if table is not None and tkey is not None:
            max_T, dev_type, dev_index, dtype = tkey
            end = rope_start + T
            if (
                rope_start >= 0
                and end <= max_T
                and device.type == dev_type
                and device.index == dev_index
                and self.freqs.dtype == dtype
            ):
                cos, sin = table
                if cos.device == device and cos.shape[-2] == max_T:
                    if rope_start == 0 and T == max_T:
                        return cos, sin
                    # T=1 decode: reuse last narrow when rope_start unchanged
                    # (cross-layer if caller omits shared cos_sin).
                    if T == 1:
                        t1_key = (
                            int(rope_start),
                            device.type,
                            device.index,
                            int(max_T),
                        )
                        hit_t1 = self._rope_t1_cis
                        if (
                            hit_t1 is not None
                            and self._rope_t1_cis_key == t1_key
                            and not torch.compiler.is_compiling()
                        ):
                            return hit_t1
                        cis = (
                            cos.narrow(-2, rope_start, 1),
                            sin.narrow(-2, rope_start, 1),
                        )
                        if not torch.compiler.is_compiling():
                            self._rope_t1_cis_key = t1_key
                            self._rope_t1_cis = cis
                            # Pair narrow from generate table (same position).
                            pairs = self._rope_table_pairs
                            if pairs is not None:
                                cp, sp = pairs
                                self._rope_t1_cis_pairs = (
                                    cp.narrow(-3, rope_start, 1),
                                    sp.narrow(-3, rope_start, 1),
                                )
                            else:
                                self._rope_t1_cis_pairs = None
                        return cis
                    return cos.narrow(-2, rope_start, T), sin.narrow(-2, rope_start, T)

        if rope_start != 0:
            return self.phases_cos_sin(self._rope_phases(T, rope_start, device))

        key = self._rope_cis_cache_key(T, device)
        hit = self._rope_cis
        if hit is not None and self._rope_cis_key == key:
            cos, sin = hit
            if cos.device == device and cos.shape[-2] == T:
                return cos, sin

        cos, sin = self.phases_cos_sin(self._rope_phases(T, 0, device))
        # Detach so a cached table never holds an autograd graph across steps.
        cos, sin = cos.detach(), sin.detach()
        # Skip Python attribute writes while Dynamo is tracing/compiling.
        # Mutating ``_rope_cis`` mid-trace can confuse guards / inductor on CPU
        # when T changes; eager path and post-compile execution still warm cache.
        if not torch.compiler.is_compiling():
            self._rope_cis_key = key
            self._rope_cis = (cos, sin)
        return cos, sin

    def t1_cis_pairs(self, rope_start: int, device):
        """Return cached T=1 paired cis narrow, or build from ``_rope_table_pairs``.

        ``None`` when no covering generate table / pairs. Call after
        ``rope_cos_sin(1, rope_start, device)`` (or ``ensure_rope_table``) so the
        single-slot flat+pair caches stay aligned.
        """
        if torch.compiler.is_compiling():
            return None
        pairs = self._rope_table_pairs
        table = self._rope_table
        tkey = self._rope_table_key
        if pairs is None or table is None or tkey is None:
            return None
        max_T, dev_type, dev_index, dtype = tkey
        if not (
            0 <= rope_start < max_T
            and device.type == dev_type
            and device.index == dev_index
            and self.freqs.dtype == dtype
        ):
            return None
        t1_key = (int(rope_start), device.type, device.index, int(max_T))
        if (
            self._rope_t1_cis_pairs is not None
            and self._rope_t1_cis_key == t1_key
        ):
            return self._rope_t1_cis_pairs
        cp, sp = pairs
        cis_p = (
            cp.narrow(-3, rope_start, 1),
            sp.narrow(-3, rope_start, 1),
        )
        self._rope_t1_cis_key = t1_key
        self._rope_t1_cis_pairs = cis_p
        # Keep flat narrow cache in sync when missing (pairs-only warm path).
        if self._rope_t1_cis is None:
            cos, sin = table
            self._rope_t1_cis = (
                cos.narrow(-2, rope_start, 1),
                sin.narrow(-2, rope_start, 1),
            )
        return cis_p


    def forward(
        self,
        Q,
        K,
        V,
        rope_start: int = 0,
        past_kr: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        out_kr: Optional[torch.Tensor] = None,
    ):
        """
        Q, K: (B, nh, T, N) — K must be Q (shared latent).
        V:    (B, 1, T, D).

        Returns (out, kr_to_store, v_to_store) where store tensors are the
        RoPE'd keys / values for this block only (caller concatenates cache).

        cos_sin: optional (cos, sin) from ``rope_cos_sin`` — avoids redoing
        remainder/trig every layer when the caller shares phases.

        out_kr: optional preallocated KR write view (e.g. CacheManager.reserve).
        When set, RoPE writes in-place into it (no extra KR ``copy_`` into the
        packed cache). Must not alias Q.

        Cold path (``past_kr is None``): dispatches via
        ``kernels.attention_dispatch.bdh_attn`` according to ``BDH_ATTN_IMPL``
        (``eager`` | ``blocked`` | ``triton`` | ``cuda``; default ``eager``).

        Cached / incremental T=1 decode: ``eager`` keeps the two-GEMM form;
        ``blocked`` / ``triton`` / ``cuda`` use ``bdh_attn_decode`` against
        packed past KR/V (``cuda`` → ``kernels.cuda_attn.tril_decode``).
        Opt-in ``BDH_ATTN_AUTO=1`` switches eager→triton|blocked when length
        exceeds ``BDH_ATTN_AUTO_THRESHOLD`` (default 512): T=1 decode on
        ``past_len``, and cold/prefill on prompt ``T`` (same knobs — so long-S
        generate prefill is not stuck on eager ``T×T``). Default remains
        ``eager`` / AUTO off.

        Train path: ``BDH_ATTN_AUTOGRAD=1`` routes cold (+ multi-token-with-past)
        through ``StrictTrilAttnFn`` / analytic Q/K/V backward. T=1 decode is
        unchanged so ``CacheManager`` / ``generate`` keep working with the flag.
        """
        assert K is Q
        B, nh, T, _ = Q.size()
        if cos_sin is None:
            cos_sin = self.rope_cos_sin(T, rope_start, Q.device)
        # T=1 + generate table pairs: rotate from paired cis (skip per-step
        # flat→pair reshape). Falls back to dispatch for T>1 / no table.
        if T == 1:
            paired = self.t1_cis_pairs(rope_start, Q.device)
            if paired is not None:
                QR = rope_rotate_paired(Q, paired[0], paired[1], out=out_kr)
            else:
                QR = self.rope(None, Q, cos_sin=cos_sin, out=out_kr)
        else:
            QR = self.rope(None, Q, cos_sin=cos_sin, out=out_kr)

        if past_kr is None:
            # Training / cold prefill: unified backend dispatch.
            # Semantics: tril(QR @ QR.T, diagonal=-1) @ V — no softmax, no scale.
            # BDH_ATTN_AUTOGRAD=1 → StrictTrilAttnFn + analytic Q/K/V bwd
            # (first-class train path; default OFF keeps eager PyTorch autograd).
            out = bdh_attn(QR, QR, V)
            return out, QR, V

        # Incremental: queries attend to all past positions + earlier positions
        # in this chunk (strict: no self). Equivalent to full tril(diagonal=-1)
        # over concat(past, new). Single-token decode (T==1, S>0) is the hot path.
        S = past_kr.size(2)
        assert past_v.size(2) == S

        if T == 1 and S == 0:
            # First token: no keys to attend to → zeros
            out = V.new_zeros(B, nh, T, V.size(-1))
            return out, QR, V

        # Hoisted by generate when set; else cached env resolve.
        impl = self._attn_impl_override
        if impl is None:
            impl = resolve_attn_impl()

        if T == 1:
            # Hot decode: attend only to past (j < i). New token does not attend
            # to itself. All impls go through bdh_attn_decode (eager =
            # _two_gemm_decode; blocked/triton/cuda = tiled / fused vs packed
            # KR/V). Never materializes a full TxT score matrix.
            out = bdh_attn_decode(QR, past_kr, past_v, impl=impl)
            return out, QR, V

        # Multi-token chunk with past (prefill continuation / speculative).
        # Prefer a zero-copy past||new view when QR/V already sit in the packed
        # buffer adjacent to past (CacheManager.reserve). Else empty+copy_
        # (still cat-free). Eager keeps the split form + tril(diagonal=-1)
        # unless BDH_ATTN_AUTOGRAD=1, in which case we go through bdh_attn →
        # StrictTrilAttnFn so analytic train covers this path too. T=1 decode
        # above is unchanged (generate is @torch.no_grad; CacheManager safe).
        use_analytic = _env_autograd_enabled()
        if impl != "eager" or use_analytic:
            KR_all = _try_extend_seq(past_kr, QR)
            V_all = _try_extend_seq(past_v, V)
            if KR_all is None or V_all is None:
                _, _, _, Ndim = QR.shape
                Ddim = V.size(-1)
                KR_all = QR.new_empty(B, nh, S + T, Ndim)
                KR_all[:, :, :S].copy_(past_kr)
                KR_all[:, :, S:].copy_(QR)
                V_all = V.new_empty(B, 1, S + T, Ddim)
                V_all[:, :, :S].copy_(past_v)
                V_all[:, :, S:].copy_(V)
            out_all = bdh_attn(KR_all, KR_all, V_all, impl=impl)
            out = out_all[:, :, S:, :]
            return out, QR, V

        # Eager split: scores vs past + tril self-block; V_all via view or copy_.
        if S > 0 and T > 1:
            scores = QR.new_empty(B, nh, T, S + T)
            scores[:, :, :, :S] = QR @ past_kr.mT
            self_scores = QR @ QR.mT
            self_scores.tril_(diagonal=-1)
            scores[:, :, :, S:] = self_scores
            V_all = _try_extend_seq(past_v, V)
            if V_all is None:
                V_all = V.new_empty(B, 1, S + T, V.size(-1))
                V_all[:, :, :S].copy_(past_v)
                V_all[:, :, S:].copy_(V)
            out = scores @ V_all
        elif S > 0:
            out = (QR @ past_kr.mT) @ past_v
        else:
            self_scores = QR @ QR.mT
            self_scores.tril_(diagonal=-1)
            out = self_scores @ V

        return out, QR, V


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Return a nullcontext or torch.autocast for optional AMP.

    Default (amp_dtype=None) is fp32 — no autocast. Supported: float16, bfloat16.
    """
    from contextlib import nullcontext

    if amp_dtype is None:
        return nullcontext()
    if amp_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"amp_dtype must be torch.float16 or torch.bfloat16, got {amp_dtype}"
        )
    device_type = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        # Weight layouts (baseline shapes preserved for state_dict / tests):
        #   encoder / encoder_v: (nh, D, N) — train einsum → contiguous (B,T,nh,N);
        #     eval encoder also keeps a versioned contiguous (nh*N, D) F.linear cache
        #   decoder: (nh*N, D) — F.linear with .T view (no hot-path .contiguous())
        #   lm_head: (D, vocab) — same F.linear pattern
        # Optional biases registered as None; F.linear / add fuse when present.
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        # Keep nn.LayerNorm for baseline API parity (no affine params → empty
        # state_dict). Hot path never calls self.ln — only F.layer_norm.
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self._ln_shape = (D,)
        self._ln_eps = float(self.ln.eps)
        self.embed = nn.Embedding(config.vocab_size, D)
        # Float p + F.dropout (not nn.Dropout): torch RNG only, compile-friendly.
        # p==0 or eval → true identity (no RNG op) — bit-identical to baseline.
        self.dropout_p = float(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # Vocab projection: baseline stores untied lm_head (D, V). Optional
        # tie_weights shares embed.weight via F.linear (no separate Parameter).
        if config.tie_weights:
            self.register_parameter("lm_head", None)
        else:
            self.lm_head = nn.Parameter(
                torch.zeros((D, config.vocab_size)).normal_(std=0.02)
            )
        # Optional bias for F.linear epilogue; absent in baseline checkpoints.
        self.register_parameter("lm_head_bias", None)

        # Optional fused biases (absent by default — not in baseline checkpoints).
        self.register_parameter("encoder_bias", None)
        self.register_parameter("encoder_v_bias", None)
        self.register_parameter("decoder_bias", None)

        # Layout-v2: versioned contiguous (nh*N, D) for eval F.linear encoder path.
        # Not Parameters — absent from state_dict. Refresh when weight._version/ptr bumps.
        self._encoder_w_lin = None
        self._encoder_w_lin_ver = None
        self._encoder_w_lin_ptr = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _ln(self, x: torch.Tensor) -> torch.Tensor:
        """Affine-free LayerNorm over the last dim via ``F.layer_norm``.

        Prefer ``F.layer_norm`` over ``self.ln(x)`` (``nn.Module.__call__`` /
        hooks) so Dynamo sees a single ATen op. Explicit ``weight=None`` /
        ``bias=None`` — no affine Python optionality. Bit-identical to ``self.ln``.
        """
        return F.layer_norm(
            x, self._ln_shape, weight=None, bias=None, eps=self._ln_eps
        )

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Token embed → LN → (B, 1, T, D) for the residual body.

        LN runs on contiguous ``(B, T, D)`` from ``nn.Embedding`` (no cast), then
        a size-1 unsqueeze for head-broadcast layout. Same last-dim LN math as
        ``LN(embed(idx).unsqueeze(1))``.
        """
        x = self._ln(self.embed(idx))
        return x.unsqueeze(1)

    def _vocab_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Final hidden ``(B, 1, T, D)`` or ``(B, T, D)`` → contiguous ``(B, T, V)``.

        Untied (default): ``F.linear(h, lm_head.T)`` — transpose is a *view*;
        no ``.contiguous()`` copy of a (V, D) clone. Bit-identical to
        ``h @ lm_head`` for the stored ``(D, V)`` Parameter.

        Tied (``tie_weights=True``): ``F.linear(h, embed.weight)`` sharing the
        embedding matrix (published baseline does **not** tie).
        """
        # Squeeze singleton broadcast dim without copy when size == 1.
        h = x.squeeze(1) if x.dim() == 4 and x.size(1) == 1 else x.reshape(
            x.size(0), -1, x.size(-1)
        )
        if not h.is_contiguous():
            h = h.contiguous()
        bias = self.lm_head_bias
        if self.lm_head is None:
            return F.linear(h, self.embed.weight, bias)
        w = self.lm_head
        if not w.is_contiguous():
            w = w.contiguous()
        # F.linear wants (out, in)=(V, D); stored weight is (D, V).
        return F.linear(h, w.transpose(0, 1), bias)

    def _lm_head_last_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Write last-token logits into preallocated ``out`` ``(B, V)`` (fp32).

        Hot path for T=1 decode: ``torch.mm`` / ``addmm`` (``B>1``) or ``mv`` /
        ``addmv`` (``B==1``) with ``out=`` avoids a fresh ``(B, 1, V)`` alloc
        from ``_vocab_logits`` each step. Math matches
        ``_vocab_logits(x)[:, -1, :].float()``.
        """
        if x.dim() == 4:
            h = x[:, 0, -1, :]
        else:
            h = x[:, -1, :]
        if h.dtype != out.dtype:
            h = h.to(dtype=out.dtype)
        if not h.is_contiguous():
            h = h.contiguous()
        bias = self.lm_head_bias
        # out= GEMMs disallow requires_grad args; decode is inference_mode but
        # detach so the helper is safe if called outside that context.
        h = h.detach()
        B = h.size(0)
        if self.lm_head is None:
            # F.linear(h, embed.weight) = h @ weight.T + bias; weight (V, D).
            w = self.embed.weight.detach()
            if w.dtype != out.dtype:
                w = w.to(dtype=out.dtype)
            # B=1: mv/addmv into out[0] (same math as mm; fewer BLAS dims).
            if B == 1:
                if bias is not None:
                    torch.addmv(
                        bias.detach().to(dtype=out.dtype),
                        w,
                        h.squeeze(0),
                        out=out.squeeze(0),
                    )
                else:
                    torch.mv(w, h.squeeze(0), out=out.squeeze(0))
            else:
                wt = w.transpose(0, 1)  # (D, V) view
                if bias is not None:
                    torch.addmm(bias.detach().to(dtype=out.dtype), h, wt, out=out)
                else:
                    torch.mm(h, wt, out=out)
        else:
            w = self.lm_head.detach()
            if not w.is_contiguous():
                w = w.contiguous()
            if w.dtype != out.dtype:
                w = w.to(dtype=out.dtype)
            if B == 1:
                # stored (D, V); logits = h @ w ≡ w.T @ h  (mv wants (V, D)).
                wt = w.transpose(0, 1)  # (V, D) view
                if bias is not None:
                    torch.addmv(
                        bias.detach().to(dtype=out.dtype),
                        wt,
                        h.squeeze(0),
                        out=out.squeeze(0),
                    )
                else:
                    torch.mv(wt, h.squeeze(0), out=out.squeeze(0))
            else:
                if bias is not None:
                    torch.addmm(bias.detach().to(dtype=out.dtype), h, w, out=out)
                else:
                    torch.mm(h, w, out=out)
        return out

    @staticmethod
    def _sample_from_logits(
        logits_bv: torch.Tensor,
        *,
        scale: float | None,
        do_topk: bool,
        top_k_n: int,
        probs_buf: torch.Tensor,
        softmax,
        multinomial,
    ) -> torch.Tensor:
        """Sample one token from ``(B, V)`` logits into reused ``probs_buf``.

        Owns ``logits_bv`` for the call (may ``mul_`` / ``masked_fill_``).
        When ``do_topk``, optional fused path: softmax+multinomial over the top-k
        values only, then ``gather`` indices — same distribution as mask/-inf,
        fewer full-vocab softmax flops. Full-vocab multinomial kept when
        ``top_k`` is None (default) for RNG parity with prior generate.
        """
        if scale is not None:
            logits_bv.mul_(scale)
        if do_topk:
            V = logits_bv.size(-1)
            k = top_k_n if top_k_n <= V else V
            # Fused top-k: sample in the k-space (distribution-equivalent to
            # mask+softmax over V; RNG stream differs — only used when top_k set).
            values, indices = torch.topk(logits_bv, k, dim=-1)
            if k < V:
                # Softmax only over k; write into a narrow of probs_buf when B*k fits,
                # else allocate a small (B, k) probs (k << V).
                probs_k = softmax(values, dim=-1)
                idx_k = multinomial(probs_k, num_samples=1)
                return indices.gather(1, idx_k)
            # k == V: fall through to full-vocab path using values order? Use mask.
            values_min = values[:, -1:]
            logits_bv.masked_fill_(logits_bv < values_min, float("-inf"))
        torch.softmax(logits_bv, dim=-1, out=probs_buf)
        return multinomial(probs_buf, num_samples=1)

    def _residual_ln(self, x: torch.Tensor, y_mlp: torch.Tensor) -> torch.Tensor:
        """``LN(x + LN(y_mlp))`` via ``F.layer_norm`` only — compile-friendly.

        Still the #30 hot path: ``F.layer_norm`` (not ``self.ln`` / module hooks),
        no Python control flow, no ``is_grad_enabled`` branch. Deepen vs #30's
        out-of-place ``x + y``: reuse the inner LN *output* buffer with
        ``y.add_(x)`` so the residual sum does not allocate a separate add
        temporary. Autograd-safe (in-place into a fresh LN output; ``y_mlp`` and
        ``x`` stay intact for backward). Dynamo: 0 graph breaks; bit-identical
        to ``self.ln(x + self.ln(y_mlp))`` at dropout=0.

        Torch has no eager affine-free fused add+LN op; inductor may still fuse
        ``layer_norm(x + y)`` on GPU compile — this path keeps ``F.layer_norm``
        so that remains available if we ever flip back to out-of-place add.
        """
        shape = self._ln_shape
        eps = self._ln_eps
        y = F.layer_norm(y_mlp, shape, weight=None, bias=None, eps=eps)
        y.add_(x)
        return F.layer_norm(y, shape, weight=None, bias=None, eps=eps)

    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Compile-friendly dropout via ``F.dropout`` (ATen / torch RNG only).

        Never uses Python ``random`` / NumPy RNG (those graph-break ``torch.compile``).
        When ``dropout_p == 0`` **or** ``not self.training`` (eval), returns ``x``
        unchanged so the compiled / FX graph has **no** dropout RNG ops
        (``bernoulli_`` / ``native_dropout``) — bit-identical to ``nn.Dropout(0)``
        and to eval-mode ``nn.Dropout(p)``.
        """
        p = self.dropout_p
        if p == 0.0 or not self.training:
            return x
        return F.dropout(x, p=p, training=True)

    @staticmethod
    def _linear(x: torch.Tensor, weight_in_out: torch.Tensor, bias=None) -> torch.Tensor:
        """``x @ weight_in_out`` (+ bias) via ``F.linear`` on a transpose *view*.

        ``weight_in_out`` is stored ``(in, out)`` (baseline decoder / lm_head).
        Never calls ``.contiguous()`` on the transpose — BLAS gets an op(A) flag.
        When ``bias`` is not None it is fused into the GEMM epilogue.
        """
        return F.linear(x, weight_in_out.transpose(-2, -1), bias)

    @staticmethod
    def _bias_relu_(out: torch.Tensor, bias=None) -> torch.Tensor:
        """Optional bias into a fresh GEMM buffer, then in-place ReLU.

        ``out`` must be a buffer we own (einsum / matmul result). When ``bias``
        is set, ``add_`` fuses into that buffer — no ``out + bias`` temporary.
        Bit-identical to ``F.relu(out + bias, inplace=True)`` / ``F.relu(out)``.
        """
        if bias is not None:
            out.add_(bias)
        return F.relu(out, inplace=True)

    @staticmethod
    def _hdn_as_linear_weight(weight_hdn: torch.Tensor) -> torch.Tensor:
        """``(nh, D, N)`` -> ``(nh*N, D)`` for ``F.linear`` (out, in).

        Layout ``(nh,D,N)`` is baseline/state_dict-compatible; the transpose+reshape
        copies once (``(nh,N,D)`` is not a free view of ``(nh,D,N)``). Train /
        ``bias is None`` default still prefers einsum to skip this copy; eval
        forward reuses a versioned contiguous cache (see ``_encoder_w_lin_cached``).
        """
        nh, D, N = weight_hdn.shape
        return weight_hdn.transpose(1, 2).reshape(nh * N, D)

    def _refresh_encoder_w_lin_cache(self, *, force: bool = False) -> None:
        """Build/refresh contiguous ``(nh*N, D)`` encoder cache (call from ``train``/``eval``).

        Kept out of ``forward`` so ``torch.compile`` does not see module-attr
        mutations mid-graph. Not a Parameter; not in ``state_dict``.
        ``force=True`` rebuilds even if version/ptr look unchanged (post-load).
        """
        w = self.encoder
        ver = w._version
        ptr = w.data_ptr()
        if (
            force
            or self._encoder_w_lin is None
            or self._encoder_w_lin_ver != ver
            or self._encoder_w_lin_ptr != ptr
        ):
            self._encoder_w_lin = self._hdn_as_linear_weight(w).contiguous()
            self._encoder_w_lin_ver = ver
            self._encoder_w_lin_ptr = ptr

    def _encoder_w_lin_cached(self) -> torch.Tensor:
        """Return contiguous ``(nh*N, D)`` for ``self.encoder``, refreshing if needed."""
        self._refresh_encoder_w_lin_cache()
        assert self._encoder_w_lin is not None
        return self._encoder_w_lin

    def train(self, mode: bool = True):
        """Enter train (einsum path) or eval (warm encoder linear cache).

        Cache refresh happens here — not inside ``forward`` — for Dynamo safety.
        """
        r = super().train(mode)
        if mode:
            self._encoder_w_lin = None
            self._encoder_w_lin_ver = None
            self._encoder_w_lin_ptr = None
        else:
            self._refresh_encoder_w_lin_cache()
        return r

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Invalidate/rebuild encoder layout cache after weight load.

        ``load_state_dict`` may ``copy_`` into the same storage (ptr unchanged) and
        on some builds leave ``_version`` ambiguous relative to our cache — using a
        stale ``(nh*N, D)`` buffer would silently desync eval ``F.linear`` from the
        Parameter (breaks compile-vs-eager decode parity). Always drop the cache
        around the load; re-warm when already in eval.
        """
        self._encoder_w_lin = None
        self._encoder_w_lin_ver = None
        self._encoder_w_lin_ptr = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        if not self.training:
            self._refresh_encoder_w_lin_cache(force=True)

    @staticmethod
    def _encoder_relu(
        x_btd: torch.Tensor,
        weight_hdn: torch.Tensor,
        bias=None,
        *,
        weight_lin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project ``(B,T,D)`` with ``(nh,D,N)`` -> contiguous ``(B,T,nh,N)`` + ReLU.

        Default (``bias is None``, no ``weight_lin``): ``einsum("btd,hdn->bthn")``
        then in-place ReLU — train hot path; no weight transpose-copy.

        When ``weight_lin`` is a contiguous ``(nh*N, D)`` (eval cache) and/or
        ``bias`` is set: ``F.linear`` with optional fused bias epilogue, then
        ``view`` + ReLU. Bit-identical to ``F.relu(einsum(...) [+ bias])`` at
        ``dropout=0``. Always-on uncached ``F.linear`` still loses on CPU (#47);
        the cached eval path is the layout-v2 deepen.
        """
        if weight_lin is None and bias is None:
            out = torch.einsum("btd,hdn->bthn", x_btd, weight_hdn)
            return F.relu(out, inplace=True)
        nh, D, N = weight_hdn.shape
        B, T, _ = x_btd.shape
        wl = weight_lin if weight_lin is not None else BDH._hdn_as_linear_weight(weight_hdn)
        b = None if bias is None else bias.reshape(nh * N)
        out = F.linear(x_btd, wl, b)
        return F.relu(out.view(B, T, nh, N), inplace=True)

    def _encoder_relu_fwd(self, x_btd: torch.Tensor) -> torch.Tensor:
        """Forward encoder proj: eval uses cached ``F.linear``; train keeps einsum.

        Forward is side-effect free (cache warmed in ``train(False)``/``eval()``).
        Under ``torch.compile`` tracing we keep the einsum path so Dynamo does not
        bake a non-Parameter layout buffer into the graph (that desynced packed
        T=1 decode vs eager). Eager eval/generate still hits the cached linear.
        channels_last / always-on uncached linear measured slower here — not used.
        ``encoder_v`` stays einsum (matmul+contig loses).
        """
        bias = self.encoder_bias
        if (
            not self.training
            and not torch.compiler.is_compiling()
        ):
            w = self.encoder
            lin = self._encoder_w_lin
            if (
                lin is not None
                and self._encoder_w_lin_ver == w._version
                and self._encoder_w_lin_ptr == w.data_ptr()
            ):
                return self._encoder_relu(x_btd, w, bias, weight_lin=lin)
        return self._encoder_relu(x_btd, self.encoder, bias)

    @staticmethod
    def _encoder_v_relu(
        y_bhtd: torch.Tensor, weight_hdn: torch.Tensor, bias=None
    ) -> torch.Tensor:
        """Project ``(B,nh,T,D)`` with ``(nh,D,N)`` -> contiguous ``(B,T,nh,N)`` + ReLU.

        Stays on einsum: ``matmul`` yields ``(B,nh,T,N)`` and a permute→contiguous
        to ``(B,T,nh,N)`` costs an extra activation copy (measured slower on CPU).
        Optional bias still fused via in-place ``add_`` before ReLU.
        """
        out = torch.einsum("bhtd,hdn->bthn", y_bhtd, weight_hdn)
        return BDH._bias_relu_(out, bias)

    def _mlp_merge(
        self,
        x_bthn: torch.Tensor,
        y_bthn: torch.Tensor,
        B: int,
        T: int,
        nh: int,
        N: int,
    ) -> torch.Tensor:
        """Sparse product → dropout → free ``view`` → decoder ``F.linear``.

        Expects contiguous ``(B,T,nh,N)`` from the encoder paths so
        ``view(B,T,nh*N)`` is a zero-copy reshape (no ``permute→contiguous``).
        Decoder bias (if set) fuses in the ``F.linear`` epilogue. Always
        out-of-place ``x*y`` (no ``is_grad_enabled`` branch — one Dynamo graph).
        """
        xy = self._dropout(x_bthn * y_bthn)
        # Hot path: contiguous (B,T,nh,N) → view. Never unconditional .contiguous().
        return self._linear(xy.view(B, T, nh * N), self.decoder, self.decoder_bias)

    def forward(
        self,
        idx,
        targets=None,
        cache: Optional[Union[list, CacheManager]] = None,
        *,
        logits_out: Optional[torch.Tensor] = None,
    ):
        """
        cache: optional packed ``CacheManager`` (preferred) or legacy list of
        length n_layer with None | {'kr', 'v'} (cat every step). Mutated in place.
        When cache is provided, idx is the new token block; RoPE continues from
        cached length.

        logits_out: optional preallocated ``(B, V)`` fp32 buffer. When set and
        ``T==1``, writes last-token logits via ``_lm_head_last_into`` and returns the
        same ``(B, V)`` buffer — generate decode reuse (no per-step vocab alloc).

        AMP: wrap the call in torch.autocast(...) or use generate(amp_dtype=...).
        Default path remains fp32. RoPE phases stay float32.
        """
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Keep activations as (B, T, D) — avoid a permanent unsqueeze(1) that forces
        # broadcast GEMMs and a later permute+contiguous before the decoder.
        x = self._ln(self.embed(idx))

        packed = isinstance(cache, CacheManager)
        rope_start = 0
        if packed:
            rope_start = cache.seq_len
        elif cache is not None and cache[0] is not None:
            rope_start = cache[0]["kr"].size(2)

        # One cos/sin for all layers (same T, rope_start) — cuts RoPE trig allocs.
        cos_sin = self.attn.rope_cos_sin(T, rope_start, x.device)

        # Used only for packed-cache in-place RoPE safety (not for sparse product).
        grad_enabled = torch.is_grad_enabled()

        for level in range(C.n_layer):
            # Contiguous (B, T, nh, N): decoder merge is a view; attn gets a permute view.
            x_bthn = self._encoder_relu_fwd(x)
            x_sparse = x_bthn.permute(0, 2, 1, 3)  # (B, nh, T, N) — view, no copy

            past_kr = past_v = None
            out_kr = None
            v_tok = x.unsqueeze(1)
            if packed:
                # Inference + matching dtypes: RoPE straight into packed KR slot
                # (skips one copy_ per layer). V still copy_'d into its slot.
                # Under grad, keep a fresh QR so autograd is not writing into
                # the cache buffer.
                # reserve BEFORE get_past so page growth cannot invalidate past views.
                inplace_ok = (
                    not grad_enabled
                    and cache.storage_matches_compute
                    and v_tok.dtype == cache.storage_dtype
                )
                if inplace_ok:
                    dst_kr, dst_v = cache.reserve(level, T)
                    past_kr, past_v = cache.get_past(level)
                    dst_v.copy_(v_tok)
                    out_kr = dst_kr
                    v_tok = dst_v  # attn + return share packed V slot
                else:
                    past_kr, past_v = cache.get_past(level)
            elif cache is not None and cache[level] is not None:
                past_kr = cache[level]["kr"]
                past_v = cache[level]["v"]

            yKV, new_kr, new_v = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=v_tok,
                rope_start=rope_start,
                past_kr=past_kr,
                past_v=past_v,
                cos_sin=cos_sin,
                out_kr=out_kr,
            )
            if packed:
                if out_kr is None:
                    cache.append(level, new_kr, new_v)
                # else: reserve already wrote KR (RoPE) + V (copy_)
            elif cache is not None:
                if cache[level] is None:
                    cache[level] = {"kr": new_kr, "v": new_v}
                else:
                    cache[level] = {
                        "kr": torch.cat([past_kr, new_kr], dim=2),
                        "v": torch.cat([past_v, new_v], dim=2),
                    }

            yKV = self._ln(yKV)

            # encoder_v -> contiguous (B,T,nh,N); _mlp_merge does product/dropout/view/linear.
            y_bthn = self._encoder_v_relu(yKV, self.encoder_v, self.encoder_v_bias)
            yMLP = self._mlp_merge(x_bthn, y_bthn, B, T, nh, N)
            # Residual + double LN: F.layer_norm only (see _residual_ln).
            x = self._residual_ln(x, yMLP)

        if packed:
            cache.commit()

        loss = None
        if logits_out is not None:
            if T != 1:
                raise ValueError("logits_out requires T==1 decode")
            self._lm_head_last_into(x, logits_out)
            # Return the owned (B, V) buffer (no unsqueeze); generate ignores it.
            logits = logits_out
        else:
            logits = self._vocab_logits(x)
            if targets is not None:
                # Contiguous (B*T, V) for CE; view is safe after _vocab_logits.
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )

        return logits, loss

    @torch.inference_mode()
    @torch.compiler.disable
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        *,
        cache_dtype: torch.dtype | None = None,
        amp_dtype: Optional[torch.dtype] = None,
        cache_page_size: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive decode with packed KR/V cache (preallocated max_seq).

        Marked ``torch.compiler.disable``: dynamic length, multinomial,
        and CacheManager mutation graph-break / fight CUDA graphs. Call from
        eager (or after training); ``forward`` itself may still be compiled.

        ``torch.inference_mode`` (not only ``no_grad``): skips view-tracking
        version counters on the decode loop — measurable host win on CPU.

        cache_dtype: optional storage dtype for the cache (e.g. torch.float16).
        Compute stays fp32 for RoPE score GEMMs when storage is narrower.

        amp_dtype: optional torch.float16 / torch.bfloat16 for autocast over
        forward. Default None keeps fp32 (no autocast). Sampling logits use fp32.

        cache_page_size: optional CacheManager page growth (see bdh_cache).
        Default None preallocates exactly ``prompt + max_new_tokens``.

        Output tokens are written into a preallocated buffer (no per-step
        ``torch.cat`` on the token sequence). Decode steps fuse ``lm_head`` into
        a reused ``(B, V)`` buffer (``logits_out``) and sample via reused probs.
        """
        was_training = self.training
        self.eval()

        B, prompt_len = idx.size()
        max_seq = prompt_len + max_new_tokens
        device = idx.device
        cache = CacheManager.from_config(
            self.config,
            batch_size=B,
            max_seq=max_seq,
            device=device,
            compute_dtype=torch.float32,
            storage_dtype=cache_dtype,
            page_size=cache_page_size,
        )
        # One RoPE trig pass for the whole generate span; per-step narrow slices.
        self.attn.ensure_rope_table(max_seq, device)
        # Resolve attn/rope backends once for the whole decode loop.
        attn_impl = resolve_attn_impl()
        resolve_rope_impl()  # warm env cache for rope path
        self.attn._attn_impl_override = attn_impl

        # Preallocate full output — eliminates generate's remaining aten::cat.
        out = torch.empty(B, max_seq, dtype=idx.dtype, device=device)
        out[:, :prompt_len].copy_(idx)

        # Hoist sampling constants / callables out of the decode loop.
        temp = float(temperature)
        scale = None if temp == 1.0 else (1.0 / temp)
        do_topk = top_k is not None
        top_k_n = int(top_k) if do_topk else 0
        softmax = F.softmax
        multinomial = torch.multinomial
        vocab = int(self.config.vocab_size)
        # Reused across decode steps: fused lm_head (mm out=) + softmax out=.
        logits_buf = torch.empty(B, vocab, dtype=torch.float32, device=device)
        probs_buf = torch.empty(B, vocab, dtype=torch.float32, device=device)
        sample = self._sample_from_logits

        try:
            with _autocast_context(device, amp_dtype):
                # Prefill: full (B, T, V) once. First sample may use a view when
                # fp32 + no in-place scale/top-k (avoids an extra Tensor.copy_).
                # Later steps: lm_head writes straight into logits_buf (T=1).
                logits, _ = self(idx, cache=cache)

                for t in range(max_new_tokens):
                    if t == 0:
                        step = logits[:, -1, :]
                        need_owned = (
                            step.dtype != torch.float32
                            or scale is not None
                            or do_topk
                        )
                        if need_owned:
                            if step.dtype != torch.float32:
                                logits_buf.copy_(step.float())
                            else:
                                logits_buf.copy_(step)
                            sample_in = logits_buf
                        else:
                            sample_in = step
                    else:
                        sample_in = logits_buf
                    idx_next = sample(
                        sample_in,
                        scale=scale,
                        do_topk=do_topk,
                        top_k_n=top_k_n,
                        probs_buf=probs_buf,
                        softmax=softmax,
                        multinomial=multinomial,
                    )
                    out[:, prompt_len + t] = idx_next[:, 0]
                    self(idx_next, cache=cache, logits_out=logits_buf)
        finally:
            self.attn._attn_impl_override = None

        if was_training:
            self.train()
        return out
