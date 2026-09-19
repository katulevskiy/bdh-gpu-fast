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

# Eager import so Attention.forward has no lazy-import graph break under Dynamo.
from kernels.attention_dispatch import bdh_attn, bdh_attn_decode
from kernels.attention_bwd import _env_autograd_enabled


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

        from kernels.rope_dispatch import bdh_rope_rotate

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

    def rope_cos_sin(self, T: int, rope_start: int, device):
        """Precompute (cos, sin); cache by (T, head_dim, device, dtype) when rope_start=0.

        Training / full prefill (rope_start=0) regenerates phases only when T,
        head_dim, device, or dtype change — reused across layers and batches.
        Decode (rope_start!=0) computes fresh so absolute positions stay correct
        without polluting the fixed-T training table.
        """
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
        Default remains ``eager``.

        Train path: ``BDH_ATTN_AUTOGRAD=1`` routes cold (+ multi-token-with-past)
        through ``StrictTrilAttnFn`` / analytic Q/K/V backward. T=1 decode is
        unchanged so ``CacheManager`` / ``generate`` keep working with the flag.
        """
        assert K is Q
        B, nh, T, _ = Q.size()
        if cos_sin is None:
            cos_sin = self.rope_cos_sin(T, rope_start, Q.device)
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

        impl = os.environ.get("BDH_ATTN_IMPL", "eager").strip().lower()

        if T == 1:
            # Hot decode: attend only to past (j < i). New token does not attend
            # to itself. blocked/triton/cuda use decode vs packed KR/V slices
            # (no full TxT); eager keeps the simple two-GEMM form.
            if impl == "eager":
                scores = QR @ past_kr.mT  # (B, nh, 1, S)
                out = scores @ past_v
            else:
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
        #   encoder / encoder_v: (nh, D, N) — used via einsum → contiguous (B,T,nh,N)
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
        # p==0 is a true identity (no RNG op) — bit-identical to baseline Dropout(0).
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
        """Final hidden ``(B, 1, T, D)`` → contiguous ``(B, T, V)`` logits.

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

    def _residual_ln(self, x: torch.Tensor, y_mlp: torch.Tensor) -> torch.Tensor:
        """``LN(x + LN(y_mlp))`` via ``F.layer_norm`` only — compile-friendly.

        Pure functional (out-of-place ``x + y``): no in-place ``add_`` aliasing
        for AOTAutograd functionalization, and no Python control flow. Prefer
        ``F.layer_norm`` over ``nn.LayerNorm`` module calls. Bit-identical to
        ``self.ln(x + self.ln(y_mlp))`` and to the prior in-place reuse form.
        """
        shape = self._ln_shape
        eps = self._ln_eps
        y = F.layer_norm(y_mlp, shape, weight=None, bias=None, eps=eps)
        return F.layer_norm(x + y, shape, weight=None, bias=None, eps=eps)

    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Compile-friendly dropout via ``F.dropout`` (ATen / torch RNG only).

        Never uses Python ``random`` / NumPy RNG (those graph-break ``torch.compile``).
        When ``dropout_p == 0`` (common under ``BDH_COMPILE`` benches / baseline
        parity tests), returns ``x`` unchanged so the compiled graph has **no**
        dropout RNG ops — bit-identical to ``nn.Dropout(0)``.
        """
        p = self.dropout_p
        if p == 0.0:
            return x
        return F.dropout(x, p=p, training=self.training)

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
    def _encoder_relu(
        x_btd: torch.Tensor, weight_hdn: torch.Tensor, bias=None
    ) -> torch.Tensor:
        """Project ``(B,T,D)`` with ``(nh,D,N)`` -> contiguous ``(B,T,nh,N)`` + ReLU.

        Einsum avoids broadcasting ``(B,1,T,D) @ (nh,D,N)`` (extra expand/copy on
        CPU). Output layout is decoder-friendly: ``view(B,T,nh*N)`` is a free view.
        Optional ``bias`` is fused via in-place add before ReLU (no add temp).
        """
        out = torch.einsum("btd,hdn->bthn", x_btd, weight_hdn)
        return BDH._bias_relu_(out, bias)

    @staticmethod
    def _encoder_v_relu(
        y_bhtd: torch.Tensor, weight_hdn: torch.Tensor, bias=None
    ) -> torch.Tensor:
        """Project ``(B,nh,T,D)`` with ``(nh,D,N)`` -> contiguous ``(B,T,nh,N)`` + ReLU.

        Optional ``bias`` fused via in-place add before ReLU (same as encoder).
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
    ):
        """
        cache: optional packed ``CacheManager`` (preferred) or legacy list of
        length n_layer with None | {'kr', 'v'} (cat every step). Mutated in place.
        When cache is provided, idx is the new token block; RoPE continues from
        cached length.

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
            x_bthn = self._encoder_relu(x, self.encoder, self.encoder_bias)
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
                    and cache.storage_dtype == cache.compute_dtype
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

        logits = self._vocab_logits(x)
        loss = None
        if targets is not None:
            # Contiguous (B*T, V) for CE; view is safe after _vocab_logits.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
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

        cache_dtype: optional storage dtype for the cache (e.g. torch.float16).
        Compute stays fp32 for RoPE score GEMMs when storage is narrower.

        amp_dtype: optional torch.float16 / torch.bfloat16 for autocast over
        forward. Default None keeps fp32 (no autocast). Sampling logits use fp32.

        cache_page_size: optional CacheManager page growth (see bdh_cache).
        Default None preallocates exactly ``prompt + max_new_tokens``.

        Output tokens are written into a preallocated buffer (no per-step
        ``torch.cat`` on the token sequence).
        """
        was_training = self.training
        self.eval()

        B, prompt_len = idx.size()
        max_seq = prompt_len + max_new_tokens
        cache = CacheManager.from_config(
            self.config,
            batch_size=B,
            max_seq=max_seq,
            device=idx.device,
            compute_dtype=torch.float32,
            storage_dtype=cache_dtype,
            page_size=cache_page_size,
        )
        # Preallocate full output — eliminates generate's remaining aten::cat.
        out = torch.empty(
            B, max_seq, dtype=idx.dtype, device=idx.device
        )
        out[:, :prompt_len].copy_(idx)
        device = idx.device
        with _autocast_context(device, amp_dtype):
            logits, _ = self(idx, cache=cache)

            for t in range(max_new_tokens):
                # Sampling in fp32 for numerical stability under AMP
                step_logits = logits[:, -1, :].float() / temperature
                if top_k is not None:
                    values, _ = torch.topk(
                        step_logits, min(top_k, step_logits.size(-1))
                    )
                    step_logits = step_logits.clone()
                    step_logits[step_logits < values[:, [-1]]] = float("-inf")
                probs = F.softmax(step_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                pos = prompt_len + t
                out[:, pos : pos + 1] = idx_next
                logits, _ = self(idx_next, cache=cache)

        if was_training:
            self.train()
        return out
