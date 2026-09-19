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


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


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

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v, out: Optional[torch.Tensor] = None, cos_sin=None):
        """Rotate adjacent pairs into ``out`` (or a fresh empty_like).

        Avoids a full-size ``v_rot`` temporary. When ``cos_sin`` is provided,
        skips recomputing cos/sin (shared across layers in one forward).
        Skips redundant casts when dtypes already match ``v``.
        """
        if cos_sin is None:
            phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        else:
            phases_cos, phases_sin = cos_sin

        if out is None:
            out = torch.empty_like(v)
        elif out is v:
            raise ValueError("rope out= must not alias v")

        ve = v[..., 0::2]
        vo = v[..., 1::2]
        ce = phases_cos[..., 0::2]
        se = phases_sin[..., 0::2]
        co = phases_cos[..., 1::2]
        so = phases_sin[..., 1::2]

        if v.dtype != phases_cos.dtype:
            # Match baseline cast-then-add rounding when phases are fp32 and v is not.
            out[..., 0::2] = (ve * ce).to(v.dtype) + ((-vo) * se).to(v.dtype)
            out[..., 1::2] = (vo * co).to(v.dtype) + (ve * so).to(v.dtype)
        else:
            out[..., 0::2] = ve * ce - vo * se
            out[..., 1::2] = vo * co + ve * so
        return out

    def _rope_phases(self, T: int, rope_start: int, device):
        assert self.freqs.dtype == torch.float32
        positions = torch.arange(
            rope_start,
            rope_start + T,
            device=device,
            dtype=self.freqs.dtype,
        ).view(1, 1, -1, 1)
        return positions * self.freqs

    def rope_cos_sin(self, T: int, rope_start: int, device):
        """Precompute (cos, sin) once per forward; reuse across layers."""
        return self.phases_cos_sin(self._rope_phases(T, rope_start, device))

    def forward(
        self,
        Q,
        K,
        V,
        rope_start: int = 0,
        past_kr: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Q, K: (B, nh, T, N) — K must be Q (shared latent).
        V:    (B, 1, T, D).

        Returns (out, kr_to_store, v_to_store) where store tensors are the
        RoPE'd keys / values for this block only (caller concatenates cache).

        cos_sin: optional (cos, sin) from ``rope_cos_sin`` — avoids redoing
        remainder/trig every layer when the caller shares phases.

        Cold path (``past_kr is None``): dispatches via
        ``kernels.attention_dispatch.bdh_attn`` according to ``BDH_ATTN_IMPL``
        (``eager`` | ``blocked`` | ``triton`` | ``cuda``; default ``eager``).

        Cached / incremental path: always eager PyTorch matmuls. Custom
        kernels (blocked/triton/cuda) do not yet support incremental decode,
        so ``generate()`` only applies ``BDH_ATTN_IMPL`` on the cold prefill
        when the cache is empty; decode steps stay eager.
        """
        assert K is Q
        B, nh, T, _ = Q.size()
        if cos_sin is None:
            cos_sin = self.rope_cos_sin(T, rope_start, Q.device)
        QR = self.rope(None, Q, cos_sin=cos_sin)

        if past_kr is None:
            # Training / cold prefill: unified backend dispatch.
            # Semantics: tril(QR @ QR.T, diagonal=-1) @ V — no softmax, no scale.
            # BDH_ATTN_AUTOGRAD=1 remains supported by the dispatcher.
            from kernels.attention_dispatch import bdh_attn

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
            # to itself. blocked/triton use tiled decode vs packed KR/V slices
            # (no full TxT); eager keeps the simple two-GEMM form.
            if impl == "eager":
                scores = QR @ past_kr.transpose(-2, -1)  # (B, nh, 1, S)
                out = scores @ past_v
            else:
                from kernels.attention_dispatch import bdh_attn_decode

                out = bdh_attn_decode(QR, past_kr, past_v, impl=impl)
            return out, QR, V

        # Multi-token chunk with past (prefill continuation / speculative).
        # Prefer concat + dispatched tril attn when not eager so blocked/triton
        # stay consistent with the cold path; eager keeps the split form.
        if impl != "eager":
            from kernels.attention_dispatch import bdh_attn

            KR_all = torch.cat([past_kr, QR], dim=2)
            V_all = torch.cat([past_v, V], dim=2)
            out_all = bdh_attn(KR_all, KR_all, V_all, impl=impl)
            out = out_all[:, :, S:, :]
            return out, QR, V

        parts = []
        if S > 0:
            parts.append(QR @ past_kr.transpose(-2, -1))  # (B, nh, T, S)
        if T > 1:
            self_scores = QR @ QR.transpose(-2, -1)
            self_scores.tril_(diagonal=-1)
            parts.append(self_scores)  # (B, nh, T, T)

        if S > 0 and T > 1:
            scores = torch.cat(parts, dim=-1)  # (B, nh, T, S+T)
            V_all = torch.cat([past_v, V], dim=2)
            out = scores @ V_all
        elif S > 0:
            out = parts[0] @ past_v
        else:
            out = parts[0] @ V

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
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        packed = isinstance(cache, CacheManager)
        rope_start = 0
        if packed:
            rope_start = cache.seq_len
        elif cache is not None and cache[0] is not None:
            rope_start = cache[0]["kr"].size(2)

        # One cos/sin for all layers (same T, rope_start) — cuts RoPE trig allocs.
        cos_sin = self.attn.rope_cos_sin(T, rope_start, x.device)

        for level in range(C.n_layer):
            # Q/K latent: fuse ReLU in-place onto encoder GEMM output (no 2nd buffer).
            x_sparse = F.relu(x @ self.encoder, inplace=True)

            past_kr = past_v = None
            if packed:
                past_kr, past_v = cache.get_past(level)
            elif cache is not None and cache[level] is not None:
                past_kr = cache[level]["kr"]
                past_v = cache[level]["v"]

            yKV, new_kr, new_v = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                rope_start=rope_start,
                past_kr=past_kr,
                past_v=past_v,
                cos_sin=cos_sin,
            )
            if packed:
                cache.append(level, new_kr, new_v)
            elif cache is not None:
                if cache[level] is None:
                    cache[level] = {"kr": new_kr, "v": new_v}
                else:
                    cache[level] = {
                        "kr": torch.cat([past_kr, new_kr], dim=2),
                        "v": torch.cat([past_v, new_v], dim=2),
                    }

            yKV = self.ln(yKV)

            # encoder_v projection + in-place ReLU (same fuse as Q path)
            y_sparse = F.relu(yKV @ self.encoder_v, inplace=True)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            # Contiguous (B, T, nh, N) before merge — friendlier for compile/GEMM
            xy = xy_sparse.permute(0, 2, 1, 3).contiguous().view(B, 1, T, N * nh)
            yMLP = xy @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        if packed:
            cache.commit()

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        *,
        cache_dtype: torch.dtype | None = None,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Autoregressive decode with packed KR/V cache (preallocated max_seq).

        cache_dtype: optional storage dtype for the cache (e.g. torch.float16).
        Compute stays fp32 for RoPE score GEMMs when storage is narrower.

        amp_dtype: optional torch.float16 / torch.bfloat16 for autocast over
        forward. Default None keeps fp32 (no autocast). Sampling logits use fp32.
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
        )
        device = idx.device
        with _autocast_context(device, amp_dtype):
            logits, _ = self(idx, cache=cache)

            for _ in range(max_new_tokens):
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
                idx = torch.cat((idx, idx_next), dim=1)
                logits, _ = self(idx_next, cache=cache)

        if was_training:
            self.train()
        return idx
