"""Unified env-gated dispatch for BDH attention backends.

Backends (``BDH_ATTN_IMPL``)::

    export BDH_ATTN_IMPL=triton   # blocked PyTorch on CPU; Triton on CUDA
    export BDH_ATTN_IMPL=eager    # default — full T×T then tril_(diagonal=-1)
    export BDH_ATTN_IMPL=blocked  # online fused tiles, no full T×T scores
    export BDH_ATTN_IMPL=online   # alias of blocked
    export BDH_ATTN_IMPL=cuda    # native ext if built, else CPU/CUDA ref
    export BDH_ATTN_AUTOGRAD=1    # optional — StrictTrilAttnFn + analytic bwd
                                   # blocked|online|triton|cuda → tiled analytic bwd
                                   # (no full T×T); eager → dense M recompute

Opt-in long-S auto-select (does **not** change default)::

    export BDH_ATTN_AUTO=1                      # off unless set
    export BDH_ATTN_AUTO_THRESHOLD=512          # decode past_len > thr → prefer
    export BDH_ATTN_AUTO_COLD_THRESHOLD=256     # optional; cold T > cold_thr → prefer
                                                # unset → mirrors AUTO_THRESHOLD

When ``BDH_ATTN_AUTO`` is truthy and ``BDH_ATTN_IMPL`` resolves to ``eager``,
long sequences switch once length exceeds the (path-specific) threshold:

- **T=1 decode** (strict ``past_len > AUTO_THRESHOLD``): **Triton** when
  ``triton_decode_available()`` (CUDA + Triton); else **blocked** (#55).
- **Cold / prefill** (strict ``T > COLD_THRESHOLD``): same
  prefer-triton-else-blocked rule so long-S ``generate`` prefill is not stuck
  on eager ``T×T`` (#72/#75). ``COLD_THRESHOLD`` defaults to
  ``AUTO_THRESHOLD`` when unset; equality stays eager on both gates.

Default shared threshold stays **512** after #72/#73/#75: wall crossover ~
T/S≥512 on CPU; mid-T (128–256) blocked wins peak mem but loses wall — use a
lower ``COLD_THRESHOLD`` only when peak-score budget matters more than wall.
Short sequences and explicit non-eager ``IMPL`` are never overridden.
Default (AUTO unset) remains eager for all paths.

Backends: eager, blocked (=online), triton, or cuda.

Wire-up in ``bdh.Attention.forward``:

- Cold path (``past_kr is None``): ``bdh_attn`` → ``resolve_cold_impl(T)``
  (``BDH_ATTN_IMPL``, plus AUTO long-T via cold thr → triton|blocked). With
  ``BDH_ATTN_AUTOGRAD=1``, wraps in ``StrictTrilAttnFn`` (analytic train).
- Multi-token + past under AUTOGRAD: also ``bdh_attn`` → ``strict_tril_attn``.
- T=1 decode (packed past KR/V): ``bdh_attn_decode`` — eager two-GEMM by
  default; with ``BDH_ATTN_AUTO=1`` and long past, eager→triton (if CUDA+Triton)
  else blocked (#55); else blocked/triton/cuda use their decode paths
  (cuda → ``kernels.cuda_attn.tril_decode``). Decode stays outside
  StrictTrilAttnFn (generate is no_grad / CacheManager-safe).
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch

from .attention import (
    _HAS_TRITON,
    _can_use_triton,
    DEFAULT_BLOCK_DECODE,
    blocked_decode_attn,
    blocked_tril_attn,
    eager_decode_attn,
    eager_tril_attn,
    triton_decode_attn,
    triton_decode_available,
    triton_tril_attn,
)
from .attention_bwd import _env_autograd_enabled, strict_tril_attn

ImplName = Literal["eager", "blocked", "triton", "cuda"]
_VALID = ("eager", "blocked", "triton", "cuda")
# "online" is accepted as an alias of blocked (fuse-scorev / OPT notes name).
_ALIASES = {"online": "blocked"}

# CPU-honest default from #55/#72/#73/#75: mid-T (128–256) blocked wins peak
# mem but loses wall; wall crossover ~T/S≥512 (cold ~3×@512, generate AUTO
# ~1.26×@1024 / ~1.39×@2048 after prefill-blocked). 512 stays the shared
# default; operators may lower COLD_THRESHOLD for peak-mem mid-T.
DEFAULT_ATTN_AUTO_THRESHOLD = 512
_TRUTHY = ("1", "true", "yes", "on")


# Env resolve cache: generate hits this every layer/step; skip strip/lower
# when the raw env string is unchanged. Explicit ``requested=`` bypasses cache.
_ATTN_IMPL_ENV: object | None = object()
_ATTN_IMPL_RESOLVED: ImplName = "eager"

_ATTN_AUTO_ENV: object | None = object()
_ATTN_AUTO_ENABLED: bool = False
_ATTN_AUTO_THR_ENV: object | None = object()
_ATTN_AUTO_THR: int = DEFAULT_ATTN_AUTO_THRESHOLD
_ATTN_AUTO_COLD_THR_ENV: object | None = object()
_ATTN_AUTO_COLD_THR: int | None = None  # None → mirror decode thr


def resolve_attn_impl(requested: str | None = None) -> ImplName:
    """Resolve BDH_ATTN_IMPL (or explicit override) to a concrete backend name.

    ``online`` is accepted and mapped to ``blocked``. Env string is cached so
    generate (n_layer × steps) skips repeated strip/lower/alias when unchanged.
    Explicit ``requested=`` bypasses the env cache.
    """
    global _ATTN_IMPL_ENV, _ATTN_IMPL_RESOLVED

    def _normalize(raw: str) -> ImplName:
        raw = (raw or "eager").strip().lower()
        raw = _ALIASES.get(raw, raw)
        if raw not in _VALID:
            raise ValueError(
                f"BDH_ATTN_IMPL must be {'|'.join(_VALID)}|online, got {raw!r}"
            )
        return raw  # type: ignore[return-value]

    if requested is not None:
        return _normalize(requested)
    env = os.environ.get("BDH_ATTN_IMPL", "eager")
    if env is _ATTN_IMPL_ENV or env == _ATTN_IMPL_ENV:
        return _ATTN_IMPL_RESOLVED
    resolved = _normalize(env or "eager")
    _ATTN_IMPL_ENV = env
    _ATTN_IMPL_RESOLVED = resolved
    return _ATTN_IMPL_RESOLVED


def attn_auto_enabled() -> bool:
    """True when ``BDH_ATTN_AUTO`` is truthy (``1|true|yes|on``). Default off."""
    global _ATTN_AUTO_ENV, _ATTN_AUTO_ENABLED
    env = os.environ.get("BDH_ATTN_AUTO", "")
    if env is _ATTN_AUTO_ENV or env == _ATTN_AUTO_ENV:
        return _ATTN_AUTO_ENABLED
    raw = (env or "").strip().lower()
    _ATTN_AUTO_ENABLED = raw in _TRUTHY
    _ATTN_AUTO_ENV = env
    return _ATTN_AUTO_ENABLED


def attn_auto_threshold() -> int:
    """Decode past-length threshold for AUTO eager→triton|blocked (default 512).

    Also the fallback for cold/prefill when ``BDH_ATTN_AUTO_COLD_THRESHOLD`` is
    unset. Retained at 512 after #72/#73/#75 (CPU wall crossover ~≥512).
    """
    global _ATTN_AUTO_THR_ENV, _ATTN_AUTO_THR
    env = os.environ.get("BDH_ATTN_AUTO_THRESHOLD", "")
    if env is _ATTN_AUTO_THR_ENV or env == _ATTN_AUTO_THR_ENV:
        return _ATTN_AUTO_THR
    raw = (env or "").strip()
    if not raw:
        thr = DEFAULT_ATTN_AUTO_THRESHOLD
    else:
        try:
            thr = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"BDH_ATTN_AUTO_THRESHOLD must be an int, got {raw!r}"
            ) from exc
        if thr < 0:
            raise ValueError(
                f"BDH_ATTN_AUTO_THRESHOLD must be >= 0, got {thr}"
            )
    _ATTN_AUTO_THR_ENV = env
    _ATTN_AUTO_THR = thr
    return _ATTN_AUTO_THR


def attn_auto_cold_threshold() -> int:
    """Cold/prefill length threshold for AUTO eager→triton|blocked.

    Reads ``BDH_ATTN_AUTO_COLD_THRESHOLD``. When unset / empty, mirrors
    ``attn_auto_threshold()`` so a single knob still covers both paths.
    Lower (e.g. 256) only when mid-T peak-score budget matters more than
    wall (#73: T∈{128,256} peak↓ wall↑).
    """
    global _ATTN_AUTO_COLD_THR_ENV, _ATTN_AUTO_COLD_THR
    env = os.environ.get("BDH_ATTN_AUTO_COLD_THRESHOLD", "")
    if env is _ATTN_AUTO_COLD_THR_ENV or env == _ATTN_AUTO_COLD_THR_ENV:
        if _ATTN_AUTO_COLD_THR is None:
            return attn_auto_threshold()
        return _ATTN_AUTO_COLD_THR
    raw = (env or "").strip()
    if not raw:
        _ATTN_AUTO_COLD_THR_ENV = env
        _ATTN_AUTO_COLD_THR = None
        return attn_auto_threshold()
    try:
        thr = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"BDH_ATTN_AUTO_COLD_THRESHOLD must be an int, got {raw!r}"
        ) from exc
    if thr < 0:
        raise ValueError(
            f"BDH_ATTN_AUTO_COLD_THRESHOLD must be >= 0, got {thr}"
        )
    _ATTN_AUTO_COLD_THR_ENV = env
    _ATTN_AUTO_COLD_THR = thr
    return _ATTN_AUTO_COLD_THR


def _resolve_auto_impl(length: int, threshold: int) -> ImplName:
    """Apply one strict AUTO gate and return its effective long-path backend."""
    if int(length) <= int(threshold):
        return "eager"
    return "triton" if triton_decode_available() else "blocked"


def resolve_decode_impl(
    past_len: int,
    requested: str | None = None,
) -> ImplName:
    """Resolve T=1 decode backend, applying optional long-S AUTO switch.

    Cold/prefill uses the twin ``resolve_cold_impl(seq_len)`` with the optional
    independent cold threshold. When AUTO is off, or ``BDH_ATTN_IMPL`` is already
    non-eager, this
    matches ``resolve_attn_impl``. When AUTO is on and the base impl is eager
    and ``past_len > threshold``:

    - ``triton`` if ``triton_decode_available()`` (CUDA + Triton) — GPU fused
      decode; host still falls back to #55 blocked when the kernel cannot run.
    - ``blocked`` otherwise — CPU-honest #55 long-S path.
    """
    name = resolve_attn_impl(requested)
    if name != "eager":
        return name
    if not attn_auto_enabled():
        return "eager"
    return _resolve_auto_impl(past_len, attn_auto_threshold())


def resolve_cold_impl(
    seq_len: int,
    requested: str | None = None,
) -> ImplName:
    """Resolve cold/prefill backend, applying optional long-T AUTO switch.

    Mirrors ``resolve_decode_impl`` with ``BDH_ATTN_AUTO`` and prefer
    Triton-when-available else blocked. Length gate uses
    ``attn_auto_cold_threshold()`` (``BDH_ATTN_AUTO_COLD_THRESHOLD``, falling
    back to ``BDH_ATTN_AUTO_THRESHOLD`` when unset) so operators can lower the
    cold switch for peak-mem mid-T without moving decode (#73/#75).

    When AUTO is off, or ``BDH_ATTN_IMPL`` is already non-eager, this matches
    ``resolve_attn_impl``. When AUTO is on and base is eager and
    ``seq_len > cold_threshold``, switches so long ``generate`` prefill is not
    stuck on full ``T×T`` eager (see ``opt/gen-long-bench`` / #72/#75).

    Short ``T`` (≤ cold threshold) stays eager under AUTO — preserves prior
    short-cold behavior. Explicit non-eager ``requested`` / ``IMPL`` is never
    overridden.
    """
    name = resolve_attn_impl(requested)
    if name != "eager":
        return name
    if not attn_auto_enabled():
        return "eager"
    return _resolve_auto_impl(seq_len, attn_auto_cold_threshold())


def bdh_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    impl: str | None = None,
    use_autograd_fn: Optional[bool] = None,
) -> torch.Tensor:
    """Compute tril(Q @ K.T, diagonal=-1) @ V with the selected backend.

    - eager:   materialize full scores (reference; matches original bdh.py)
    - blocked: online fused tiles (no full T×T scores; CPU/CUDA)
    - triton:  Triton fused kernel on CUDA; else blocked
    - cuda:    native ext via ``kernels.cuda_attn.tril_score_v`` when present,
               else that module's pure-PyTorch reference (same math)

    When ``use_autograd_fn`` is True (or BDH_ATTN_AUTOGRAD=1 and the arg is
    None), wraps the forward in ``StrictTrilAttnFn`` with analytic Q/K/V
    backward. For ``blocked`` / ``online`` / ``triton`` / ``cuda``, the
    analytic bwd is **tiled** (no full T×T); eager still uses dense M
    recompute. Default is False / env-off so the eager training path is
    unchanged.

    With ``BDH_ATTN_AUTO=1`` and base ``eager``, switches when
    ``Q.size(2) > BDH_ATTN_AUTO_COLD_THRESHOLD`` (falls back to
    ``BDH_ATTN_AUTO_THRESHOLD``, default 512) to ``triton`` if CUDA+Triton
    are available, else ``blocked``. Explicit non-eager ``impl`` /
    ``BDH_ATTN_IMPL`` is never overridden.
    """
    name = resolve_cold_impl(int(Q.size(2)), requested=impl)
    if use_autograd_fn is None:
        use_autograd_fn = _env_autograd_enabled()
    if use_autograd_fn:
        return strict_tril_attn(Q, K, V, impl=name, use_fn=True)
    if name == "eager":
        return eager_tril_attn(Q, K, V)
    if name == "blocked":
        return blocked_tril_attn(Q, K, V)
    if name == "triton":
        if _can_use_triton(Q):
            return triton_tril_attn(Q, K, V)
        return blocked_tril_attn(Q, K, V)
    # cuda
    from .cuda_attn import tril_score_v

    return tril_score_v(Q, K, V)


def bdh_attn_decode(
    Q: torch.Tensor,
    K_past: torch.Tensor,
    V_past: torch.Tensor,
    *,
    impl: str | None = None,
    block_size: int = DEFAULT_BLOCK_DECODE,
) -> torch.Tensor:
    """Attend new queries to packed past KR/V only (no full TxT).

    Used by ``Attention.forward`` incremental decode (Tq typically 1).
    Preserves tril(diagonal=-1): past slices exclude the new token, so the
    query never attends to itself.

    - eager:   ``_two_gemm_decode`` (Tq=1 BH-bmm when V per-head; else 4D @)
    - blocked / online: online tiled decode (broadcast-V tight oneshot; direct
      inference epilogue; long-S peak ~Tq×tile; larger default tile)
    - triton:  fused decode + V_BROADCAST on CUDA; blocked fallback on CPU
    - cuda:    ``tril_decode`` (Tq=1 + adaptive DECODE_TILE_N CUDA ext if built, else ref)

    With ``BDH_ATTN_AUTO=1`` and base ``eager``, switches when
    ``K_past.size(2) > BDH_ATTN_AUTO_THRESHOLD`` (default 512) to ``triton``
    if CUDA+Triton are available, else ``blocked`` (#55). Explicit non-eager
    ``impl`` / ``BDH_ATTN_IMPL`` is never overridden.
    """
    past_len = int(K_past.size(2))
    name = resolve_decode_impl(past_len, requested=impl)
    if name == "eager":
        return eager_decode_attn(Q, K_past, V_past)
    if name == "blocked":
        return blocked_decode_attn(Q, K_past, V_past, block_size=block_size)
    if name == "cuda":
        from .cuda_attn import tril_decode

        return tril_decode(Q, K_past, V_past)
    # triton → fused decode on CUDA; blocked (_tiled_score_v) otherwise
    return triton_decode_attn(Q, K_past, V_past, block_size=block_size)


def backend_info() -> dict:
    """Diagnostics for benchmarks / OPT_NOTES."""
    from .cuda_attn import has_cuda_ext, has_cuda_kernel

    impl = resolve_attn_impl()
    cuda_dev = torch.cuda.is_available()
    if impl == "eager":
        effective = "eager"
    elif impl == "blocked":
        effective = "blocked"
    elif impl == "triton":
        effective = (
            "triton_kernel"
            if cuda_dev and _HAS_TRITON
            else "blocked"
        )
    else:  # cuda
        if has_cuda_ext():
            effective = "cuda_ext" if (not cuda_dev or has_cuda_kernel()) else "cuda_ext_cpu"
        else:
            effective = "cuda_ref"
    return {
        "BDH_ATTN_IMPL": impl,
        "BDH_ATTN_AUTOGRAD": _env_autograd_enabled(),
        "BDH_ATTN_AUTO": attn_auto_enabled(),
        "BDH_ATTN_AUTO_THRESHOLD": attn_auto_threshold(),
        "BDH_ATTN_AUTO_COLD_THRESHOLD": attn_auto_cold_threshold(),
        "has_triton": _HAS_TRITON,
        "triton_decode_available": triton_decode_available(),
        "auto_decode_prefers": (
            "triton" if triton_decode_available() else "blocked"
        ),
        "auto_cold_prefers": (
            "triton" if triton_decode_available() else "blocked"
        ),
        "has_cuda_ext": has_cuda_ext(),
        "has_cuda_kernel": has_cuda_kernel(),
        "cuda": cuda_dev,
        "effective": effective,
    }
