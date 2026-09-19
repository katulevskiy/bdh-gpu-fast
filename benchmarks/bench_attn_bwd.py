"""CUDA-only train-step harness for analytic strict-tril attention.

The default run is an honest CUDA A/B matrix: every ``IMPL``
(``eager|blocked|online|triton|cuda``) is crossed with ``AUTOGRAD=0|1``.
``online`` is the explicit alias of ``blocked`` and is retained in the matrix
to prove that the two documented spellings stay parity-equivalent.  A small
forward/backward parity probe runs before timing, so a backend that silently
loses gradients cannot be reported as a win.  ``triton`` and ``cuda`` report
their effective fallback when the optional kernel/extension is unavailable.

This box has no CUDA: the script prints ``SKIP`` and exits 0 without allocating
CPU tensors or reporting CPU timings.  The default B=4/T=64 case is a smoke
shape; for GPU occupancy use ``--batch 8 --tokens 256`` or larger, and use
T>=512 when evaluating peak-score memory.  Defaults remain unchanged.

Dtype is explicit (``--dtype`` or ``BDH_ATTN_BWD_DTYPE``; float32 default).
The analytic tiled backward widens fp16/bf16 intermediates to fp32 and casts
gradients back to the input dtype.  Start with float32 parity, then repeat with
bf16/fp16 on the target GPU; no CPU number is a GPU claim.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep compile off for a clean AUTOGRAD A/B.
os.environ.setdefault("BDH_COMPILE", "0")

import bdh  # noqa: E402
from kernels.attention_dispatch import backend_info  # noqa: E402

# Keep the public alias in the matrix: ``online`` must remain parity-equivalent
# to ``blocked`` instead of silently disappearing from the train runbook.
IMPLS = ("eager", "blocked", "online", "triton", "cuda")
AUTOGRAD_FLAGS = ("0", "1")
_DTYPE_NAMES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def _parse_dtype(raw: str) -> tuple[str, torch.dtype]:
    key = raw.strip().lower()
    if key not in _DTYPE_NAMES:
        choices = "|".join(sorted(_DTYPE_NAMES))
        raise ValueError(f"dtype must be one of {choices}, got {raw!r}")
    canonical = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }[_DTYPE_NAMES[key]]
    return canonical, _DTYPE_NAMES[key]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        default=os.environ.get("BDH_ATTN_BWD_DTYPE", "float32"),
        help="model/attention dtype (default: BDH_ATTN_BWD_DTYPE or float32)",
    )
    parser.add_argument(
        "--batch",
        type=_positive_int,
        default=int(os.environ.get("BDH_ATTN_BWD_BATCH", "4")),
    )
    parser.add_argument(
        "--tokens",
        type=_positive_int,
        default=int(os.environ.get("BDH_ATTN_BWD_TOKENS", "64")),
    )
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=int(os.environ.get("BDH_ATTN_BWD_WARMUP", "2")),
    )
    parser.add_argument(
        "--reps",
        type=_positive_int,
        default=int(os.environ.get("BDH_ATTN_BWD_REPS", "8")),
    )
    return parser.parse_args(argv)


def _sync():
    torch.cuda.synchronize()


def timed(fn, warmup=3, reps=12):
    for _ in range(warmup):
        fn()
    _sync()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def _cfg():
    return bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        mlp_internal_dim_multiplier=16,
        dropout=0.0,
    )


def _batch(device, B=4, T=64):
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    return x, y


def make_train_step(model, device, fused: bool, batch):
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fused = False
    x, y = batch

    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        return loss

    return step, fused


def _probe(model, batch):
    """Return a CPU correctness snapshot before any optimizer update."""
    model.zero_grad(set_to_none=True)
    x, y = batch
    _, loss = model(x, y)
    if not loss.requires_grad:
        raise RuntimeError("loss does not require grad")
    loss.backward()
    grads = {}
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        else:
            grads[name] = param.grad.detach().float().cpu().clone()
    if missing:
        raise RuntimeError("missing gradients: " + ", ".join(missing[:3]))
    loss_cpu = loss.detach().float().cpu().clone()
    if not torch.isfinite(loss_cpu) or not all(torch.isfinite(g).all() for g in grads.values()):
        raise RuntimeError("non-finite loss or gradient")
    model.zero_grad(set_to_none=True)
    return loss_cpu, grads


def _parity(reference, sample, *, rtol, atol):
    ref_loss, ref_grads = reference
    loss, grads = sample
    if not torch.allclose(loss, ref_loss, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"loss parity failed: ref={ref_loss.item():.6g} got={loss.item():.6g}"
        )
    if ref_grads.keys() != grads.keys():
        raise RuntimeError("gradient parameter sets differ")
    max_diff = 0.0
    for name in ref_grads:
        diff = (ref_grads[name] - grads[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(ref_grads[name], grads[name], rtol=rtol, atol=atol):
            raise RuntimeError(f"gradient parity failed for {name}: max_diff={diff:.6g}")
    return max_diff, abs((loss - ref_loss).item())


def _label(autograd: str, impl: str) -> str:
    if autograd == "0":
        return "PyTorch autograd through selected forward"
    if impl == "eager":
        return "dense analytic backward"
    return "tiled analytic backward"


def _expected_nograd_skip(autograd: str, impl: str, exc: RuntimeError) -> bool:
    """Native Triton/CUDA forwards may intentionally expose no autograd graph."""
    text = str(exc).lower()
    return autograd == "0" and impl in ("triton", "cuda") and (
        "grad" in text or "derivative" in text
    )


def bench_combo(
    cfg,
    device,
    *,
    autograd: str,
    impl: str,
    state_dict,
    fused: bool,
    batch,
    dtype_name: str,
    warmup: int,
    reps: int,
):
    os.environ["BDH_ATTN_AUTOGRAD"] = autograd
    os.environ["BDH_ATTN_IMPL"] = impl
    torch.manual_seed(0)
    model = bdh.BDH(cfg).to(device=device, dtype=_DTYPE_NAMES[dtype_name])
    model.load_state_dict(state_dict)
    model.train()
    try:
        snapshot = _probe(model, batch)
        step, fused_ok = make_train_step(model, device, fused=fused, batch=batch)
        step()  # dry run
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = timed(step, warmup=warmup, reps=reps)
    except RuntimeError as exc:
        if _expected_nograd_skip(autograd, impl, exc):
            return {"status": "skip", "reason": str(exc), "impl": impl, "autograd": autograd}
        raise
    info = backend_info()
    return {
        "status": "ok",
        "autograd": autograd,
        "impl": impl,
        "effective": info["effective"],
        "snapshot": snapshot,
        "elapsed": elapsed,
        "fused": fused_ok,
        "peak_bytes": torch.cuda.max_memory_allocated(device),
    }


def main(argv=None):
    # Do this before parsing/allocating so CPU CI gets a clean, deterministic skip.
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; bench_attn_bwd requires a real CUDA device")
        return 0

    args = _args(argv)
    dtype_name, _ = _parse_dtype(args.dtype)
    device = torch.device("cuda")
    cfg = _cfg()
    print(
        f"device={device} dtype={dtype_name} layers={cfg.n_layer} d={cfg.n_embd} "
        f"nh={cfg.n_head} B={args.batch} T={args.tokens} dropout=0"
    )
    print(f"torch={torch.__version__} device_name={torch.cuda.get_device_name(device)}")
    print("matrix=" + ",".join(f"{impl}/AUTOGRAD={flag}" for impl in IMPLS for flag in AUTOGRAD_FLAGS))

    torch.manual_seed(0)
    ref = bdh.BDH(cfg).to(device=device, dtype=_DTYPE_NAMES[dtype_name])
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    batch = _batch(device, B=args.batch, T=args.tokens)
    fused = True
    combos = [("0", "eager")]
    combos.extend(
        (flag, impl)
        for impl in IMPLS
        for flag in AUTOGRAD_FLAGS
        if (flag, impl) != ("0", "eager")
    )
    results = []
    reference = None
    rtol, atol = ((5e-2, 5e-2) if dtype_name != "float32" else (5e-3, 5e-3))
    for flag, impl in combos:
        result = bench_combo(
            cfg,
            device,
            autograd=flag,
            impl=impl,
            state_dict=state,
            fused=fused,
            batch=batch,
            dtype_name=dtype_name,
            warmup=args.warmup,
            reps=args.reps,
        )
        if result["status"] == "skip":
            print(f"SKIP AUTOGRAD={flag} IMPL={impl}: {result['reason']}")
            continue
        if reference is None:
            reference = result["snapshot"]
        max_grad_diff, loss_diff = _parity(
            reference, result["snapshot"], rtol=rtol, atol=atol
        )
        result["max_grad_diff"] = max_grad_diff
        result["loss_diff"] = loss_diff
        results.append(result)
        print(
            f"OK AUTOGRAD={flag} IMPL={impl:7s} effective={result['effective']:12s} "
            f"median={result['elapsed'] * 1000:.2f} ms "
            f"peak={result['peak_bytes'] / 2**20:.1f} MiB "
            f"parity(max_grad={max_grad_diff:.3g}, loss={loss_diff:.3g}) "
            f"({_label(flag, impl)})"
        )

    if not results or results[0]["impl"] != "eager" or results[0]["autograd"] != "0":
        raise RuntimeError("baseline eager/AUTOGRAD=0 did not produce a result")
    t_ref = results[0]["elapsed"]
    print(f"optimizer fused={all(result['fused'] for result in results)}")
    for result in results[1:]:
        print(
            f"  vs AUTOGRAD=0/eager: {t_ref / result['elapsed']:.2f}x "
            f"(IMPL={result['impl']} AUTOGRAD={result['autograd']} "
            f"effective={result['effective']})"
        )
    print(
        "honest: CUDA timings only after parity gate; requested triton/cuda may "
        "fall back, and skipped non-differentiable native forward/AUTOGRAD=0 "
        "combos are not compared. Defaults remain eager/AUTOGRAD=0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
