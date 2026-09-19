"""Operator-level CPU profiler for BDH Attention / forward / generate.

Uses torch.profiler (CPU; CUDA activities when available). Writes a Chrome
trace under benchmarks/traces/ and prints top ops by CPU time / self CPU.

Usage:
  .venv/bin/python benchmarks/profile_forward.py
  .venv/bin/python benchmarks/profile_forward.py --mode attn --T 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bdh  # noqa: E402

TRACES = Path(__file__).resolve().parent / "traces"


def _activities(device: torch.device) -> list:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _make_model(cfg: bdh.BDHConfig, device: torch.device) -> bdh.BDH:
    torch.manual_seed(0)
    return bdh.BDH(cfg).to(device).eval()


def _default_cfg() -> bdh.BDHConfig:
    return bdh.BDHConfig(
        n_layer=4,
        n_embd=128,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
    )


def profile_attention(device, B, T, warmup, wait, active, trace_dir=TRACES):
    cfg = _default_cfg()
    m = _make_model(cfg, device)
    nh, D = cfg.n_head, cfg.n_embd
    N = D * cfg.mlp_internal_dim_multiplier // nh
    Q = torch.randn(B, nh, T, N, device=device)
    V = torch.randn(B, 1, T, D, device=device)

    def run():
        with torch.no_grad():
            with record_function("Attention.forward"):
                m.attn(Q=Q, K=Q, V=V)

    return _run_profiler(run, "attn", device, warmup, wait, active, trace_dir)


def profile_forward(device, B, T, warmup, wait, active, trace_dir=TRACES):
    cfg = _default_cfg()
    m = _make_model(cfg, device)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def run():
        with torch.no_grad():
            with record_function("BDH.forward"):
                m(x)

    return _run_profiler(run, "forward", device, warmup, wait, active, trace_dir)


def profile_generate(device, prompt_len, new_tokens, warmup, wait, active, trace_dir=TRACES):
    cfg = _default_cfg()
    m = _make_model(cfg, device)
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)

    def run():
        with torch.no_grad():
            with record_function("BDH.generate"):
                torch.manual_seed(0)
                m.generate(prompt.clone(), max_new_tokens=new_tokens)

    return _run_profiler(run, "generate", device, warmup, wait, active, trace_dir)


def _run_profiler(fn, label, device, warmup, wait, active, trace_dir=TRACES):
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    trace_path = trace_dir / f"bdh_{label}_{device.type}.json"
    with profile(
        activities=_activities(device),
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=None,
    ) as prof:
        for _ in range(wait + warmup + active):
            fn()
            prof.step()
    prof.export_chrome_trace(str(trace_path))
    print(f"\n=== profile mode={label} device={device} ===")
    print(f"chrome_trace={trace_path}")
    print("\n--- top ops by CPU total time ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    print("\n--- top ops by self CPU time ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    return prof, trace_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("attn", "forward", "generate", "all"), default="all")
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--wait", type=int, default=1)
    p.add_argument("--active", type=int, default=3)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--trace-dir", type=Path, default=TRACES)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    modes = ("attn", "forward", "generate") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode == "attn":
            profile_attention(device, args.B, args.T, args.warmup, args.wait, args.active, args.trace_dir)
        elif mode == "forward":
            profile_forward(device, args.B, args.T, args.warmup, args.wait, args.active, args.trace_dir)
        else:
            profile_generate(device, args.prompt, args.new_tokens, args.warmup, args.wait, args.active, args.trace_dir)


if __name__ == "__main__":
    main()
