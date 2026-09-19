# Copyright Pathway Technology, Inc.
# Optional fast entrypoint: aggressive compile + CUDA-graph-oriented notes.
#
# Prefer `python train.py` for the default private-opt path. This module sets
# env defaults then runs the same loop via train helpers.
#
# CUDA graphs (when a GPU exists)
# -------------------------------
# `torch.compile(mode="reduce-overhead")` enables CUDA graphs for the compiled
# region when shapes/addresses are stable. Requirements / gotchas:
#   1. Static input shapes (fixed B, T) — our train loop already uses fixed
#      BATCH_SIZE × BLOCK_SIZE.
#   2. No CPU↔GPU sync inside the step (no `.item()`, `.cpu()`, print on CUDA
#      tensors) — logging uses `float(loss.detach())` only every LOG_FREQ.
#   3. `zero_grad(set_to_none=True)` plays nicer with graph capture than
#      filling grads with zeros in-place.
#   4. Fused AdamW (`fused=True`) keeps the optimizer on-device.
#   5. Prefetch via a side CUDA stream (BatchPrefetcher) so H2D does not stall
#      the captured compute stream.
#   6. Dynamic `generate()` / variable-T decode will graph-break; keep sample
#      generation outside the hot train loop (already the case).
# On CPU-only boxes, reduce-overhead has little benefit; inductor still needs
# a working g++ + Python.h. Fall back with BDH_COMPILE=0 if compile fails.

from __future__ import annotations

import os

import torch

# Aggressive defaults — must be set before importing train (it reads env at import).
os.environ.setdefault("BDH_COMPILE", "1")
os.environ.setdefault(
    "BDH_COMPILE_MODE",
    "reduce-overhead" if torch.cuda.is_available() else "default",
)
os.environ.setdefault("BDH_FUSED_ADAMW", "1")

import train as tr  # noqa: E402


def main():
    print(
        "train_fast: "
        f"compile={tr.USE_COMPILE} mode={tr.COMPILE_MODE} "
        f"fused_adamw={tr.USE_FUSED_ADAMW} device={tr.device}"
    )
    tr.fetch_data()
    model = tr.bdh.BDH(tr.BDH_CONFIG).to(tr.device)
    loader = tr.BatchPrefetcher("train")
    x, y = loader.next()
    model = tr.maybe_compile(model, example_x=x, example_y=y)
    optimizer = tr.make_optimizer(model)
    loss_acc = 0.0
    loss_steps = 0
    for step in range(tr.MAX_ITERS):
        loss = tr.train_step(model, optimizer, x, y)
        x, y = loader.next()
        loss_acc += float(loss.detach())
        loss_steps += 1
        if step % tr.LOG_FREQ == 0:
            print(f"Step: {step}/{tr.MAX_ITERS} loss {loss_acc / loss_steps:.3}")
            loss_acc = 0.0
            loss_steps = 0
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=tr.device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)


if __name__ == "__main__":
    if "BDH_MAX_ITERS" in os.environ:
        tr.MAX_ITERS = int(os.environ["BDH_MAX_ITERS"])
    main()
