# Copyright Pathway Technology, Inc.
# Private opt: compile-friendly training loop (see OPT_NOTES.md).

from __future__ import annotations

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
_use_scaler = dtype == "float16" and device.type == "cuda"
scaler = torch.amp.GradScaler(device=device.type, enabled=_use_scaler)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

# Compile / opt knobs (env overrides for benches and CPU boxes without inductor deps)
USE_COMPILE = os.environ.get("BDH_COMPILE", "1") not in ("0", "false", "False")
COMPILE_MODE = os.environ.get("BDH_COMPILE_MODE", "default")  # default|reduce-overhead|max-autotune
USE_FUSED_ADAMW = os.environ.get("BDH_FUSED_ADAMW", "1") not in ("0", "false", "False")

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


# Cached train/val byte tensors (pre-tensorized once from memmap).
_train_data = None
_val_data = None
# Reusable index buffers to cut per-step tensor allocs (compile / allocator friendly).
_offsets = None


def _load_splits():
    """Load corpus once: train = first 90%, val = last 10% (same as before)."""
    global _train_data, _val_data, _offsets
    if _train_data is not None:
        return
    mm = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    n = len(mm)
    split = int(0.9 * n)
    # Pre-tensorize int64 once; tiny Shakespeare is ~1MB → ~8MB resident.
    _train_data = torch.from_numpy(np.asarray(mm[:split], dtype=np.int64))
    _val_data = torch.from_numpy(np.asarray(mm[split:], dtype=np.int64))
    _offsets = torch.arange(BLOCK_SIZE + 1)


def get_batch(split):
    # Vectorized window gather — no Python list of from_numpy per sample.
    global _offsets
    _load_splits()
    if _offsets is None or _offsets.numel() != BLOCK_SIZE + 1:
        _offsets = torch.arange(BLOCK_SIZE + 1)
    data = _train_data if split == "train" else _val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    windows = data[ix.unsqueeze(1) + _offsets.unsqueeze(0)]  # (B, BLOCK_SIZE+1)
    # contiguous: windows[:, 1:] has stride (T+1,1) and breaks targets.view(-1) in CE
    x = windows[:, :-1].contiguous()
    y = windows[:, 1:].contiguous()
    if device.type == "cuda":
        # pin → async H2D so the next step can overlap with compute when prefetched
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


class BatchPrefetcher:
    """One-slot prefetch: build/copy next batch while the current step runs.

    On CUDA, host→device copies use a side stream so they can overlap with
    backward/optimizer on the default stream. On CPU this still pipelines
    the (cheap) gather ahead of the next forward.
    """

    def __init__(self, split: str = "train"):
        self.split = split
        self._next = None
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None

    def preload(self):
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._next = get_batch(self.split)
        else:
            self._next = get_batch(self.split)

    def next(self):
        if self._next is None:
            self.preload()
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next
        self._next = None
        self.preload()  # kick next copy/gather immediately
        return batch


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with fused kernel when available (CUDA and recent CPU builds)."""
    kwargs = dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_FUSED_ADAMW:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (RuntimeError, ValueError) as e:
            print(f"fused AdamW unavailable ({e}); falling back")
    return torch.optim.AdamW(model.parameters(), **kwargs)


def maybe_compile(
    model: torch.nn.Module,
    example_x: torch.Tensor | None = None,
    example_y: torch.Tensor | None = None,
) -> torch.nn.Module:
    """torch.compile with graceful fallback when inductor/CXX is missing.

    Inductor errors often surface on the *first forward*, not at compile() time,
    so pass an example batch to probe when possible.

    Notes to avoid graph breaks in the *training loop* (model is what we compile):
    - Keep control flow / logging / get_batch outside the compiled module.
    - Prefer tensor ops inside BDH; avoid `.item()` / print inside forward.
    - Detach losses before Python accumulators so the graph is freed.
    - CUDA graphs (`mode='reduce-overhead'`) need static shapes + no CPU sync;
      see train_fast.py / OPT_NOTES.md. Not enabled by default on dynamic T.
    """
    if not USE_COMPILE:
        print("torch.compile disabled (BDH_COMPILE=0)")
        return model
    try:
        compiled = torch.compile(model, mode=COMPILE_MODE)
    except Exception as e:
        print(f"torch.compile failed ({e}); using eager")
        return model
    if example_x is not None:
        was_training = model.training
        try:
            compiled.eval()
            with torch.no_grad(), ctx:
                if example_y is not None:
                    compiled(example_x, example_y)
                else:
                    compiled(example_x)
            print(f"torch.compile enabled (mode={COMPILE_MODE})")
        except Exception as e:
            print(f"torch.compile probe failed ({e}); using eager")
            if was_training:
                model.train()
            return model
        if was_training:
            compiled.train()
        return compiled
    print(f"torch.compile enabled (mode={COMPILE_MODE}, unprobed)")
    return compiled


def eval(model):
    model.eval()


def train_step(model, optimizer, x, y):
    """Single optimize step — kept as a function for benches / future fullgraph."""
    with ctx:
        _logits, loss = model(x, y)
    if _use_scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    loader = BatchPrefetcher("train")
    x, y = loader.next()
    model = maybe_compile(model, example_x=x, example_y=y)
    optimizer = make_optimizer(model)

    loss_acc = 0.0
    loss_steps = 0
    for step in range(MAX_ITERS):
        loss = train_step(model, optimizer, x, y)
        x, y = loader.next()  # already prefetched during step on CUDA side-stream
        # Detach so we do not retain the autograd graph across iterations.
        loss_acc += float(loss.detach())
        loss_steps += 1
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc / loss_steps:.3}")
            loss_acc = 0.0
            loss_steps = 0
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    # generate() is dynamic-length; keep it eager-friendly (compiled model still OK).
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
