# Copyright Pathway Technology, Inc.
# Private opt (opt/amp-deepen): harden opt-in BDH_AMP_DTYPE; GradScaler fp16+CUDA only.
# Defaults stay fp32 / COMPILE=0 / eager. AMP throughput wins are GPU-only.

from __future__ import annotations

import os
import queue
import threading
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

# Optional AMP via BDH_AMP_DTYPE (default float32 / unset = no autocast).
# Values: float32|fp32|off|"" , bfloat16|bf16 , float16|fp16|half
# GradScaler is enabled ONLY for float16 + CUDA (bf16 needs no loss scaling).
# Throughput wins (Tensor Cores / reduced HBM) are GPU-only; CPU AMP is for
# correctness smoke and is often *slower* than fp32 (cast overhead).
# Optional BDH_AMP_FORWARD_ONLY=1: autocast wraps hot logits forward only; CE
# runs in fp32 outside autocast. Default 0 keeps model(x,y) fully under ctx.
_AMP_NAME_ALIASES = {
    "": "float32",
    "off": "float32",
    "fp32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "float16": "float16",
}
_PTDTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

dtype = "float32"
ptdtype = torch.float32
ctx = nullcontext()
_use_scaler = False
scaler = None  # set in configure_amp()
# Opt-in: autocast hot forward (logits) only; CE outside in fp32. Default off.
_amp_forward_only = False


def parse_amp_dtype(raw: str | None) -> str:
    """Normalize BDH_AMP_DTYPE string → float32|bfloat16|float16."""
    if raw is None:
        return "float32"
    key = raw.strip().lower()
    if key not in _AMP_NAME_ALIASES:
        raise ValueError(
            f"BDH_AMP_DTYPE must be float32|bfloat16|float16 (or aliases), got {raw!r}"
        )
    return _AMP_NAME_ALIASES[key]


def _cpu_autocast_smoke(pt_dtype: torch.dtype) -> bool:
    """True if a tiny CPU matmul under autocast(pt_dtype) succeeds."""
    try:
        a = torch.randn(2, 2)
        with torch.autocast(device_type="cpu", dtype=pt_dtype):
            _ = a @ a
        return True
    except Exception:
        return False


def cpu_bf16_available() -> bool:
    """True if CPU autocast(bfloat16) works (torch.cpu.is_bf16_supported or smoke)."""
    check = getattr(getattr(torch, "cpu", None), "is_bf16_supported", None)
    if callable(check):
        try:
            return bool(check())
        except Exception:
            pass
    return _cpu_autocast_smoke(torch.bfloat16)


def cpu_fp16_available() -> bool:
    """True if CPU autocast(float16) works (smoke matmul)."""
    return _cpu_autocast_smoke(torch.float16)


def amp_throughput_claim_device() -> str:
    """Honest claim surface: AMP train speedups are CUDA-only on this project.

    Returns 'cuda' when a CUDA device is present (Tensor Core / HBM path), else
    'none' — CPU AMP is correctness/smoke, not a throughput claim.
    """
    return "cuda" if device.type == "cuda" else "none"


def configure_amp(amp_name: str | None = None, *, forward_only: bool | None = None) -> str:
    """Set module-level dtype / autocast ctx / GradScaler from name or env.

    Returns the resolved dtype name (float32|bfloat16|float16).
    GradScaler: enabled only when dtype==float16 and device is CUDA.
    CPU: bf16/fp16 enable autocast when requested (smoke / parity); no scaler.

    forward_only: if True, train_step autocasts logits-only forward and computes
    CE in fp32. None → read BDH_AMP_FORWARD_ONLY (default 0 / off).
    """
    global dtype, ptdtype, ctx, _use_scaler, scaler, _amp_forward_only
    if amp_name is None:
        amp_name = os.environ.get("BDH_AMP_DTYPE", "float32")
    dtype = parse_amp_dtype(amp_name)
    ptdtype = _PTDTYPE[dtype]
    if forward_only is None:
        forward_only = os.environ.get("BDH_AMP_FORWARD_ONLY", "0") in (
            "1",
            "true",
            "True",
        )
    _amp_forward_only = bool(forward_only) and dtype != "float32"
    if dtype == "float32":
        ctx = nullcontext()
        _use_scaler = False
        _amp_forward_only = False
    else:
        # Prefer bdh helper (CPU/CUDA/MPS); keeps generate AMP consistent.
        ctx = bdh._autocast_context(device, ptdtype)
        _use_scaler = dtype == "float16" and device.type == "cuda"
        if dtype == "bfloat16" and device.type == "cpu" and not cpu_bf16_available():
            raise RuntimeError(
                "BDH_AMP_DTYPE=bfloat16 requested but CPU bf16 autocast unavailable"
            )
        if dtype == "float16" and device.type == "cpu" and not cpu_fp16_available():
            raise RuntimeError(
                "BDH_AMP_DTYPE=float16 requested but CPU fp16 autocast unavailable"
            )
    # GradScaler device arg is the amp device type; keep constructed even when
    # disabled so train_step branches stay simple. Never enable for bf16 or CPU.
    scaler = torch.amp.GradScaler(device=device.type, enabled=_use_scaler)
    return dtype


configure_amp()
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# Quiet at import for pytest; __main__ / train_fast print explicitly.


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

# Compile / opt knobs (env overrides for benches and CPU boxes without inductor deps)
USE_COMPILE = os.environ.get("BDH_COMPILE", "0") in ("1", "true", "True")
COMPILE_MODE = os.environ.get("BDH_COMPILE_MODE", "default")  # default|reduce-overhead|max-autotune
# Probe depth when BDH_COMPILE=1: eval | train | train_bwd (default train — matches train loop)
COMPILE_PROBE = os.environ.get("BDH_COMPILE_PROBE", "train").strip().lower()
COMPILE_FULLGRAPH = os.environ.get("BDH_COMPILE_FULLGRAPH", "0") in ("1", "true", "True")
USE_FUSED_ADAMW = os.environ.get("BDH_FUSED_ADAMW", "1") not in ("0", "false", "False")
# Data path: default = vectorized get_batch + BatchPrefetcher; optional torch DataLoader
USE_DATALOADER = os.environ.get("BDH_DATALOADER", "0") in ("1", "true", "True")
NUM_WORKERS = int(os.environ.get("BDH_NUM_WORKERS", "2" if USE_DATALOADER else "0"))
# Async host-thread double-buffer for BatchPrefetcher (overlap gather with train_step).
# Set BDH_PREFETCH_ASYNC=0 to force synchronous preload (debug / A/B).
USE_PREFETCH_ASYNC = os.environ.get("BDH_PREFETCH_ASYNC", "1") not in ("0", "false", "False")

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
    # Page-lock corpus on CUDA so gather source is pinned (batch still pinned below).
    if device.type == "cuda":
        _train_data = _train_data.pin_memory()
        _val_data = _val_data.pin_memory()
    _offsets = torch.arange(BLOCK_SIZE + 1)


def _gather_batch_host(split: str):
    """Vectorized window gather on CPU — no Python list of from_numpy per sample.

    Returns contiguous host tensors (B, T). Does not touch the device or call
    .item() / .cpu() on CUDA tensors (no host sync).
    """
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
    return x, y


def _to_train_device(x: torch.Tensor, y: torch.Tensor):
    """H2D with pin_memory + non_blocking on CUDA; plain .to on CPU."""
    if device.type == "cuda":
        # pin → async H2D so prefetch / DataLoader can overlap with compute
        if not x.is_pinned():
            x = x.pin_memory()
        if not y.is_pinned():
            y = y.pin_memory()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def _gather_batch_host_numpy(split: str):
    """Producer-thread gather via numpy — avoids OpenMP steal from train_step.

    Same window semantics as ``_gather_batch_host`` (contiguous LM x/y), but
    uses ``np.random`` + numpy indexing so the background thread does not
    contend on PyTorch's intra-op threadpool. Not seed-compatible with
    ``torch.manual_seed`` (prefetch samples are still valid IID).
    """
    _load_splits()
    data_t = _train_data if split == "train" else _val_data
    assert data_t is not None
    data = data_t.numpy() if data_t.device.type == "cpu" else data_t.cpu().numpy()
    n = len(data) - BLOCK_SIZE
    ix = np.random.randint(0, n, size=BATCH_SIZE)
    offsets = np.arange(BLOCK_SIZE + 1)
    windows = data[ix[:, None] + offsets[None, :]]
    x = torch.from_numpy(np.ascontiguousarray(windows[:, :-1], dtype=np.int64))
    y = torch.from_numpy(np.ascontiguousarray(windows[:, 1:], dtype=np.int64))
    return x, y


def get_batch(split):
    """Vectorized get_batch: host gather + pin/non_blocking H2D when CUDA."""
    x, y = _gather_batch_host(split)
    return _to_train_device(x, y)


def _dataloader_worker_init(worker_id: int) -> None:
    # Distinct RNG streams per worker (DataLoader shares base seed otherwise).
    base = torch.initial_seed() % (2**32)
    torch.manual_seed(base + worker_id)


class _HostBatchIterable(torch.utils.data.IterableDataset):
    """Yields full (x, y) host batches so collate stays vectorized (batch_size=None)."""

    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split

    def __iter__(self):
        while True:
            yield _gather_batch_host(self.split)


def make_torch_dataloader(split: str = "train") -> torch.utils.data.DataLoader:
    """Optional DataLoader around vectorized host gather.

    Env: BDH_DATALOADER=1, BDH_NUM_WORKERS=N (persistent_workers when N>0).
    pin_memory enabled on CUDA so callers can .to(..., non_blocking=True).
    """
    n_workers = max(0, NUM_WORKERS)
    pin = device.type == "cuda"
    kwargs = dict(
        dataset=_HostBatchIterable(split),
        batch_size=None,  # dataset already yields a full batch (keeps vectorized gather)
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=(n_workers > 0),
    )
    if n_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = _dataloader_worker_init
    return torch.utils.data.DataLoader(**kwargs)


class BatchPrefetcher:
    """One-slot double-buffer: next host gather (+ H2D) overlaps train_step.

    Hot-loop contract (see ``__main__``)::

        loss = train_step(model, opt, x, y)
        x, y = loader.next()  # wait for ready slot; producer already refilling

    Async mode (default, ``BDH_PREFETCH_ASYNC=1``): a daemon host thread keeps
    one pinned host batch in a ``queue.Queue(maxsize=1)``. While ``train_step``
    runs, the producer refills the slot. ``next()`` only blocks if the step
    finished before the gather (rare when step ≫ gather).

    CUDA: producer does gather + ``pin_memory``; caller enqueues H2D on a
    side stream and waits on the recorded event before returning tensors.
    CPU: producer does gather; caller ``.to(device)`` (no-op copy elision).

    Sync mode (``BDH_PREFETCH_ASYNC=0`` or ``async_host=False``): same one-slot
    API but preload runs on the caller thread (A/B / debug).
    """

    def __init__(self, split: str = "train", *, async_host: bool | None = None):
        self.split = split
        self._async = USE_PREFETCH_ASYNC if async_host is None else bool(async_host)
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._host_next: tuple[torch.Tensor, torch.Tensor] | None = None
        self._error: BaseException | None = None
        if self._async:
            self._q = queue.Queue(maxsize=1)
            self._thread = threading.Thread(
                target=self._producer_loop,
                name="bdh-prefetch",
                daemon=True,
            )
            self._thread.start()
        else:
            self.preload()

    def _gather_pinned_host(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Host gather for the producer thread.

        Uses a numpy advanced-index path so the background thread does not
        launch PyTorch OpenMP work that steals cores from ``train_step``.
        Public ``get_batch`` / ``_gather_batch_host`` stay on the torch path
        (seed-stable for tests). Training sample streams remain valid IID.
        """
        x, y = _gather_batch_host_numpy(self.split)
        if device.type == "cuda":
            if not x.is_pinned():
                x = x.pin_memory()
            if not y.is_pinned():
                y = y.pin_memory()
        return x, y

    def _producer_loop(self) -> None:
        assert self._q is not None
        while not self._stop.is_set():
            try:
                batch = self._gather_pinned_host()
            except BaseException as e:  # noqa: BLE001 — surface on next()
                self._error = e
                break
            # Block until consumer takes the slot or stop is requested.
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.05)
                    break
                except queue.Full:
                    continue

    def _to_device(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Caller-thread device transfer (CUDA side-stream when available)."""
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                xd = x.to(device, non_blocking=True)
                yd = y.to(device, non_blocking=True)
                ev = self._stream.record_event()
            torch.cuda.current_stream().wait_event(ev)
            return xd, yd
        return _to_train_device(x, y)

    def preload(self) -> None:
        """Sync path only: fill the one host slot on the caller thread."""
        if self._async:
            return
        self._host_next = self._gather_pinned_host()

    def _take_host(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._async:
            assert self._q is not None
            while True:
                if self._error is not None:
                    raise RuntimeError("prefetch producer failed") from self._error
                try:
                    return self._q.get(timeout=0.05)
                except queue.Empty:
                    if self._thread is not None and not self._thread.is_alive():
                        if self._error is not None:
                            raise RuntimeError(
                                "prefetch producer failed"
                            ) from self._error
                        raise RuntimeError("prefetch producer thread exited")
                    continue
        if self._host_next is None:
            self.preload()
        host = self._host_next
        self._host_next = None
        assert host is not None
        return host

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        host = self._take_host()
        if not self._async:
            # Sync: refill before return (same as historical one-slot behavior).
            self.preload()
        # Async: producer already blocked on the empty slot and will refill
        # while the caller runs train_step — no kick needed here.
        return self._to_device(*host)

    def close(self) -> None:
        """Stop producer thread / clear sync slot (tests / shutdown)."""
        self._stop.set()
        if self._q is not None:
            # Unblock a producer stuck on Full / consumer waiting.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._q = None
        self._host_next = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DataLoaderBatchSource:
    """Adapter with the same .next() API as BatchPrefetcher for the train loop."""

    def __init__(self, split: str = "train"):
        self.split = split
        self._loader = make_torch_dataloader(split)
        self._it = iter(self._loader)

    def next(self):
        x, y = next(self._it)
        # DataLoader already pin_memory'd on CUDA; non_blocking H2D, no .item()
        return _to_train_device(x, y)


def make_batch_source(split: str = "train"):
    """Default: BatchPrefetcher. Set BDH_DATALOADER=1 for DataLoader + workers."""
    if USE_DATALOADER:
        return DataLoaderBatchSource(split)
    return BatchPrefetcher(split)


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

    Inductor errors often surface on the *first forward* (or first *train*
    forward), not at ``torch.compile()`` time — pass an example batch to probe.

    Probe depth (``BDH_COMPILE_PROBE``):
      - ``eval``      — ``eval()`` + ``no_grad`` forward (old default)
      - ``train``     — ``train()`` forward with targets (default; matches loop)
      - ``train_bwd`` — train forward + ``loss.backward()`` then zero grads

    Eval-only probes leave a *separate* Dynamo graph for training (dropout /
    ``is_grad_enabled`` guards differ). Prefer ``train`` or ``train_bwd``.

    Graph-break boundaries (documented in OPT_NOTES.md ``opt/compile-harden``):
    - Compile **only the module**; keep get_batch / logging / ``.item()`` outside.
    - ``generate()`` is dynamic-length and marked ``torch.compiler.disable``.
    - RoPE table cache skips Python attribute writes while Dynamo is tracing.
    - ``BDH_ATTN_IMPL`` / env are specialized at compile time (change → recompile).
    - On CPU, recommend ``BDH_COMPILE=1`` only with ``BDH_ATTN_IMPL=eager``
      (#46); ``blocked``/``online``/``triton`` log a warning (measured regression).
    - CUDA graphs (``mode='reduce-overhead'``) need a real GPU + static shapes;
      on CPU this mode is accepted but does **not** claim graph capture wins.
    """
    if not USE_COMPILE:
        print("torch.compile disabled (set BDH_COMPILE=1 to enable)")
        return model

    # CPU matrix (#46): COMPILE=1 + eager ~5.25 ms vs ~7.76 eager baseline;
    # COMPILE=1 + blocked ~70–100× slower. Warn only — do not change defaults.
    _attn_raw = os.environ.get("BDH_ATTN_IMPL", "eager").strip().lower()
    if _attn_raw in ("blocked", "online", "triton"):
        print(
            f"torch.compile warning: BDH_ATTN_IMPL={_attn_raw} with BDH_COMPILE=1 "
            "is a measured CPU train-step regression vs eager compile "
            "(#46: blocked ~70–100× slower than COMPILE=1+eager). "
            "Prefer BDH_ATTN_IMPL=eager when compiling on CPU; GPU still open."
        )

    probe = COMPILE_PROBE if COMPILE_PROBE in ("eval", "train", "train_bwd") else "train"
    device_tag = f"{device.type}" + (
        f":{device.index}" if getattr(device, "index", None) is not None else ""
    )
    if COMPILE_MODE == "reduce-overhead" and device.type != "cuda":
        print(
            f"torch.compile note: mode=reduce-overhead on {device_tag} "
            "(no CUDA graphs on this device; inductor still may run)"
        )

    compile_kwargs = dict(mode=COMPILE_MODE)
    if COMPILE_FULLGRAPH:
        compile_kwargs["fullgraph"] = True

    try:
        compiled = torch.compile(model, **compile_kwargs)
    except Exception as e:
        print(
            f"torch.compile failed ({type(e).__name__}: {e}); using eager "
            f"[device={device_tag} mode={COMPILE_MODE}]"
        )
        return model

    if example_x is None:
        print(
            f"torch.compile enabled (mode={COMPILE_MODE}, fullgraph={COMPILE_FULLGRAPH}, "
            f"device={device_tag}, unprobed)"
        )
        return compiled

    was_training = model.training
    try:
        if probe == "eval":
            compiled.eval()
            with torch.no_grad(), ctx:
                if example_y is not None:
                    compiled(example_x, example_y)
                else:
                    compiled(example_x)
        else:
            # Warm the *training* graph (same guards as train_step).
            compiled.train()
            with ctx:
                _logits, loss = (
                    compiled(example_x, example_y)
                    if example_y is not None
                    else compiled(example_x)
                )
            if probe == "train_bwd":
                if loss is None:
                    raise RuntimeError(
                        "BDH_COMPILE_PROBE=train_bwd requires example_y (loss)"
                    )
                loss.backward()
                # Drop probe grads so the first real step starts clean.
                compiled.zero_grad(set_to_none=True)
        print(
            f"torch.compile enabled (mode={COMPILE_MODE}, probe={probe}, "
            f"fullgraph={COMPILE_FULLGRAPH}, device={device_tag})"
        )
    except Exception as e:
        print(
            f"torch.compile probe failed ({type(e).__name__}: {e}); using eager "
            f"[probe={probe} device={device_tag} mode={COMPILE_MODE}]"
        )
        if was_training:
            model.train()
        else:
            model.eval()
        return model

    if was_training:
        compiled.train()
    else:
        compiled.eval()
    return compiled


def eval(model):
    model.eval()


def train_step(model, optimizer, x, y):
    """Single optimize step — kept as a function for benches / future fullgraph.

    AMP: module ``ctx`` wraps the hot forward. With ``_amp_forward_only``
    (BDH_AMP_FORWARD_ONLY=1), only logits are under autocast; CE is fp32.
    Default keeps ``model(x, y)`` (logits + CE) under the same ctx.
    GradScaler still only for float16+CUDA.
    """
    if _amp_forward_only:
        # Hot body under autocast; CE outside in fp32 for numerical honesty.
        with ctx:
            logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
    else:
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
    print(
        f"Using device: {device} amp_dtype={dtype} "
        f"scaler={_use_scaler} forward_only={_amp_forward_only} "
        f"amp_claim={amp_throughput_claim_device()} (BDH_AMP_DTYPE)"
    )
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    loader = make_batch_source("train")
    x, y = loader.next()
    model = maybe_compile(model, example_x=x, example_y=y)
    optimizer = make_optimizer(model)

    # Detach + accumulate on-device; .item() only at LOG_FREQ (avoids per-step CUDA sync).
    loss_acc = None
    loss_steps = 0
    for step in range(MAX_ITERS):
        loss = train_step(model, optimizer, x, y)
        x, y = loader.next()  # host gather overlapped previous train_step; CUDA H2D on side-stream
        det = loss.detach()
        loss_acc = det if loss_acc is None else (loss_acc + det)
        loss_steps += 1
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = None
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
