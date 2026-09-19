# Copyright Pathway Technology, Inc.
# Private opt (opt/log-sync): cut train logging host sync further (see OPT_NOTES.md).
# Defaults stay fp32 / COMPILE=0 / eager / LOG_FREQ=100. tril(diagonal=-1) untouched.

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
    """True only when CPU bf16 is advertised *and* a real autocast smoke works."""
    check = getattr(getattr(torch, "cpu", None), "is_bf16_supported", None)
    if callable(check):
        try:
            if not bool(check()):
                return False
        except Exception:
            pass
    # The version probe is only a hint; the matmul catches runtime/backend gaps.
    return _cpu_autocast_smoke(torch.bfloat16)


def cpu_fp16_available() -> bool:
    """True only when a CPU float16 autocast matmul succeeds."""
    return _cpu_autocast_smoke(torch.float16)


def _cpu_amp_unavailable_message(dtype_name: str) -> str:
    return (
        f"BDH_AMP_DTYPE={dtype_name} requested on CPU, but {dtype_name} "
        "autocast is unavailable; use float32 or a CPU build with AMP support"
    )


def _grad_scaler_allowed(dtype_name: str) -> bool:
    """Keep loss scaling strictly on CUDA float16 with a live CUDA runtime."""
    return (
        dtype_name == "float16"
        and device.type == "cuda"
        and torch.cuda.is_available()
    )


def amp_throughput_claim_device() -> str:
    """Honest claim surface: AMP train speedups are CUDA-only on this project.

    Returns 'cuda' when a CUDA device is present (Tensor Core / HBM path), else
    'none' — CPU AMP is correctness/smoke, not a throughput claim.
    """
    return "cuda" if device.type == "cuda" and torch.cuda.is_available() else "none"


def configure_amp(amp_name: str | None = None, *, forward_only: bool | None = None) -> str:
    """Set module-level dtype / autocast ctx / GradScaler from name or env.

    Returns the resolved dtype name (float32|bfloat16|float16).
    GradScaler is enabled only for float16 on a live CUDA device. CPU bf16/fp16
    use autocast only when a backend smoke succeeds; an unavailable request
    raises a clear error without partially changing the previous AMP state.

    forward_only: if True, train_step autocasts logits-only forward and computes
    CE in fp32. None → read BDH_AMP_FORWARD_ONLY (default 0 / off).
    """
    global dtype, ptdtype, ctx, _use_scaler, scaler, _amp_forward_only
    if amp_name is None:
        amp_name = os.environ.get("BDH_AMP_DTYPE", "float32")
    resolved = parse_amp_dtype(amp_name)
    resolved_ptdtype = _PTDTYPE[resolved]
    if forward_only is None:
        forward_only = os.environ.get("BDH_AMP_FORWARD_ONLY", "0") in (
            "1",
            "true",
            "True",
        )
    resolved_forward_only = bool(forward_only) and resolved != "float32"

    # Validate CPU support before touching module state. This makes a failed
    # opt-in safe for test/bench callers that want to fall back to fp32.
    if resolved == "bfloat16" and device.type == "cpu" and not cpu_bf16_available():
        raise RuntimeError(_cpu_amp_unavailable_message(resolved))
    if resolved == "float16" and device.type == "cpu" and not cpu_fp16_available():
        raise RuntimeError(_cpu_amp_unavailable_message(resolved))

    if resolved == "float32":
        resolved_ctx = nullcontext()
        use_scaler = False
        resolved_forward_only = False
    else:
        # Prefer bdh helper (CPU/CUDA/MPS); keeps generate AMP consistent.
        resolved_ctx = bdh._autocast_context(device, resolved_ptdtype)
        use_scaler = _grad_scaler_allowed(resolved)

    # Use a known-supported disabled scaler device on non-CUDA backends. The
    # scaler object remains present for callers/tests, but can never activate.
    resolved_scaler = torch.amp.GradScaler(
        device="cuda" if use_scaler else "cpu",
        enabled=use_scaler,
    )

    # Commit all state only after validation and construction have succeeded.
    dtype = resolved
    ptdtype = resolved_ptdtype
    ctx = resolved_ctx
    _use_scaler = use_scaler
    scaler = resolved_scaler
    _amp_forward_only = resolved_forward_only
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
LOG_FREQ = int(os.environ.get("BDH_LOG_FREQ", "100"))
# CUDA: defer .item() one LOG_FREQ window via non_blocking D2H (default on).
# Set BDH_LOG_ASYNC=0 to .item() at the log boundary (train-fuse timing).
USE_LOG_ASYNC = os.environ.get("BDH_LOG_ASYNC", "1") not in ("0", "false", "False")

# Compile / opt knobs (env overrides for benches and CPU boxes without inductor deps)
USE_COMPILE = os.environ.get("BDH_COMPILE", "0") in ("1", "true", "True")
COMPILE_MODE = os.environ.get("BDH_COMPILE_MODE", "default")  # default|reduce-overhead|max-autotune
# Probe depth when BDH_COMPILE=1: eval | train | train_bwd.
# The default includes backward because this is a train-step path; use
# BDH_COMPILE_PROBE=train for a forward-only diagnostic when startup cost matters.
COMPILE_PROBE = os.environ.get("BDH_COMPILE_PROBE", "train_bwd").strip().lower()
COMPILE_FULLGRAPH = os.environ.get("BDH_COMPILE_FULLGRAPH", "0") in ("1", "true", "True")
USE_FUSED_ADAMW = os.environ.get("BDH_FUSED_ADAMW", "1") not in ("0", "false", "False")
# Data path: default = vectorized get_batch + BatchPrefetcher; optional torch DataLoader
USE_DATALOADER = os.environ.get("BDH_DATALOADER", "0") in ("1", "true", "True")
NUM_WORKERS = int(os.environ.get("BDH_NUM_WORKERS", "2" if USE_DATALOADER else "0"))
# Async host-thread double-buffer for BatchPrefetcher (overlap gather with train_step).
# Set BDH_PREFETCH_ASYNC=0 to force synchronous preload (debug / A/B).
USE_PREFETCH_ASYNC = os.environ.get("BDH_PREFETCH_ASYNC", "1") not in ("0", "false", "False")
# CUDA side-stream H2D staging is opt-out. It is always disabled on CPU, so the
# flag is harmless in CPU-only jobs and keeps the historical CPU path unchanged.
# Set BDH_PREFETCH_H2D=0 to keep host prefetch but issue H2D on the caller stream.
USE_PREFETCH_H2D = os.environ.get("BDH_PREFETCH_H2D", "1") not in ("0", "false", "False")


def prefetch_h2d_skip_reason() -> str | None:
    """Return the CPU-safe reason why CUDA H2D staging is unavailable.

    ``None`` means the requested device is CUDA and the CUDA runtime is
    available. This probe intentionally performs no allocation, stream, or
    event construction, so CPU tests can inspect the gate safely.
    """
    if device.type != "cuda":
        return f"device-not-cuda: {device.type}"
    try:
        cuda_available = torch.cuda.is_available()
    except (AssertionError, RuntimeError) as exc:
        return (
            "CUDA unavailable: torch.cuda.is_available() raised "
            f"{type(exc).__name__}"
        )
    if not cuda_available:
        return "CUDA unavailable: torch.cuda.is_available() is false"
    return None


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

    CUDA: producer does gather + ``pin_memory``; with ``BDH_PREFETCH_H2D=1``
    the async path keeps one device batch staged ahead on a side stream and
    waits on a recorded event only when that batch is consumed. Set the flag to
    ``0`` to retain host prefetch but use the caller stream for H2D.
    CPU: producer does gather; caller ``.to(device)`` is an identity transfer.
    The H2D opt-in is deliberately a no-op here: no CUDA stream/event or
    staged device tuple is created, and the CPU tensors stay on the CPU.

    Sync mode (``BDH_PREFETCH_ASYNC=0`` or ``async_host=False``): same one-slot
    API but preload runs on the caller thread (A/B / debug).
    """

    def __init__(
        self,
        split: str = "train",
        *,
        async_host: bool | None = None,
        cuda_staging: bool | None = None,
    ):
        """Create a host prefetcher with optional CUDA H2D lookahead.

        ``cuda_staging`` overrides ``BDH_PREFETCH_H2D`` for tests/A-B runs.
        It is ignored on non-CUDA devices.
        """
        self.split = split
        self._async = USE_PREFETCH_ASYNC if async_host is None else bool(async_host)
        requested_cuda_staging = (
            USE_PREFETCH_H2D if cuda_staging is None else bool(cuda_staging)
        )
        # Gate the opt-in before constructing any CUDA object. On CPU, or
        # when a CUDA device is requested without a live CUDA runtime, this
        # keeps the stream/event path and device lookahead as clean no-ops.
        self._cuda_staging = bool(
            requested_cuda_staging and prefetch_h2d_skip_reason() is None
        )
        self._stream = (
            torch.cuda.Stream(device=device) if self._cuda_staging else None
        )
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._host_next: tuple[torch.Tensor, torch.Tensor] | None = None
        # A staged tuple retains its pinned source tensors until its H2D event
        # completes. record_stream below additionally hands lifetime tracking to
        # CUDA's allocator for the asynchronous copy.
        self._device_next: tuple[
            torch.Tensor, torch.Tensor, torch.cuda.Event, torch.Tensor, torch.Tensor
        ] | None = None
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

    def _stage_to_device(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.cuda.Event, torch.Tensor, torch.Tensor
    ]:
        """Queue one pinned host batch on the dedicated CUDA copy stream.

        The returned tuple owns the pinned source tensors until its event has
        been waited on; ``_device_next`` retains that tuple for the staged
        lookahead, while ``_wait_staged`` consumes it for the caller.
        """
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            xd = x.to(device, non_blocking=True)
            yd = y.to(device, non_blocking=True)
            # Keep the host storage alive until the copy stream is done. This
            # matters because the producer queue can hand off its Python refs
            # immediately after staging.
            x.record_stream(self._stream)
            y.record_stream(self._stream)
            ev = self._stream.record_event()
        return xd, yd, ev, x, y

    @staticmethod
    def _wait_staged(
        staged: tuple[
            torch.Tensor, torch.Tensor, torch.cuda.Event, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make a staged batch visible to the caller's current CUDA stream."""
        xd, yd, ev, _x, _y = staged
        torch.cuda.current_stream(device=device).wait_event(ev)
        return xd, yd

    def _to_device(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Caller-thread transfer; CUDA staging is opt-in and CPU is identity.

        Keeping the CPU branch on ``_to_train_device`` is intentional: its
        same-device ``.to`` calls preserve tensor identity, so enabling the
        CUDA flag cannot add a CPU copy or synchronization point.
        """
        if self._cuda_staging:
            return self._wait_staged(self._stage_to_device(x, y))
        return _to_train_device(x, y)

    def _next_cuda_staged(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one device batch while staging the following batch.

        The first call primes two H2D copies. On later calls, the copy stream
        can issue batch N+1 while the caller's current stream computes on batch
        N, then the current stream waits only for the already-queued event.
        """
        ready = self._device_next
        if ready is None:
            ready = self._stage_to_device(*self._take_host())
        self._device_next = None
        try:
            # Keep one device batch ahead. The host producer is already filling
            # the queue while train_step runs, so this should not add a CPU
            # stall beyond the normal prefetch handoff.
            self._device_next = self._stage_to_device(*self._take_host())
        except BaseException:
            # Do not retain a stale staged batch after surfacing a producer
            # failure; the caller should see the failure rather than reuse it.
            self._device_next = None
            raise
        return self._wait_staged(ready)

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
        if self._cuda_staging and self._async:
            return self._next_cuda_staged()
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
        self._device_next = None

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

    def close(self) -> None:
        """Stop DataLoader workers and make shutdown explicit and idempotent."""
        iterator = self._it
        if iterator is None:
            return
        try:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        finally:
            self._it = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_batch_source(split: str = "train"):
    """Default: BatchPrefetcher. Set BDH_DATALOADER=1 for DataLoader + workers."""
    if USE_DATALOADER:
        return DataLoaderBatchSource(split)
    return BatchPrefetcher(split)


def clear_grads(module_or_optimizer) -> None:
    """Single chokepoint: ``zero_grad(set_to_none=True)``.

    Frees grad storage instead of filling zeros — friendlier to the allocator
    and to CUDA-graph capture. Train path must not call ``zero_grad()`` with
    the default ``set_to_none=False``. Accepts ``nn.Module`` or ``Optimizer``.
    """
    module_or_optimizer.zero_grad(set_to_none=True)


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW with fused kernel when available (CUDA and recent CPU builds).

    Default ``BDH_FUSED_ADAMW=1`` tries ``fused=True`` then falls back.
    Pair with ``clear_grads`` / ``train_step`` (``set_to_none``) for the
    allocator-friendly train path. Defaults unchanged when env is unset.
    """
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
      - ``eval``      — ``eval()`` + ``no_grad`` forward (diagnostic only)
      - ``train``     — ``train()`` forward with targets (forward-only diagnostic)
      - ``train_bwd`` — train forward + ``loss.backward()`` then zero grads
        (default; validates the compiled portion of ``train_step``)

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
      on CPU this mode is accepted but is **not useful** (no CUDA graphs) —
      prefer ``default``; soft-fallback still applies if compile/probe fails.

    ``BDH_COMPILE_FULLGRAPH=1`` applies to the module probe, not the Python
    optimizer step. With ``BDH_ATTN_AUTOGRAD=1`` and ``train_bwd`` the probe
    also exercises the analytic attention backward path; a graph-break or
    unsupported backward still soft-falls back to the original eager module.
    The optimizer step and gradient clearing remain eager in either case.
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

    probe = (
        COMPILE_PROBE
        if COMPILE_PROBE in ("eval", "train", "train_bwd")
        else "train_bwd"
    )
    device_tag = f"{device.type}" + (
        f":{device.index}" if getattr(device, "index", None) is not None else ""
    )
    if COMPILE_MODE == "reduce-overhead" and device.type != "cuda":
        print(
            f"torch.compile warning: mode=reduce-overhead on {device_tag} "
            "is not useful without CUDA graphs (CPU has none). "
            "Prefer BDH_COMPILE_MODE=default on CPU; try reduce-overhead on a real GPU."
        )

    compile_kwargs = dict(mode=COMPILE_MODE)
    if COMPILE_FULLGRAPH:
        compile_kwargs["fullgraph"] = True

    try:
        compiled = torch.compile(model, **compile_kwargs)
    except Exception as e:
        print(
            f"torch.compile failed ({type(e).__name__}: {e}); "
            "soft-fallback to original eager module "
            f"[device={device_tag} mode={COMPILE_MODE} probe={probe} "
            f"fullgraph={COMPILE_FULLGRAPH}]"
        )
        return model

    if example_x is None:
        print(
            f"torch.compile enabled without a first probe (mode={COMPILE_MODE}, "
            f"fullgraph={COMPILE_FULLGRAPH}, device={device_tag}, probe={probe}); "
            "the first real call may still fail. Pass example_x and, for "
            "train_bwd, example_y to maybe_compile for soft-fallback coverage."
        )
        return compiled

    if probe == "train_bwd" and example_y is None:
        # Do not return an apparently train-ready OptimizedModule when the
        # selected probe cannot exercise backward. Returning the original
        # module is a soft fallback and avoids a first-step surprise.
        print(
            "torch.compile probe skipped: requested backward probe "
            "probe=train_bwd requires example_y; no backward probe was "
            "attempted; "
            "pass example_y to maybe_compile or set BDH_COMPILE_PROBE=train "
            "for a forward-only probe; soft-fallback to original eager module "
            f"[device={device_tag} mode={COMPILE_MODE} probe={probe} "
            f"fullgraph={COMPILE_FULLGRAPH}]"
        )
        return model

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
                clear_grads(compiled)
        print(
            f"torch.compile enabled (mode={COMPILE_MODE}, probe={probe}, "
            f"fullgraph={COMPILE_FULLGRAPH}, device={device_tag})"
        )
    except Exception as e:
        print(
            f"torch.compile first probe failed ({type(e).__name__}: {e}); "
            "the compiled wrapper is discarded and the original eager module "
            "is retained; check the probe inputs and backend before retrying "
            f"[probe={probe} device={device_tag} mode={COMPILE_MODE} "
            f"fullgraph={COMPILE_FULLGRAPH}]"
            + (
                " — FULLGRAPH=1 requires a single Dynamo graph (graph breaks "
                "unsupported); soft-fallback to eager"
                if COMPILE_FULLGRAPH
                else ""
            )
        )
        if probe == "train_bwd":
            # A failed forward/backward probe may have left partial gradients
            # on the original parameters. Keep the eager fallback clean just
            # like the successful probe path above.
            try:
                clear_grads(model)
            except Exception as cleanup_error:
                print(
                    "torch.compile probe gradient cleanup failed "
                    f"({type(cleanup_error).__name__}: {cleanup_error})"
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



class TrainLossLogger:
    """On-device loss accumulate; host sync only at LOG_FREQ (CUDA: deferred).

    ``async_cuda`` is a CUDA-only request: on CPU it resolves to a no-op, so
    logging stays synchronous and does not allocate pinned host storage or CUDA
    events.

    Each step: ``loss.detach().float()`` into an on-device fp32 running sum
    (in-place ``add_`` — no per-step ``.item()``). At ``LOG_FREQ`` boundaries
    (and on ``close()``):

    * **CUDA + async (default):** ``non_blocking`` copy into a pinned host
      scalar + record event; ``.item()`` / print run on the *next* boundary
      (or ``close``), so the sync overlaps the following train steps.
    * **CPU or ``BDH_LOG_ASYNC=0``:** print immediately (CPU has no device sync
      to hide; sync mode matches train-fuse timing).

    Printed value = mean loss over the window since the previous boundary
    (same semantics as ``opt/train-fuse``). Does not touch attention / tril.
    """

    def __init__(
        self,
        log_freq: int | None = None,
        *,
        device: torch.device | None = None,
        async_cuda: bool | None = None,
        max_iters: int | None = None,
    ):
        self.log_freq = int(LOG_FREQ if log_freq is None else log_freq)
        if self.log_freq < 1:
            raise ValueError(f"log_freq must be >= 1, got {self.log_freq}")
        self.device = device if device is not None else globals()["device"]
        self.async_cuda = USE_LOG_ASYNC if async_cuda is None else bool(async_cuda)
        self.max_iters = max_iters
        self._acc: torch.Tensor | None = None
        self._steps = 0
        # Resolve the request once. CPU (and CUDA-unavailable environments)
        # deliberately take the ordinary synchronous path even when the
        # default/requested async flag is true.
        self._use_cuda_async = (
            self.async_cuda
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._host: torch.Tensor | None = None
        self._event = None  # torch.cuda.Event when pending
        self._pending_step: int | None = None
        self._closed = False

    @property
    def uses_deferred_cuda(self) -> bool:
        """Whether the deferred CUDA D2H path is active for this logger."""
        return self._use_cuda_async

    def _ensure_host(self) -> torch.Tensor:
        if self._host is None:
            self._host = torch.empty((), dtype=torch.float32, pin_memory=True)
        return self._host

    def _flush_pending(self) -> None:
        """Sync previous deferred D2H (if any) and print."""
        if self._pending_step is None:
            return
        assert self._host is not None
        if self._event is not None:
            self._event.synchronize()
            self._event = None
        mean = float(self._host)
        total = self.max_iters if self.max_iters is not None else "?"
        print(f"Step: {self._pending_step}/{total} loss {mean:.3}")
        self._pending_step = None

    def _schedule_or_print(self, step: int) -> None:
        assert self._acc is not None and self._steps > 0
        mean = self._acc / self._steps
        total = self.max_iters if self.max_iters is not None else "?"
        if self._use_cuda_async:
            # Flush prior window first (sync after ~LOG_FREQ more compute).
            self._flush_pending()
            host = self._ensure_host()
            host.copy_(mean.detach(), non_blocking=True)
            evt = torch.cuda.Event()
            evt.record()
            self._event = evt
            self._pending_step = step
        else:
            print(f"Step: {step}/{total} loss {mean.item():.3}")
        self._acc = None
        self._steps = 0

    def update(self, loss: torch.Tensor, step: int) -> None:
        """Record one step's loss; maybe schedule/print at LOG_FREQ."""
        if self._closed:
            raise RuntimeError("TrainLossLogger.update after close()")
        det = loss.detach().float()
        if self._acc is None:
            self._acc = det.clone()
        else:
            self._acc.add_(det)
        self._steps += 1
        if step % self.log_freq == 0:
            self._schedule_or_print(step)

    def close(self) -> None:
        """Flush any partial window + deferred pending print (end of training)."""
        if self._closed:
            return
        self._closed = True
        if self._steps > 0 and self._acc is not None:
            step = (
                (self.max_iters - 1)
                if self.max_iters is not None
                else (self._pending_step or 0) + self._steps
            )
            self._schedule_or_print(step)
        self._flush_pending()


def run_train_loop(model, optimizer, loader, x, y, *, max_iters: int | None = None):
    """Shared train loop for train.py / train_fast.py (sync-light logging).

    Caller supplies the first ``(x, y)`` (already used for compile probe).
    Prefetch of the *next* batch runs before any rare log sync so host gather
    / H2D still overlaps compute on the non-log path.
    """
    n = MAX_ITERS if max_iters is None else int(max_iters)
    logger = TrainLossLogger(max_iters=n)
    for step in range(n):
        loss = train_step(model, optimizer, x, y)
        x, y = loader.next()
        logger.update(loss, step)
    logger.close()



def eval(model):
    model.eval()


def train_step(model, optimizer, x, y):
    """Single optimize step — kept as a function for benches / future fullgraph.

    The module forward/backward may be a ``torch.compile`` region, but the
    optimizer step and grad clearing intentionally remain outside that region.
    Always ends with ``clear_grads(optimizer)`` → ``zero_grad(set_to_none=True)``
    so the next step does not pay fill-zero allocator traffic (and plays nicer
    with CUDA-graph capture). Grads are ``None`` after return — inspect before
    the clear if needed. Pair with ``make_optimizer`` (fused AdamW when available).

    AMP: module ``ctx`` wraps the hot forward. With ``_amp_forward_only``
    (BDH_AMP_FORWARD_ONLY=1), only logits are under autocast; CE is fp32.
    Default keeps ``model(x, y)`` (logits + CE) under the same ctx.
    GradScaler still only for float16+CUDA.
    """
    try:
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
        use_scaler = bool(_use_scaler and scaler is not None and scaler.is_enabled())
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return loss
    finally:
        clear_grads(optimizer)


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

    # Sync-light logging: on-device accumulate; CUDA D2H deferred across LOG_FREQ.
    run_train_loop(model, optimizer, loader, x, y, max_iters=MAX_ITERS)
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
