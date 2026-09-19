"""Train data path: vectorized get_batch, 90/10 split, optional DataLoader workers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tr(monkeypatch):
    """Fresh train module knobs; reset cached splits between tests."""
    import train as tr

    monkeypatch.setattr(tr, "input_file_path", str(ROOT / "input.txt"))
    monkeypatch.setattr(tr, "BLOCK_SIZE", 32)
    monkeypatch.setattr(tr, "BATCH_SIZE", 4)
    monkeypatch.setattr(tr, "device", torch.device("cpu"))
    monkeypatch.setattr(tr, "_train_data", None)
    monkeypatch.setattr(tr, "_val_data", None)
    monkeypatch.setattr(tr, "_offsets", None)
    monkeypatch.setattr(tr, "USE_PREFETCH_ASYNC", True)
    tr.fetch_data()
    return tr


def test_split_is_90_10(tr):
    tr._load_splits()
    mm = np.memmap(tr.input_file_path, dtype=np.uint8, mode="r")
    n = len(mm)
    split = int(0.9 * n)
    assert len(tr._train_data) == split
    assert len(tr._val_data) == n - split
    assert len(tr._train_data) + len(tr._val_data) == n


def test_vectorized_get_batch_matches_list_gather(tr):
    """Same seed → same windows as list/from_numpy baseline (loss math inputs)."""
    block, batch = tr.BLOCK_SIZE, tr.BATCH_SIZE
    data = np.memmap(tr.input_file_path, dtype=np.uint8, mode="r")
    data = data[: int(0.9 * len(data))]

    def baseline():
        ix = torch.randint(len(data) - block, (batch,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block]).astype(np.int64))
                for i in ix
            ]
        )
        return x, y

    torch.manual_seed(0)
    xb, yb = baseline()
    torch.manual_seed(0)
    xo, yo = tr.get_batch("train")
    assert xo.shape == (batch, block) and yo.shape == (batch, block)
    assert torch.equal(xb, xo) and torch.equal(yb, yo)


def test_gather_host_no_device_tensor(tr):
    x, y = tr._gather_batch_host("train")
    assert x.device.type == "cpu" and y.device.type == "cpu"
    assert x.is_contiguous() and y.is_contiguous()
    # shifted LM targets
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_to_train_device_cpu_passthrough(tr):
    x, y = tr._gather_batch_host("val")
    xd, yd = tr._to_train_device(x, y)
    assert xd.device.type == "cpu"
    assert torch.equal(x, xd) and torch.equal(y, yd)


@pytest.mark.parametrize("async_host", [True, False])
@pytest.mark.parametrize("cuda_staging", [None, False, True])
def test_batch_prefetcher_cuda_staging_is_cpu_noop(
    tr, monkeypatch, async_host, cuda_staging
):
    """Every H2D setting is a CPU identity path with no CUDA objects/lookahead."""

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU H2D path must not construct CUDA stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    loader = tr.BatchPrefetcher(
        "train", async_host=async_host, cuda_staging=cuda_staging
    )
    try:
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
        host_x, host_y = tr._gather_batch_host("val")
        device_x, device_y = loader._to_device(host_x, host_y)
        assert device_x is host_x and device_y is host_y
        x, y = loader.next()
        assert x.device.type == "cpu" and y.device.type == "cpu"
        assert torch.equal(x[:, 1:], y[:, :-1])
    finally:
        loader.close()


def test_batch_prefetcher_cpu_staging_opt_in_preserves_batch_identity(tr, monkeypatch):
    """A CPU H2D opt-in stays an identity path through ``next()``."""
    host_x = torch.zeros((tr.BATCH_SIZE, tr.BLOCK_SIZE), dtype=torch.int64)
    host_y = torch.ones_like(host_x)

    def fail_cuda_probe():
        pytest.fail("CPU H2D path must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU H2D path must not construct CUDA stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(
        tr.BatchPrefetcher,
        "_gather_pinned_host",
        lambda self: (host_x, host_y),
    )
    loader = tr.BatchPrefetcher("train", async_host=False, cuda_staging=True)
    try:
        x, y = loader.next()
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
        assert x is host_x and y is host_y
    finally:
        loader.close()


def test_batch_prefetch_defaults_unchanged(tr):
    """The H2D and host-prefetch defaults remain enabled; CPU still no-ops H2D."""
    assert tr.USE_PREFETCH_ASYNC is True
    assert tr.USE_PREFETCH_H2D is True


def test_prefetch_h2d_cpu_gate_does_not_probe_cuda(tr, monkeypatch):
    """A CPU device short-circuits before querying CUDA availability."""

    def fail_cuda_probe():
        pytest.fail("CPU H2D gate must not query CUDA availability")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    assert tr.prefetch_h2d_skip_reason() == "device-not-cuda: cpu"


def test_prefetch_h2d_opt_out_does_not_probe_cuda(tr, monkeypatch):
    """An explicit H2D opt-out skips CUDA probing and allocation entirely."""
    monkeypatch.setattr(tr, "device", torch.device("cuda"))

    def fail_cuda_probe():
        pytest.fail("H2D opt-out must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("H2D opt-out must not construct CUDA stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(
        tr.BatchPrefetcher,
        "_gather_pinned_host",
        lambda self: (torch.zeros(1, 1, dtype=torch.int64),) * 2,
    )
    loader = tr.BatchPrefetcher("train", async_host=False, cuda_staging=False)
    try:
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
    finally:
        loader.close()


def test_prefetch_h2d_global_opt_out_does_not_probe_cuda(tr, monkeypatch):
    """The default H2D flag opt-out skips CUDA probing and allocation."""
    monkeypatch.setattr(tr, "device", torch.device("cuda"))
    monkeypatch.setattr(tr, "USE_PREFETCH_H2D", False)

    def fail_cuda_probe():
        pytest.fail("global H2D opt-out must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("global H2D opt-out must not construct CUDA stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(
        tr.BatchPrefetcher,
        "_gather_pinned_host",
        lambda self: (torch.zeros(1, 1, dtype=torch.int64),) * 2,
    )
    loader = tr.BatchPrefetcher("train", async_host=False)
    try:
        assert tr.USE_PREFETCH_H2D is False
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
    finally:
        loader.close()


def test_prefetch_h2d_skip_reason_is_cpu_safe(tr, monkeypatch):
    """The H2D gate reports CPU/runtime skips without CUDA construction."""
    assert tr.prefetch_h2d_skip_reason() == "device-not-cuda: cpu"

    monkeypatch.setattr(tr, "device", torch.device("cuda"))
    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False)
    assert (
        tr.prefetch_h2d_skip_reason()
        == "CUDA unavailable: torch.cuda.is_available() is false"
    )

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("unavailable CUDA must not construct stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(
        tr.BatchPrefetcher,
        "_gather_pinned_host",
        lambda self: (torch.zeros(1, 1, dtype=torch.int64),) * 2,
    )
    loader = tr.BatchPrefetcher("train", async_host=False, cuda_staging=True)
    try:
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
    finally:
        loader.close()


@pytest.mark.parametrize("exc_type", [AssertionError, RuntimeError])
def test_prefetch_h2d_skip_reason_handles_cuda_probe_errors(
    tr, monkeypatch, exc_type
):
    """A failing CUDA availability probe remains a no-allocation skip."""
    monkeypatch.setattr(tr, "device", torch.device("cuda"))

    def raise_probe_error():
        raise exc_type("simulated CUDA runtime failure")

    monkeypatch.setattr(tr.torch.cuda, "is_available", raise_probe_error)
    assert (
        tr.prefetch_h2d_skip_reason()
        == f"CUDA unavailable: torch.cuda.is_available() raised {exc_type.__name__}"
    )

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("failed CUDA probe must not construct stream/event objects")

    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(
        tr.BatchPrefetcher,
        "_gather_pinned_host",
        lambda self: (torch.zeros(1, 1, dtype=torch.int64),) * 2,
    )
    loader = tr.BatchPrefetcher("train", async_host=False, cuda_staging=True)
    try:
        assert loader._cuda_staging is False
        assert loader._stream is None
        assert loader._device_next is None
    finally:
        loader.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required: CPU-only runs cover the no-op contract, not H2D lookahead",
)
def test_batch_prefetcher_cuda_staging_device_lookahead(tr, monkeypatch):
    """CUDA staging returns device batches and keeps one staged lookahead."""
    monkeypatch.setattr(tr, "device", torch.device("cuda"))
    monkeypatch.setattr(tr, "_train_data", None)
    monkeypatch.setattr(tr, "_val_data", None)
    monkeypatch.setattr(tr, "_offsets", None)
    loader = tr.BatchPrefetcher("train", async_host=True, cuda_staging=True)
    try:
        x, y = loader.next()
        assert loader._cuda_staging is True
        assert loader._stream is not None
        assert loader._device_next is not None
        assert x.device.type == "cuda" and y.device.type == "cuda"
        assert x.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
    finally:
        loader.close()
        torch.cuda.synchronize()


def test_batch_prefetcher_next(tr):
    loader = tr.BatchPrefetcher("train")
    try:
        x1, y1 = loader.next()
        x2, y2 = loader.next()
        assert x1.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
        assert x2.shape == x1.shape
        # consecutive random batches should almost never be identical
        assert not torch.equal(x1, x2)
    finally:
        loader.close()


def test_batch_prefetcher_async_and_sync(tr):
    """Async and sync paths both yield contiguous LM windows of the right shape."""
    for async_host in (True, False):
        loader = tr.BatchPrefetcher("train", async_host=async_host)
        try:
            assert loader._async is async_host
            xs = []
            for _ in range(3):
                x, y = loader.next()
                assert x.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
                assert y.shape == x.shape
                assert x.is_contiguous() and y.is_contiguous()
                assert torch.equal(x[:, 1:], y[:, :-1])
                xs.append(x.clone())
            assert not torch.equal(xs[0], xs[1])
        finally:
            loader.close()


def test_batch_prefetcher_producer_alive(tr):
    """Async path keeps a daemon producer thread filling the one-slot queue."""
    loader = tr.BatchPrefetcher("train", async_host=True)
    try:
        assert loader._thread is not None and loader._thread.is_alive()
        assert loader._q is not None
        x, y = loader.next()
        assert x.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
        assert y.dtype == torch.int64
        # Producer refills while we hold the batch (queue depth 1).
        x2, y2 = loader.next()
        assert x2.shape == x.shape
        assert not torch.equal(x, x2)
    finally:
        loader.close()
        assert loader._thread is None


def test_batch_prefetcher_producer_failure_surfaces_on_next(tr, monkeypatch):
    """Async producer failures reach the consumer instead of hanging forever."""

    def fail_gather(_split):
        raise ValueError("simulated host gather failure")

    monkeypatch.setattr(tr, "_gather_batch_host_numpy", fail_gather)
    loader = tr.BatchPrefetcher("train", async_host=True)
    try:
        with pytest.raises(RuntimeError, match="prefetch producer failed") as exc_info:
            loader.next()
        assert isinstance(exc_info.value.__cause__, ValueError)
    finally:
        loader.close()


def test_batch_prefetcher_close_idempotent(tr):
    loader = tr.BatchPrefetcher("val", async_host=True)
    _ = loader.next()
    loader.close()
    loader.close()  # second close must not raise


def test_dataloader_num_workers_zero(tr, monkeypatch):
    monkeypatch.setattr(tr, "USE_DATALOADER", True)
    monkeypatch.setattr(tr, "NUM_WORKERS", 0)
    src = tr.DataLoaderBatchSource("train")
    x, y = src.next()
    assert x.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
    assert y.dtype == torch.int64
    dl = src._loader
    assert dl.num_workers == 0
    assert dl.pin_memory is False  # CPU device
    assert dl.persistent_workers is False


def test_dataloader_negative_workers_clamp_to_cpu_safe_zero(tr, monkeypatch):
    """Invalid negative worker counts stay a synchronous CPU DataLoader."""
    monkeypatch.setattr(tr, "NUM_WORKERS", -1)
    src = tr.DataLoaderBatchSource("train")
    x, y = src.next()
    dl = src._loader
    assert x.device.type == "cpu" and y.device.type == "cpu"
    assert dl.num_workers == 0
    assert dl.pin_memory is False
    assert dl.persistent_workers is False
    assert not hasattr(dl, "prefetch_factor") or dl.prefetch_factor is None


def test_dataloader_worker_prefetch_contract(tr, monkeypatch):
    """Worker DataLoader keeps full host batches and bounded lookahead."""
    monkeypatch.setattr(tr, "NUM_WORKERS", 2)
    src = tr.DataLoaderBatchSource("train")
    try:
        dl = src._loader
        assert dl.batch_size is None
        assert dl.prefetch_factor == 2
        assert dl.worker_init_fn is tr._dataloader_worker_init
        assert dl.persistent_workers is True
    finally:
        del src


def test_dataloader_worker_init_separates_cpu_rng_streams(tr, monkeypatch):
    """Worker initialization gives distinct, reproducible CPU RNG streams."""
    monkeypatch.setattr(tr.torch, "initial_seed", lambda: 100)
    state = torch.random.get_rng_state()
    try:
        tr._dataloader_worker_init(0)
        worker_zero = torch.rand(4)
        tr._dataloader_worker_init(1)
        worker_one = torch.rand(4)
        tr._dataloader_worker_init(0)
        worker_zero_repeat = torch.rand(4)
    finally:
        torch.random.set_rng_state(state)

    assert not torch.equal(worker_zero, worker_one)
    assert torch.equal(worker_zero, worker_zero_repeat)


def test_dataloader_cpu_h2d_identity_does_not_probe_cuda(tr, monkeypatch):
    """CPU DataLoader H2D remains identity-only without CUDA initialization."""
    host_x = torch.zeros((tr.BATCH_SIZE, tr.BLOCK_SIZE), dtype=torch.int64)
    host_y = torch.ones_like(host_x)

    def fail_cuda_probe():
        pytest.fail("CPU DataLoader path must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU DataLoader path must not construct CUDA objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(tr, "NUM_WORKERS", 0)
    monkeypatch.setattr(
        tr._HostBatchIterable,
        "__iter__",
        lambda self: iter(((host_x, host_y),)),
    )
    src = tr.DataLoaderBatchSource("train")
    x, y = src.next()
    assert src._loader.pin_memory is False
    assert x is host_x and y is host_y


def test_dataloader_cpu_h2d_opt_out_preserves_identity(tr, monkeypatch):
    """DataLoader CPU batches stay identity-only with H2D staging opted out."""
    host_x = torch.zeros((tr.BATCH_SIZE, tr.BLOCK_SIZE), dtype=torch.int64)
    host_y = torch.ones_like(host_x)

    def fail_cuda_probe():
        pytest.fail("CPU DataLoader H2D opt-out must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU DataLoader H2D opt-out must not construct CUDA objects")

    monkeypatch.setattr(tr, "USE_PREFETCH_H2D", False)
    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(tr, "NUM_WORKERS", 0)
    monkeypatch.setattr(
        tr._HostBatchIterable,
        "__iter__",
        lambda self: iter(((host_x, host_y),)),
    )
    src = tr.DataLoaderBatchSource("train")
    x, y = src.next()
    assert tr.USE_PREFETCH_H2D is False
    assert src._loader.pin_memory is False
    assert x is host_x and y is host_y


def test_dataloader_preserves_validation_split_contract(tr, monkeypatch):
    """The DataLoader adapter forwards ``val`` without changing CPU identity."""
    host_x = torch.zeros((tr.BATCH_SIZE, tr.BLOCK_SIZE), dtype=torch.int64)
    host_y = torch.ones_like(host_x)
    seen = []

    def gather(split):
        seen.append(split)
        return host_x, host_y

    monkeypatch.setattr(tr, "NUM_WORKERS", 0)
    monkeypatch.setattr(tr, "_gather_batch_host", gather)
    src = tr.DataLoaderBatchSource("val")
    x, y = src.next()
    assert seen == ["val"]
    assert src._loader.pin_memory is False
    assert x is host_x and y is host_y


def test_dataloader_cpu_h2d_workers_keep_pin_memory_off(tr, monkeypatch):
    """CPU DataLoader workers keep H2D staging disabled without CUDA setup."""
    def fail_cuda_probe():
        pytest.fail("CPU DataLoader workers must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU DataLoader workers must not construct CUDA objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(tr, "NUM_WORKERS", 2)
    src = tr.DataLoaderBatchSource("train")
    x, y = src.next()
    assert src._loader.num_workers == 2
    assert src._loader.pin_memory is False
    assert src._loader.persistent_workers is True
    assert x.device.type == "cpu" and y.device.type == "cpu"
    assert x.is_contiguous() and y.is_contiguous()
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_dataloader_cpu_h2d_workers_preserve_identity(tr, monkeypatch):
    """CPU worker batches stay identity-only through the H2D helper."""
    def fail_cuda_probe():
        pytest.fail("CPU DataLoader workers must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU DataLoader workers must not construct CUDA objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(tr, "NUM_WORKERS", 2)
    original_to_device = tr._to_train_device
    calls = {"count": 0}

    def checked_to_device(x, y):
        out = original_to_device(x, y)
        calls["count"] += 1
        assert out[0] is x and out[1] is y
        return out

    monkeypatch.setattr(tr, "_to_train_device", checked_to_device)
    src = tr.DataLoaderBatchSource("train")
    try:
        x, y = src.next()
        assert calls["count"] == 1
        assert x.device.type == "cpu" and y.device.type == "cpu"
        assert x.is_contiguous() and y.is_contiguous()
    finally:
        del src


def test_dataloader_cpu_h2d_workers_preserve_identity_across_batches(
    tr, monkeypatch
):
    """Queued CPU worker batches stay identity-only across repeated ``next()`` calls."""

    def fail_cuda_probe():
        pytest.fail("CPU DataLoader workers must not query CUDA availability")

    def fail_cuda_factory(*_args, **_kwargs):
        pytest.fail("CPU DataLoader workers must not construct CUDA objects")

    monkeypatch.setattr(tr.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(tr.torch.cuda, "Stream", fail_cuda_factory)
    monkeypatch.setattr(tr.torch.cuda, "Event", fail_cuda_factory)
    monkeypatch.setattr(tr, "NUM_WORKERS", 2)
    original_to_device = tr._to_train_device
    calls = {"count": 0}

    def checked_to_device(x, y):
        out = original_to_device(x, y)
        calls["count"] += 1
        assert out[0] is x and out[1] is y
        return out

    monkeypatch.setattr(tr, "_to_train_device", checked_to_device)
    src = tr.DataLoaderBatchSource("train")
    try:
        for expected_calls in range(1, 4):
            x, y = src.next()
            assert calls["count"] == expected_calls
            assert x.device.type == "cpu" and y.device.type == "cpu"
            assert x.is_contiguous() and y.is_contiguous()
            assert torch.equal(x[:, 1:], y[:, :-1])
    finally:
        del src


def test_dataloader_persistent_workers(tr, monkeypatch):
    monkeypatch.setattr(tr, "USE_DATALOADER", True)
    monkeypatch.setattr(tr, "NUM_WORKERS", 2)
    src = tr.DataLoaderBatchSource("train")
    dl = src._loader
    assert dl.num_workers == 2
    assert dl.persistent_workers is True
    # pull a few batches (workers stay alive)
    shapes = []
    for _ in range(3):
        x, y = src.next()
        shapes.append(x.shape)
        assert y.shape == x.shape
    assert shapes == [(tr.BATCH_SIZE, tr.BLOCK_SIZE)] * 3


def test_make_batch_source_default_is_prefetcher(tr, monkeypatch):
    monkeypatch.setattr(tr, "USE_DATALOADER", False)
    src = tr.make_batch_source("train")
    assert isinstance(src, tr.BatchPrefetcher)


def test_make_batch_source_dataloader(tr, monkeypatch):
    monkeypatch.setattr(tr, "USE_DATALOADER", True)
    monkeypatch.setattr(tr, "NUM_WORKERS", 0)
    src = tr.make_batch_source("train")
    assert isinstance(src, tr.DataLoaderBatchSource)


def test_hot_loop_logging_defers_item(tr):
    """Sanity: train_step returns a tensor; TrainLossLogger owns rare .item()."""
    model = tr.bdh.BDH(tr.bdh.BDHConfig(n_layer=1, n_head=1, n_embd=32)).to(tr.device)
    opt = tr.make_optimizer(model)
    x, y = tr.get_batch("train")
    loss = tr.train_step(model, opt, x, y)
    assert torch.is_tensor(loss)
    logger = tr.TrainLossLogger(log_freq=10, device=tr.device, async_cuda=False, max_iters=1)
    # step 1 is non-boundary → no print / no .item required
    logger.update(loss, step=1)
    assert logger._steps == 1 and logger._acc is not None
    logger.close()  # flushes partial window
