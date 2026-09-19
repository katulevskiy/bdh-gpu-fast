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


def test_batch_prefetcher_next(tr):
    loader = tr.BatchPrefetcher("train")
    x1, y1 = loader.next()
    x2, y2 = loader.next()
    assert x1.shape == (tr.BATCH_SIZE, tr.BLOCK_SIZE)
    assert x2.shape == x1.shape
    # consecutive random batches should almost never be identical
    assert not torch.equal(x1, x2)


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
    """Sanity: train_step returns a tensor; logging path uses detach + rare .item()."""
    model = tr.bdh.BDH(tr.bdh.BDHConfig(n_layer=1, n_head=1, n_embd=32)).to(tr.device)
    opt = tr.make_optimizer(model)
    x, y = tr.get_batch("train")
    loss = tr.train_step(model, opt, x, y)
    assert torch.is_tensor(loss)
    det = loss.detach()
    assert det.device == loss.device
    # .item() only when printing — not required for the step itself
    _ = float(det)  # allowed outside hot path
