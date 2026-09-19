"""CPU microbench: baseline list/from_numpy get_batch vs vectorized train.get_batch."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT = ROOT / "input.txt"
BLOCK_SIZE = 512
BATCH_SIZE = 32


def fetch_data():
    if not INPUT.exists():
        import requests

        url = (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        )
        INPUT.write_text(requests.get(url, timeout=60).text)


def get_batch_baseline(split: str, device: torch.device):
    """Original pathway-style: Python list comprehension of from_numpy per sample."""
    data = np.memmap(INPUT, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


def timed(fn, warmup=5, reps=50):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def main():
    fetch_data()
    device = torch.device("cpu")

    # Import after data exists; train.py prints device on import.
    import train as tr

    tr.input_file_path = str(INPUT)
    tr.BLOCK_SIZE = BLOCK_SIZE
    tr.BATCH_SIZE = BATCH_SIZE
    tr.device = device
    tr._train_data = None
    tr._val_data = None
    tr._offsets = None

    # Correctness: same shapes / dtypes / split bounds for a fixed seed.
    torch.manual_seed(0)
    xb, yb = get_batch_baseline("train", device)
    torch.manual_seed(0)
    xo, yo = tr.get_batch("train")
    assert xb.shape == xo.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert yb.shape == yo.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert torch.equal(xb, xo) and torch.equal(yb, yo)

    torch.manual_seed(1)
    xb_v, yb_v = get_batch_baseline("val", device)
    torch.manual_seed(1)
    xo_v, yo_v = tr.get_batch("val")
    assert torch.equal(xb_v, xo_v) and torch.equal(yb_v, yo_v)

    def run_base():
        get_batch_baseline("train", device)

    def run_opt():
        tr.get_batch("train")

    tb = timed(run_base)
    to = timed(run_opt)
    print(f"device={device} BLOCK_SIZE={BLOCK_SIZE} BATCH_SIZE={BATCH_SIZE}")
    print(f"get_batch baseline (list/from_numpy) median: {tb*1000:.3f} ms")
    print(f"get_batch vectorized+pretensor median:       {to*1000:.3f} ms  ({tb/to:.2f}x)")


if __name__ == "__main__":
    main()
