"""Small CPU-only profiler smoke used by local checks and scheduled CI.

This intentionally makes no GPU availability or performance claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import profile_forward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=profile_forward.TRACES,
        help="directory for the generated, gitignored Chrome trace",
    )
    args = parser.parse_args()

    trace_dir = args.trace_dir.resolve()
    _, trace_path = profile_forward.profile_forward(
        torch.device("cpu"),
        B=1,
        T=8,
        warmup=0,
        wait=0,
        active=1,
        trace_dir=trace_dir,
    )
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise RuntimeError(f"profiler did not write a non-empty trace: {trace_path}")
    print(f"CPU profile smoke passed: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
