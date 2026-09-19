"""Contract tests for sparse probe guardrail sampling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import bench_sparse_probe as probe


def test_density_guardrail_checks_latest_snapshot_not_best_sample():
    """A late density collapse must fail despite an earlier passing sample."""
    history = [
        {"x": 0.40, "y": 0.40, "xy": 0.20},
        {"x": 0.19, "y": 0.30, "xy": 0.07},
    ]

    ok, detail = probe.density_guardrail(
        history,
        min_final_x=0.20,
        min_final_xy=0.08,
    )

    assert not ok
    assert detail == (
        "final_x=0.1900 final_xy=0.0700 "
        "required_x>=0.2000 required_xy>=0.0800"
    )
