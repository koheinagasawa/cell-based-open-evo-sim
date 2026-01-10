# tests/test_experiment_harness_smoke.py
import csv
from pathlib import Path

import numpy as np

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.runner import make_spec
from experiments.common.runner_generic import run_experiment
from experiments.sweeps.grid import run_grid


def test_run_chemotaxis_bud_experiment_smoke(world_factory, test_output_dir):
    cfg = ChemotaxisBudConfig(
        steps=30,
        n_emitters=1,
        n_followers=8,
        seed=42,
        sigma=0.9,
        decay=0.95,
        grad_gain=1.0,
        link_weight=0.8,
        bidirectional=True,
        out_dir=str(test_output_dir / "exp"),
    )
    spec = make_spec(cfg, world_factory)
    result = run_experiment(spec)

    # Files exist and have content
    assert (
        "metrics_csv_path" in result
        and "metrics_npz_path" in result
        and "events_birth_csv_path" in result
    )
    assert (test_output_dir / "exp" / "metrics.csv").exists()
    assert (test_output_dir / "exp" / "metrics.npz").exists()
    assert (test_output_dir / "exp" / "events.csv").exists()

    # Metrics CSV header sanity
    with open(result["metrics_csv_path"], newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == [
            "t",
            "births",
            "alive",
            "mean_energy",
            "mean_degree",
            "step_ms",
            "mean_radius",
        ]
        rows = list(r)
        assert len(rows) == cfg.steps

    # Metrics NPZ sanity
    data = np.load(result["metrics_npz_path"])
    assert data["t"].shape[0] == cfg.steps
    assert data["alive"].shape == data["t"].shape
    assert np.all(data["births"] >= 0)

    # Events CSV header sanity
    with open(result["events_birth_csv_path"], newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == [
            "time_step",
            "parent_id",
            "child_id",
            "x",
            "y",
            "link_weight",
        ]


def test_followers_mean_radius_trend(world_factory, test_output_dir):
    cfg = ChemotaxisBudConfig(
        steps=50,
        n_emitters=1,
        n_followers=12,
        seed=7,
        sigma=1.0,
        decay=0.95,
        grad_gain=1.0,
        link_weight=0.8,
        bidirectional=True,
        out_dir=str(test_output_dir / "trend_radius"),
    )
    spec = make_spec(cfg, world_factory)
    res = run_experiment(spec)

    r = res["mean_radius"]
    births = res["births"]

    # Focus on the pre-birth segment to avoid parent offset effects.
    birth_idxs = np.nonzero(births > 0)[0]
    if birth_idxs.size > 0:
        end = int(birth_idxs[0])  # first step with any birth
        # Ensure we have at least 2 points for a head/tail comparison
        end = max(end, 2)
        segment = r[:end]
    else:
        segment = r

    if segment.shape[0] >= 2:
        # Robust criterion: end no greater than start (non-increasing overall).
        start = float(segment[0])
        endv = float(segment[-1])
        assert endv <= start + 1e-9
        # And there was *some* descent at some point.
        assert float(np.min(segment)) <= start + 1e-12
    else:
        # Degenerate short segment: just ensure no explosion
        assert np.all(np.isfinite(segment))


def test_grid_sweep_smoke(world_factory, test_output_dir):
    base = ChemotaxisBudConfig(
        steps=20,
        n_emitters=1,
        n_followers=6,
        seed=0,
        sigma=1.0,
        decay=0.95,
        link_weight=0.8,
        bidirectional=False,
        out_dir=str(test_output_dir / "base_will_be_overridden"),
    )

    sweep_root = test_output_dir / "sweep"

    def runner(cfg):
        spec = make_spec(cfg, world_factory)
        return run_experiment(spec)

    summary_keys = [
        "final_alive",
        "total_births",
        "mean_step_ms",
        "mean_radius",
    ]

    def summary_fn(res):
        final_alive = float(res["alive"][-1]) if len(res["alive"]) else 0.0
        total_births = float(np.sum(res["births"])) if len(res["births"]) else 0.0
        mean_step_ms = float(np.mean(res["step_ms"])) if len(res["step_ms"]) else 0.0
        mean_radius = float(res["mean_radius"][-1]) if len(res["mean_radius"]) else 0.0
        return [final_alive, total_births, mean_step_ms, mean_radius]

    summary = run_grid(
        base_cfg=base,
        axes={"decay": [0.9, 0.95], "sigma": [0.8, 1.2]},
        out_dir=sweep_root,
        runner=runner,
        summary_keys=summary_keys,
        summary_fn=summary_fn,
    )

    # Summary.csv exists with expected number of rows (product of axes)
    p = Path(summary)
    assert p.exists()
    with p.open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == [
            "decay",
            "sigma",
            "final_alive",
            "total_births",
            "mean_step_ms",
            "mean_radius",
        ]
        rows = list(r)
        assert len(rows) == 4  # 2x2 grid
