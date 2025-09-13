# tests/test_experiment_harness_smoke.py
import csv

import numpy as np

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.runner import run_chemotaxis_bud_experiment


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
    result = run_chemotaxis_bud_experiment(world_factory, cfg)

    # Files exist and have content
    assert "csv_path" in result and "npz_path" in result
    assert (test_output_dir / "exp" / "metrics.csv").exists()
    assert (test_output_dir / "exp" / "metrics.npz").exists()

    # CSV header sanity
    with open(result["csv_path"], newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == [
            "t",
            "births",
            "alive",
            "mean_energy",
            "mean_degree",
            "step_ms",
        ]
        rows = list(r)
        assert len(rows) == cfg.steps

    # NPZ sanity
    data = np.load(result["npz_path"])
    assert data["t"].shape[0] == cfg.steps
    assert data["alive"].shape == data["t"].shape
    assert np.all(data["births"] >= 0)
