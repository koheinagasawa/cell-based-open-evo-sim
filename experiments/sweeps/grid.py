# experiments/sweeps/grid.py
from __future__ import annotations

import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.runner import run_chemotaxis_bud_experiment


def run_grid(
    world_factory,
    base_cfg: ChemotaxisBudConfig,
    axes: Dict[str, Iterable],
    out_dir: str | Path,
    *,
    summary_name: str = "summary.csv",
) -> str:
    """
    Run a small grid sweep over ChemotaxisBudConfig scalar fields.
    Each combination produces a subfolder metrics (CSV/NPZ).
    A single summary.csv is written at the sweep root.

    Returns:
        Path to the created summary.csv (as str).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare the grid keys and values
    keys: List[str] = list(axes.keys())
    values: List[List] = [list(v) for v in axes.values()]
    combos = list(product(*values))

    summary_path = out_root / summary_name
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = keys + ["final_alive", "total_births", "mean_step_ms", "mean_radius"]
        w.writerow(header)

        for idx, combo in enumerate(combos):
            # Derive cfg by applying overrides
            cfg = base_cfg
            for k, v in zip(keys, combo):
                cfg = replace(cfg, **{k: v})

            # Subdir for this combo
            tag = "_".join(f"{k}-{v}" for k, v in zip(keys, combo))
            run_dir = out_root / f"run_{idx:03d}_{tag}"
            cfg = replace(cfg, out_dir=str(run_dir))

            # Run
            res = run_chemotaxis_bud_experiment(world_factory, cfg)

            # Summaries
            final_alive = float(res["alive"][-1]) if len(res["alive"]) else 0.0
            total_births = float(np.sum(res["births"])) if len(res["births"]) else 0.0
            mean_step_ms = (
                float(np.mean(res["step_ms"])) if len(res["step_ms"]) else 0.0
            )
            mean_radius = (
                float(res["mean_radius"][-1]) if len(res["mean_radius"]) else 0.0
            )

            w.writerow(
                list(combo) + [final_alive, total_births, mean_step_ms, mean_radius]
            )

    return str(summary_path)
