from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


def ensure_dir(pathlike) -> Path:
    """Create directory if it does not exist and return it as Path."""
    p = Path(pathlike)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_metrics_csv_npz(
    out_dir,
    arrays: Mapping[str, np.ndarray],
    *,
    csv_name: str = "metrics.csv",
    npz_name: str = "metrics.npz",
    header: Iterable[str] = (
        "t",
        "births",
        "alive",
        "mean_energy",
        "mean_degree",
        "step_ms",
    ),
) -> Dict[str, str]:
    """
    Save metric arrays to CSV and NPZ in a consistent way.

    Expected keys in `arrays`:
      - "t", "births", "alive", "mean_energy", "mean_degree", "step_ms"
    Extra keys are stored in NPZ only.

    Returns:
      {"metrics_csv_path": str, "metrics_npz_path": str}
    """
    out_path = ensure_dir(out_dir)
    csv_path = out_path / csv_name
    npz_path = out_path / npz_name

    # --- CSV (only header columns) ---
    hdr = list(header)
    # Validate presence (fail fast for missing essentials)
    for k in hdr:
        if k not in arrays:
            raise KeyError(f"write_metrics_csv_npz: missing required array '{k}'")

    # All arrays in header must have same length
    n = len(arrays[hdr[0]])
    for k in hdr:
        if len(arrays[k]) != n:
            raise ValueError(f"Array '{k}' length {len(arrays[k])} != {n}")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for i in range(n):
            row = [to_scalar(arrays[k][i]) for k in hdr]
            w.writerow(row)

    # --- NPZ (store everything, including extras) ---
    # Convert to plain numpy arrays for safety
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in arrays.items()})

    return {"metrics_csv_path": str(csv_path), "metrics_npz_path": str(npz_path)}


def to_scalar(x):
    """Convert a value (possibly numpy scalar/array length-1) into a Python scalar for CSV."""
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)) and x.ndim == 0:
        return x.item()
    return float(x)


def summarize_population(alive: np.ndarray, births: np.ndarray) -> Dict[str, float]:
    """
    Lightweight summary helpers for quick experiment overviews.
    Returns totals and simple aggregates.
    """
    alive = np.asarray(alive)
    births = np.asarray(births)
    return {
        "final_alive": float(alive[-1]) if alive.size else 0.0,
        "max_alive": float(np.max(alive)) if alive.size else 0.0,
        "total_births": float(np.sum(births)) if births.size else 0.0,
        "mean_alive": float(np.mean(alive)) if alive.size else 0.0,
    }
