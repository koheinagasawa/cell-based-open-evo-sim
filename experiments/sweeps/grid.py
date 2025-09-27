from __future__ import annotations

import csv
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


def run_grid(
    base_cfg: Any,
    axes: Dict[str, Iterable],
    out_dir: str | Path,
    *,
    runner: Callable[[Any], dict],
    summary_keys: Sequence[str],
    summary_fn: Callable[[dict], List[Any]],
    subdir_tag_fn: Callable[[Sequence[str], Sequence[Any]], str] = None,
    summary_name: str = "summary.csv",
) -> str:
    """
    Run a grid sweep over a dataclass config.
    Each combination produces a subfolder for outputs.
    A single summary.csv is written at the sweep root.

    Args:
        base_cfg: The base config (dataclass or similar, used with dataclasses.replace)
        axes: Dict of parameter name -> iterable of values
        out_dir: Output directory
        runner: Function taking config and returning a result dict
        summary_keys: List of column names for summary.csv (besides parameter keys)
        summary_fn: Function(result_dict) -> list of summary values (must match summary_keys)
        subdir_tag_fn: Optional function(keys, values) -> str for subdir naming
        summary_name: Name of the summary CSV

    Returns:
        Path to the created summary.csv (as str).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    keys: List[str] = list(axes.keys())
    values: List[List] = [list(v) for v in axes.values()]
    combos = list(product(*values))

    if subdir_tag_fn is None:
        subdir_tag_fn = lambda keys, combo: "_".join(
            f"{k}-{v}" for k, v in zip(keys, combo)
        )

    summary_path = out_root / summary_name
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = keys + list(summary_keys)
        w.writerow(header)

        for idx, combo in enumerate(combos):
            cfg = base_cfg
            for k, v in zip(keys, combo):
                cfg = replace(cfg, **{k: v})

            tag = subdir_tag_fn(keys, combo)
            run_dir = out_root / f"run_{idx:03d}_{tag}"
            cfg = replace(cfg, out_dir=str(run_dir))

            res = runner(cfg)
            summary_vals = summary_fn(res)
            w.writerow(list(combo) + summary_vals)

    return str(summary_path)
