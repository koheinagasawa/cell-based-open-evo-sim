from __future__ import annotations

import importlib
import json
import os
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    genome: str
    state_size: int
    action_size: int
    steps: int = 1
    seed: int | None = None

    # NEW: generic interpreter spec (optional)
    # {
    #   "class": "package.module.ClassName",
    #   "kwargs": { ... }            # must match the class __init__ signature
    # }
    interpreter: Dict[str, Any] | None = None

    # Optional metadata blob
    metadata: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            genome=d.get("genome", "Unknown"),
            state_size=int(d.get("state_size", 4)),
            action_size=int(d.get("action_size", 2)),
            steps=int(d.get("steps", 1)),
            seed=d.get("seed", None),
            interpreter=d.get("interpreter"),
            metadata=d.get("metadata"),
        )


# ---------- Utilities ----------
def _import_from_string(path: str):
    """Import a class/function from a fully-qualified name."""
    mod, _, name = path.rpartition(".")
    if not mod:
        raise ImportError(f"Invalid import path: {path}")
    module = importlib.import_module(mod)
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ImportError(f"{path} not found") from e


def _coerce_slot_defs_in_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort: convert JSON-friendly pair-lists to Python slice objects.
    This is a convenience for interpreters that accept a 'slot_defs' dict.
    Other interpreters will just ignore or use kwargs as-is.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    if "slot_defs" in kwargs and isinstance(kwargs["slot_defs"], dict):
        new = {}
        for k, v in kwargs["slot_defs"].items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                new[k] = slice(int(v[0]), int(v[1]))
            else:
                new[k] = v
        kwargs = dict(kwargs)
        kwargs["slot_defs"] = new
    return kwargs


def _build_interpreter(interpreter_spec: Dict[str, Any] | None):
    """Instantiate an interpreter if spec is provided; otherwise return None."""
    if not interpreter_spec:
        return None
    cls_path = interpreter_spec.get("class")
    if not cls_path:
        raise ValueError("interpreter spec missing 'class' field")
    cls = _import_from_string(cls_path)
    kwargs = interpreter_spec.get("kwargs", {}) or {}
    kwargs = _coerce_slot_defs_in_kwargs(kwargs)
    return cls(**kwargs)


def _unique_exp_dir(root: pathlib.Path, base_name: str) -> pathlib.Path:
    """Create a unique experiment directory under root using base_name."""
    # Keep it filesystem-safe and not too long
    name = base_name[:80].rstrip("._-")
    path = root / name
    if not path.exists():
        return path
    i = 2
    while True:
        cand = root / f"{name}_{i:02d}"
        if not cand.exists():
            return cand
        i += 1


import importlib
import json

# ---------- Factory ----------
# tests/utils/test_utils.py（差分のみ）
import os  # re 追加
import pathlib
import re
import uuid

# ...（中略）...


def _unique_exp_dir(root: pathlib.Path, base_name: str) -> pathlib.Path:
    """Create a unique experiment directory under root using base_name."""
    # Keep it filesystem-safe and not too long
    name = base_name[:80].rstrip("._-")
    path = root / name
    if not path.exists():
        return path
    i = 2
    while True:
        cand = root / f"{name}_{i:02d}"
        if not cand.exists():
            return cand
        i += 1


def prepare_run(
    config_dict: Dict[str, Any],
    session_dir: str | None = None,
    exp_name: str | None = None,
):
    """Create (RunConfig, Recorder) with optional, generic interpreter instantiation.

    exp_name:
        If provided, the experiment subfolder will be <session_dir>/<exp_name>.
        Otherwise a unique name is generated (kept for backward-compat).
    """
    run_config = RunConfig.from_dict(config_dict)
    interpreter = _build_interpreter(run_config.interpreter)

    session_root = (
        pathlib.Path(session_dir)
        if session_dir is not None
        else pathlib.Path(
            os.environ.get("ALIFE_OUTPUT_SESSION_DIR", "outputs/session_default")
        )
    )
    session_root.mkdir(parents=True, exist_ok=True)

    if exp_name:
        exp_dir = _unique_exp_dir(session_root, exp_name)
    else:
        # fallback: keep previous behavior (short uuid)
        exp_dir = session_root / f"exp_{uuid.uuid4().hex[:8]}"

    recorder = Recorder(
        output_dir=str(exp_dir),
        run_config=run_config,
        interpreter=interpreter,
        create_now=True,
    )

    meta = {
        "genome": run_config.genome,
        "state_size": run_config.state_size,
        "action_size": run_config.action_size,
        "steps": run_config.steps,
        "seed": run_config.seed,
        "experiment_name": exp_dir.name,  # ← 追加で残す
        "interpreter_class": interpreter.__class__.__name__ if interpreter else None,
        "interpreter_module": interpreter.__class__.__module__ if interpreter else None,
        "user_metadata": run_config.metadata or {},
    }
    recorder.save_metadata(meta)
    return run_config, recorder


class Recorder:
    """Collects per-step/per-cell data and writes them under a run-specific directory.

    Directory layout:
        <session_dir>/<exp_id>/
            metadata.json
            positions.npy
            states.npy
            raw_outputs.npy            # optional if provided
            interpreted_slots.jsonl    # sparse/dict-friendly
    """

    def __init__(
        self,
        output_dir: str | None = None,
        run_config=None,
        interpreter=None,
        create_now: bool = False,
    ):
        # Determine run directory
        if output_dir is None:
            # Fallback: create within env session dir or default
            session_dir = os.environ.get(
                "ALIFE_OUTPUT_SESSION_DIR", os.path.join("outputs", "session_default")
            )
            output_dir = os.path.join(session_dir, "exp_unset")
        self.output_dir = str(output_dir)
        if create_now:
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Provenance
        self.run_config = run_config
        self.interpreter = interpreter

        # Buffers
        self.positions = []  # list of [t, cell_id, *pos]
        self.states = []  # list of [t, cell_id, *state]
        self.raw_outputs = []  # optional; list of [t, cell_id, *raw]
        self.slot_stream = []  # jsonl of {"t":..., "cell_id":..., "slots": {...}}

    def record(self, t: int, cell):
        """Record minimal cell info. Flexible to interpreter-based outputs."""
        cid = getattr(cell, "id", None) or f"cell_{id(cell)}"
        pos = np.asarray(cell.position, dtype=float).tolist()
        st = np.asarray(getattr(cell, "state", []), dtype=float).tolist()

        self.positions.append([int(t), cid, *pos])
        self.states.append([int(t), cid, *st])

        # Optional raw_output
        raw = getattr(cell, "raw_output", None)
        if raw is not None:
            self.raw_outputs.append(
                [int(t), cid, *np.asarray(raw, dtype=float).tolist()]
            )

        # Interpreter slots (dict) are serialized sparsely
        slots = getattr(cell, "output_slots", None)
        if isinstance(slots, dict):
            self.slot_stream.append(
                {"t": int(t), "cell_id": cid, "slots": _to_serializable(slots)}
            )

    # --- Saving APIs ---------------------------------------------------------

    def save_metadata(self, extra: dict[str, Any] | None = None):
        os.makedirs(self.output_dir, exist_ok=True)
        meta_path = os.path.join(self.output_dir, "metadata.json")
        meta = {
            "output_dir": self.output_dir,
            "run_config": _to_serializable(self.run_config),
        }
        # Interpreter is optional and generic
        if self.interpreter is not None:
            to_meta = getattr(self.interpreter, "to_metadata", None)
            if callable(to_meta):
                meta["interpreter"] = _to_serializable(self.interpreter.to_metadata())
            else:
                meta["interpreter"] = {
                    "class": self.interpreter.__class__.__name__,
                    "module": self.interpreter.__class__.__module__,
                }
        if extra:
            meta.update(extra)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save_all(self):
        """Write all accumulated arrays/files."""
        os.makedirs(self.output_dir, exist_ok=True)

        # npy arrays for dense numeric data
        if self.positions:
            np.save(
                os.path.join(self.output_dir, "positions.npy"),
                np.asarray(self.positions, dtype=object),
            )
        if self.states:
            np.save(
                os.path.join(self.output_dir, "states.npy"),
                np.asarray(self.states, dtype=object),
            )
        if self.raw_outputs:
            np.save(
                os.path.join(self.output_dir, "raw_outputs.npy"),
                np.asarray(self.raw_outputs, dtype=object),
            )

        # JSON Lines for heterogeneous slots
        if self.slot_stream:
            with open(
                os.path.join(self.output_dir, "interpreted_slots.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for row in self.slot_stream:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"✅ Experiment results saved in: {self.output_dir}")


# --- helpers ----------------------------------------------------------------
def _to_serializable(obj):
    """Convert objects (dataclass/ndarray) to JSON-friendly dicts."""
    if obj is None:
        return None
    try:
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def plot_state_trajectories(recorder, show=True):
    """Plot per-cell state time series from recorder.states (list of [t, cell_id, *state])."""
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    rows = getattr(recorder, "states", None) or []
    if not rows:
        print("No state data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        t = int(row[0])
        cid = row[1]
        s = np.asarray(row[2:], dtype=float)
        by_cell[cid].append((t, s))

    # Sort by time and stack
    series = []
    for cid, lst in by_cell.items():
        lst.sort(key=lambda x: x[0])
        T = np.array([t for t, _ in lst], dtype=int)
        S = np.vstack([s for _, s in lst])  # (T, state_dim)
        series.append((cid, T, S))

    import math

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (cid, T, S) in zip(axes, series):
        for j in range(S.shape[1]):
            ax.plot(T, S[:, j], label=f"state[{j}]")
        ax.set_title(f"Cell {cid}")
        ax.set_xlabel("t")
        ax.set_ylabel("state")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()


def plot_2D_position_trajectories(
    recorder, show=True, mark_start_end=True, equal_aspect=True
):
    """Plot per-cell 2D position trajectories from recorder.positions (list of [t, cell_id, x, y, ...])."""
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    rows = getattr(recorder, "positions", None) or []
    if not rows:
        print("No position data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 4:  # need at least t, id, x, y
            continue
        t = int(row[0])
        cid = row[1]
        x = float(row[2])
        y = float(row[3])
        by_cell[cid].append((t, x, y))

    series = []
    for cid, lst in by_cell.items():
        lst.sort(key=lambda z: z[0])
        X = np.array([x for _, x, _ in lst], dtype=float)
        Y = np.array([y for _, _, y in lst], dtype=float)
        series.append((cid, X, Y))

    fig, ax = plt.subplots(figsize=(6, 6))
    for cid, X, Y in series:
        ax.plot(X, Y, label=f"cell {cid}", linewidth=1.5)
        if mark_start_end and X.size > 0:
            ax.scatter([X[0]], [Y[0]], marker="o", s=30)  # start
            ax.scatter([X[-1]], [Y[-1]], marker="*", s=80)  # end

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    if show:
        plt.show()
