# experiments/chemotaxis_bud/runner.py
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.genomes import (
    EmitterContinuous,
    FollowerChemotaxisAndBud,
)
from experiments.common.event_logger import EventLogger
from experiments.common.frame_dumper import FrameDumper
from experiments.common.metrics import write_metrics_csv_npz
from simulation.cell import Cell
from simulation.fields import FieldChannel, FieldRouter
from simulation.input_layout import InputLayout
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ParentChildLinkWrapper, SimpleBudding


def _make_interpreter(state_size: int = 4) -> SlotBasedInterpreter:
    """
    Interpreter with slots:
      state: [0:S)
      move:  [S:S+2)
      bud:   [S+2]
    """
    S = state_size
    return SlotBasedInterpreter(
        {
            "state": slice(0, S),
            "move": slice(S, S + 2),
            "bud": S + 2,
        }
    )


def run_chemotaxis_bud_experiment(
    world_factory,
    cfg: ChemotaxisBudConfig,
) -> Dict[str, np.ndarray]:
    """
    Run a small chemotaxis + budding experiment and record metrics.
    Returns arrays and file paths:
      t, births, alive, mean_energy, mean_degree, step_ms, csv_path, npz_path

    The world_factory fixture must accept:
      - field_router=...
      - use_fields=True/False
      - use_neighbors=True/False
      - reproduction_policy=...
    """
    rng = np.random.default_rng(cfg.seed)
    S = 4
    interp = _make_interpreter(state_size=S)

    # Field setup (2D)
    channel = FieldChannel(
        name=cfg.field_name, dim_space=2, sigma=cfg.sigma, decay=cfg.decay
    )
    fr = FieldRouter({cfg.field_name: channel})

    # Build cells
    cells: List[Cell] = []

    # Emitters around origin (or exactly at origin if n_emitters==1)
    for _ in range(cfg.n_emitters):
        if cfg.n_emitters == 1:
            pos = np.array([0.0, 0.0])
        else:
            pos = rng.normal(0, 0.25, size=2)
        cells.append(
            Cell(
                position=pos,
                genome=EmitterContinuous(S, field_key=f"emit_field:{cfg.field_name}"),
                interpreter=interp,
                max_neighbors=0,
                recv_layout={},
                field_layout={f"field:{cfg.field_name}:val": 1},
                energy_init=cfg.energy_init,
                energy_max=cfg.energy_max,
            )
        )

    # Followers on a ring
    radius = 1.5
    for j in range(cfg.n_followers):
        ang = 2.0 * np.pi * j / max(1, cfg.n_followers)
        pos = np.array([radius * np.cos(ang), radius * np.sin(ang)])
        # Declare the follower's field layout once, and derive a layout helper from it.
        follower_field_layout = {f"field:{cfg.field_name}:grad": 2}
        follower_layout = InputLayout.from_dicts({}, follower_field_layout)
        cells.append(
            Cell(
                position=pos,
                genome=FollowerChemotaxisAndBud(
                    state_size=S,
                    field_grad_key=f"field:{cfg.field_name}:grad",
                    grad_gain=cfg.grad_gain,
                    layout=follower_layout,
                ),
                interpreter=interp,
                max_neighbors=0,
                recv_layout={},
                field_layout=follower_field_layout,
                energy_init=cfg.energy_init,
                energy_max=cfg.energy_max,
            )
        )

    # Parent-child link wrapper on top of SimpleBudding
    rp = ParentChildLinkWrapper(
        SimpleBudding(), weight=cfg.link_weight, bidirectional=cfg.bidirectional
    )

    event_logger = EventLogger(cfg.out_dir)

    frame_dumper = FrameDumper()

    world = world_factory(
        cells,
        field_router=fr,
        use_fields=True,
        use_neighbors=False,
        reproduction_policy=rp,
        seed=cfg.seed,
        birth_callback=lambda w, info: event_logger.log_birth(
            w.time,
            info.get("parent").id if info.get("parent") else None,
            info["child"].id,
            info["child"].position,
            info.get("metadata", {}).get("link_weight"),
        ),
    )

    # --- Identify the initial follower cohort (exclude emitters) ------------
    cohort_ids = {
        c.id for c in world.cells if c.genome.__class__.__name__ != "EmitterContinuous"
    }

    # Metrics buffers
    T = int(cfg.steps)
    t = np.arange(T, dtype=int)
    births = np.zeros(T, dtype=int)
    alive = np.zeros(T, dtype=int)
    mean_energy = np.zeros(T, dtype=float)
    mean_degree = np.zeros(T, dtype=float)
    step_ms = np.zeros(T, dtype=float)
    mean_radius = np.zeros(T, dtype=float)

    prev_n = len(world.cells)
    for k in range(T):
        t0 = time.perf_counter()
        world.step()
        step_ms[k] = (time.perf_counter() - t0) * 1000.0

        n = len(world.cells)
        births[k] = max(0, n - prev_n)
        alive[k] = n
        prev_n = n

        # Energy (if present)
        energies: List[float] = []
        degrees: List[int] = []
        radii: List[float] = []
        for c in world.cells:
            e = getattr(c, "energy", np.nan)
            if np.isfinite(e):
                energies.append(float(e))
            co = getattr(c, "conn_out", {}) or {}
            degrees.append(len(co))
            # Track radius for the *initial follower cohort only*.
            # Children are intentionally excluded to avoid bias from bud offsets.
            if c.id in cohort_ids:
                try:
                    r = float(np.linalg.norm(np.asarray(c.position, dtype=float)))
                except Exception:
                    r = float("nan")
                if np.isfinite(r):
                    radii.append(r)
        mean_energy[k] = float(np.mean(energies)) if energies else 0.0
        mean_degree[k] = float(np.mean(degrees)) if degrees else 0.0
        mean_radius[k] = float(np.mean(radii)) if radii else 0.0

        frame_dumper.on_step(world, step_ms[k])

    # Persist results
    arrays = {
        "t": t,
        "births": births,
        "alive": alive,
        "mean_energy": mean_energy,
        "mean_degree": mean_degree,
        "step_ms": step_ms,
        "mean_radius": mean_radius,
    }
    paths = write_metrics_csv_npz(
        cfg.out_dir,
        arrays,
        # Explicit header order including the new metric
        header=(
            "t",
            "births",
            "alive",
            "mean_energy",
            "mean_degree",
            "step_ms",
            "mean_radius",
        ),
    )

    paths["events_csv_path"] = event_logger.write_csv("events.csv")
    paths["frame_dumper_path"] = frame_dumper.write_files(cfg.out_dir)

    return {**arrays, **paths}
