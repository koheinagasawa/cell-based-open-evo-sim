from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from experiments.common.event_logger import EventLogger
from experiments.common.experiment_spec import ExperimentSpec
from experiments.common.frame_dumper import FrameDumper
from experiments.common.metrics import write_metrics_csv_npz
from experiments.common.metrics_hook import MetricsHook
from simulation.cell import Cell
from simulation.fields import FieldChannel, FieldRouter
from simulation.policies import ConstantMaintenance
from simulation.world import World


def world_factory() -> Callable[..., World]:
    """Return a factory function that builds a World with optional overrides.

    Usage:
        w = world_factory(
            [cell],
            seed=123,
            energy_policy=DummyEnergyPolicy(0.0),
            reproduction_policy=DummyBudPolicy(),
        )
    """

    def _factory(
        cells: Sequence,
        *,
        seed: int = 0,
        actions: Optional[Dict[str, Callable]] = None,
        message_router=None,
        energy_policy: Any = None,
        reproduction_policy: Any = None,
        lifecycle_policy: Any = None,
        use_neighbors: bool = True,
        field_router=None,
        use_fields: bool = False,
        birth_callback: Optional[Callable[[World, Dict[str, Any]], None]] = None,
        field_added_callback: Optional[Callable[[World, Dict[str, Any]], None]] = None,
    ) -> World:
        return World(
            cells,
            seed=seed,
            actions=actions or {},
            message_router=message_router,
            energy_policy=energy_policy or ConstantMaintenance(0.0),
            reproduction_policy=reproduction_policy,
            lifecycle_policy=lifecycle_policy,
            use_neighbors=use_neighbors,
            field_router=field_router,
            use_fields=use_fields,
            birth_callback=birth_callback,
            field_added_callback=field_added_callback,
        )

    return _factory


def run_experiment(spec: ExperimentSpec) -> Dict[str, np.ndarray]:
    """Run a generic experiment described by ExperimentSpec and record artifacts."""
    rng = np.random.default_rng(spec.seed)

    # Interpreter
    interp = spec.interpreter_factory()

    # Fields
    channels = {
        fc.name: FieldChannel(fc.name, fc.dim_space, fc.sigma, fc.decay)
        for fc in spec.field_channels
    }
    field_router = FieldRouter(channels) if channels else None

    # Instantiate hooks
    hooks: List[MetricsHook] = list(spec.metric_hooks) if spec.metric_hooks else []

    # NOTE: hooks.begin(world) is called later after world creation.
    # But currently runner structure creates world below.
    # We will pass 'world' to hooks.begin() after creation.

    # Build initial cells
    cells: List[Cell] = []
    for pop in spec.populations:
        for i in range(pop.count):
            x, y = pop.positioner(i)
            cells.append(
                Cell(
                    position=np.array([float(x), float(y)], dtype=float),
                    genome=pop.genome_factory(),
                    interpreter=interp,
                    max_neighbors=pop.max_neighbors,
                    recv_layout=pop.recv_layout,
                    field_layout=pop.field_layout,
                    energy_init=pop.energy_init,
                    energy_max=pop.energy_max,
                )
            )

    # Reproduction policy
    reproduction_policy = spec.policy_factory() if spec.policy_factory else None

    # Event logger / frame dumper
    os.makedirs(spec.out_dir, exist_ok=True)
    event_logger = EventLogger(spec.out_dir) if spec.log_events else None
    frame_dumper = (
        FrameDumper(sample_every=spec.sample_every) if spec.dump_frames else None
    )

    # Callbacks for logging
    birth_cb = None
    field_cb = None
    if event_logger:
        birth_cb = lambda w, info: event_logger.log_birth(
            w.time,
            info.get("parent").id if info.get("parent") else None,
            info["child"].id,
            info["child"].position,
            info.get("metadata", {}).get("link_weight"),
        )
        field_cb = lambda w, info: event_logger.log_field_add(
            w.time,
            info["cell"].id,
            info["pos"],
            info["field_name"],
            info["sigma"],
            info["decay"],
        )

    # Build world (contract: accepts these kwargs if relevant)
    world = spec.world_factory(
        cells,
        field_router=field_router,
        use_fields=spec.use_fields,
        use_neighbors=spec.use_neighbors,
        reproduction_policy=reproduction_policy,
        seed=spec.seed,
        birth_callback=birth_cb,
        field_added_callback=field_cb,
    )

    # Initialize hooks
    for h in hooks:
        try:
            h.begin(world)
        except Exception:
            pass

    # Metrics buffers
    T = int(spec.steps)
    t = np.arange(T, dtype=int)
    births = np.zeros(T, dtype=int)
    alive = np.zeros(T, dtype=int)
    mean_energy = np.zeros(T, dtype=float)
    mean_degree = np.zeros(T, dtype=float)
    step_ms = np.zeros(T, dtype=float)

    # Extra metrics (from hooks): name -> list[float]
    extra: Dict[str, list] = {}

    prev_n = len(world.cells)
    for k in range(T):
        t0 = time.perf_counter()
        world.step()
        step_ms[k] = (time.perf_counter() - t0) * 1000.0

        n = len(world.cells)
        births[k] = max(0, n - prev_n)
        alive[k] = n
        prev_n = n

        energies: List[float] = []
        degrees: List[int] = []
        radii: List[float] = []
        for c in world.cells:
            e = getattr(c, "energy", np.nan)
            if np.isfinite(e):
                energies.append(float(e))
            co = getattr(c, "conn_out", {}) or {}
            degrees.append(len(co))
        mean_energy[k] = float(np.mean(energies)) if energies else 0.0
        mean_degree[k] = float(np.mean(degrees)) if degrees else 0.0

        # Hooks
        for h in hooks:
            try:
                vals = h.on_step(world, k) or {}
            except Exception:
                vals = {}
            for name, v in vals.items():
                extra.setdefault(name, []).append(float(v))

        if frame_dumper:
            frame_dumper.on_step(world, step_ms[k])  # uses internal integer counter

    # Finalize hooks (optional)
    for h in hooks:
        try:
            h.end()
        except Exception:
            pass

    # Normalize extra metrics to arrays of length T (pad missing with nan)
    extra_arrays = {}
    for name, seq in extra.items():
        arr = np.full(T, np.nan, dtype=float)
        L = min(T, len(seq))
        if L > 0:
            arr[:L] = np.asarray(seq[:L], dtype=float)
        extra_arrays[name] = arr

    # Persist metrics + events + frames
    arrays = dict(
        t=t,
        births=births,
        alive=alive,
        mean_energy=mean_energy,
        mean_degree=mean_degree,
        step_ms=step_ms,
        **extra_arrays,
    )
    paths = write_metrics_csv_npz(
        spec.out_dir,
        arrays,
        header=tuple(arrays.keys()),
    )

    if event_logger:
        # returns dict of paths
        evt_paths = event_logger.write_csv("events.csv")
        paths.update(evt_paths)
    if frame_dumper:
        paths["frame_dumper_path"] = frame_dumper.write_files(spec.out_dir)

    return {**arrays, **paths}
