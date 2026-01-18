from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

# Generic machinery (scene-agnostic)
from experiments.common.experiment_spec import (
    ExperimentSpec,
    FieldChannelSpec,
    PopulationSpec,
)
from experiments.common.runner_generic import run_experiment

try:
    from tests.utils.visualization2d import animate_field_cells_connections
except ImportError:
    animate_field_cells_connections = None


def _render_gif_from_outdir(out_dir: str, gif_name: str = "field_conn_traj.gif") -> str:
    """Lazy import animator to avoid core depending on tests/."""
    import json

    import numpy as np

    from tests.utils.animation_loader import build_frames_from_recorder
    from tests.utils.visualization2d import animate_field_cells_connections

    field_frames = np.load(
        os.path.join(out_dir, "field_frames.npy"), allow_pickle=True
    ).tolist()
    cell_ids = np.load(
        os.path.join(out_dir, "cell_ids.npy"), allow_pickle=True
    ).tolist()
    cell_pos = np.load(
        os.path.join(out_dir, "cell_pos.npy"), allow_pickle=True
    ).tolist()
    try:
        edges = np.load(os.path.join(out_dir, "edges.npy"), allow_pickle=True).tolist()
    except Exception:
        edges = None
    cell_profiles_map: Dict[str, str] = {}
    prof_path = os.path.join(out_dir, "cell_profiles.npy")

    if os.path.exists(prof_path):
        # profiles_per_frame: shape (T,), each element is List[Any] matching cell_ids[t]
        profiles_frames = np.load(prof_path, allow_pickle=True)

        # Aggregate profiles from all frames into a single lookup dict
        # (Assuming a cell's profile does not change over time)
        for t in range(len(cell_ids)):
            ids_t = cell_ids[t]
            profs_t = profiles_frames[t]

            if ids_t is None or profs_t is None:
                continue

            # zip handles equal length; check length safely if needed
            for cid, prof in zip(ids_t, profs_t):
                if prof is not None and cid not in cell_profiles_map:
                    cell_profiles_map[cid] = str(prof)

    # Try to read field metadata
    field_extent = None
    try:
        with open(os.path.join(out_dir, "field_metadata.json"), "r") as f:
            meta = json.load(f)
            field_extent = tuple(meta["bounds"])
    except Exception:
        pass

    F, C, E, view_range = build_frames_from_recorder(
        field_frames=field_frames,
        ids_per_frame=cell_ids,
        pos_per_frame=cell_pos,
        edges_per_frame=edges,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, gif_name)

    # Choose colormap: 'tab10' for categorical profiles, 'viridis' for IDs
    cmap_to_use = "tab10" if cell_profiles_map else "viridis"

    animate_field_cells_connections(
        out_path=out_path,
        field_frames=F,
        cell_frames=C,
        edge_frames=E,
        view_range=view_range,
        fps=15,
        trail_len=40,
        figsize=(6, 6),
        cmap=cmap_to_use,
        cell_profiles=cell_profiles_map if cell_profiles_map else None,
        show_colorbar=True,
        field_extent=field_extent,
    )
    return out_path


# ---------- Scene-agnostic Quick API ----------
InterpreterLike = Union[Callable[[], Any], Any]
GenomeFactory = Callable[[], Any]
WorldFactory = Callable[..., Any]
PolicyLike = Union[Callable[[], Any], Any]
PosFunc = Callable[[int], Tuple[float, float]]


def _as_factory(x):
    """Return a 0-arg callable: identity for callables, lambda for instances."""
    return x if callable(x) else (lambda: x)


def build_spec_quick(
    *,
    out_dir: str,
    steps: int,
    seed: int,
    world_factory: WorldFactory,
    interpreter: InterpreterLike,
    populations: List[Any],
    policy: Optional[PolicyLike] = None,
    fields: Optional[List[Tuple[str, float, float]]] = None,  # [(name, sigma, decay)]
    sample_every: int = 1,
    log_events: bool = True,
    metric_hooks: Optional[list] = None,
) -> ExperimentSpec:
    """Construct a scene-agnostic ExperimentSpec from implementations."""
    field_specs = []
    if fields:
        for name, sigma, decay in fields:
            field_specs.append(
                FieldChannelSpec(name=name, sigma=float(sigma), decay=float(decay))
            )

    pops: List[PopulationSpec] = []
    for item in populations:
        # Support both tuple and dict styles.
        if isinstance(item, tuple):
            # (count, positioner(i)->(x,y), genome_factory)
            count, positioner, genome_factory = item
            recv_layout = {}
            field_layout = {}
            max_neighbors = 0
            energy_init = 1.0
            energy_max = 1.0
        elif isinstance(item, dict):
            count = item["count"]
            positioner = item["positioner"]
            genome_factory = item["genome_factory"]
            recv_layout = dict(item.get("recv_layout", {}))
            field_layout = dict(item.get("field_layout", {}))
            max_neighbors = int(item.get("max_neighbors", 0))
            energy_init = float(item.get("energy_init", 1.0))
            energy_max = float(item.get("energy_max", 1.0))
        else:
            raise TypeError("Each population must be a tuple or dict.")

        pops.append(
            PopulationSpec(
                count=int(count),
                positioner=positioner,
                genome_factory=genome_factory,
                recv_layout=recv_layout,
                field_layout=field_layout,
                energy_init=energy_init,
                energy_max=energy_max,
                max_neighbors=max_neighbors,
            )
        )

    interpreter_factory = _as_factory(interpreter)
    policy_factory = _as_factory(policy) if policy is not None else None

    return ExperimentSpec(
        out_dir=out_dir,
        steps=int(steps),
        seed=int(seed),
        world_factory=world_factory,
        interpreter_factory=interpreter_factory,
        field_channels=field_specs,
        populations=pops,
        policy_factory=policy_factory,
        use_fields=bool(fields),
        use_neighbors=False,
        dump_frames=True,  # ensure NPYs for gif
        sample_every=int(sample_every),  # integer counter in dumper
        log_events=log_events,
        metric_hooks=list(metric_hooks or []),
    )


class PopulationQuickDict(TypedDict, total=False):
    """Dictionary-style population definition for quick experiments."""

    count: int
    positioner: PosFunc
    genome_factory: GenomeFactory
    recv_layout: Dict[str, int]
    field_layout: Dict[str, int]
    max_neighbors: int
    energy_init: float
    energy_max: float


PopulationQuick = Union[
    Tuple[int, PosFunc, GenomeFactory],  # legacy tuple style
    PopulationQuickDict,  # extended dict style
]


def run_experiment_quick(
    *,
    out_dir: str,
    steps: int,
    seed: int,
    world_factory: WorldFactory,
    interpreter: InterpreterLike,
    populations: List[PopulationQuick],
    policy: Optional[PolicyLike] = None,
    fields: Optional[List[Tuple[str, float, float]]] = None,
    sample_every: int = 1,
    make_gif: bool = True,
    gif_name: str = "field_conn_traj.gif",
    metric_hooks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run an experiment by passing implementations (scene-agnostic) and optionally render a GIF.

    populations supports two formats:
      1) Tuple style: (count, positioner, genome_factory)
         - recv_layout/field_layout/max_neighbors are defaulted to {} / {} / 0.
      2) Dict style:
         {
           "count": int,
           "positioner": (i)->(x,y),
           "genome_factory": ()->Genome,
           "recv_layout": {...},        # optional
           "field_layout": {...},       # optional
           "max_neighbors": int,        # optional
           "energy_init": float,        # optional
           "energy_max": float,         # optional
         }
    """
    spec = build_spec_quick(
        out_dir=out_dir,
        steps=steps,
        seed=seed,
        world_factory=world_factory,
        interpreter=interpreter,
        populations=populations,
        policy=policy,
        fields=fields,
        sample_every=sample_every,
        log_events=True,
        metric_hooks=metric_hooks,
    )

    res = run_experiment(spec)

    # Always trust spec.out_dir (future-proof if out_dir becomes optional/auto-generated)
    res["out_dir"] = spec.out_dir

    if make_gif:
        res["gif"] = _render_gif_from_outdir(spec.out_dir, gif_name=gif_name)

    return res
