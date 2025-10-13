from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Generic machinery (scene-agnostic)
from experiments.common.experiment_spec import (
    ExperimentSpec,
    FieldChannelSpec,
    PopulationSpec,
)
from experiments.common.runner_generic import run_experiment


def _render_gif_from_outdir(out_dir: str, gif_name: str = "field_conn_traj.gif") -> str:
    """Lazy import animator to avoid core depending on tests/."""
    import numpy as np

    from tests.utils.animation_loader import build_frames_from_recorder
    from tests.utils.visualization2d import animate_field_cells_connections

    ff = np.load(os.path.join(out_dir, "field_frames.npy"), allow_pickle=True).tolist()
    ids = np.load(os.path.join(out_dir, "cell_ids.npy"), allow_pickle=True).tolist()
    pos = np.load(os.path.join(out_dir, "cell_pos.npy"), allow_pickle=True).tolist()
    try:
        edges = np.load(os.path.join(out_dir, "edges.npy"), allow_pickle=True).tolist()
    except Exception:
        edges = None

    F, C, E = build_frames_from_recorder(
        field_frames=ff, ids_per_frame=ids, pos_per_frame=pos, edges_per_frame=edges
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, gif_name)
    animate_field_cells_connections(
        out_path=out_path,
        field_frames=F,
        cell_frames=C,
        edge_frames=E,
        fps=15,
        trail_len=40,
        figsize=(6, 6),
        cmap="viridis",
        show_colorbar=True,
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
    populations: List[
        Tuple[int, PosFunc, GenomeFactory]
    ],  # (count, positioner(i)->(x,y), genome_factory)
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
    for count, positioner, genome_factory in populations:
        pops.append(
            PopulationSpec(
                count=int(count),
                positioner=positioner,
                genome_factory=genome_factory,
                recv_layout={},  # keep loose; genomes/interpreter define their own use
                field_layout={},  # override inside genome if needed
                energy_init=1.0,
                energy_max=1.0,
                max_neighbors=0,
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


def run_experiment_quick(
    *,
    out_dir: str,
    steps: int,
    seed: int,
    world_factory: WorldFactory,
    interpreter: InterpreterLike,
    populations: List[Tuple[int, PosFunc, GenomeFactory]],
    policy: Optional[PolicyLike] = None,
    fields: Optional[List[Tuple[str, float, float]]] = None,
    sample_every: int = 1,
    make_gif: bool = True,
    gif_name: str = "field_conn_traj.gif",
    metric_hooks: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run an experiment by passing implementations (scene-agnostic) and optionally render a GIF.
    Returns a dict including arrays and artifact paths under out_dir.
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
    res["out_dir"] = out_dir
    if make_gif:
        res["gif"] = _render_gif_from_outdir(out_dir, gif_name=gif_name)
    return res
