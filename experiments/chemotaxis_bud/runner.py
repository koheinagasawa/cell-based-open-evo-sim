# experiments/chemotaxis_bud/metrics_hooks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.genomes import (
    EmitterContinuous,
    FollowerChemotaxisAndBud,
)
from experiments.common.experiment_spec import (
    ExperimentSpec,
    FieldChannelSpec,
    PopulationSpec,
)
from experiments.common.metrics_hook import MetricsHook
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ParentChildLinkWrapper, SimpleBudding

CenterFunc = Callable[[object], Tuple[float, float]]


class MeanRadiusHook(MetricsHook):
    """
    Compute mean distance of all cells from a given center per step.
    Center can be a fixed point or a function of world (e.g., emitter centroid).
    """

    def __init__(
        self,
        center: Optional[Tuple[float, float]] = (0.0, 0.0),
        center_func: Optional[CenterFunc] = None,
        name: str = "mean_radius",
    ):
        self.center = center
        self.center_func = center_func
        self.name = name

    def _get_center(self, world) -> Tuple[float, float]:
        if self.center_func is not None:
            try:
                cx, cy = self.center_func(world)
                return float(cx), float(cy)
            except Exception:
                pass
        if self.center is not None:
            return float(self.center[0]), float(self.center[1])
        # fallback: origin
        return 0.0, 0.0

    def begin(self, world) -> None:
        pass

    def on_step(self, world, step_index: int) -> Dict[str, float]:
        cells = (
            getattr(world, "cells", None) or getattr(world, "get_cells", lambda: None)()
        )
        if not cells:
            return {self.name: 0.0}
        cx, cy = self._get_center(world)
        dists = []
        for c in cells:
            p = np.asarray(getattr(c, "position"), dtype=float).ravel()
            x = float(p[0])
            y = float(p[1] if p.size > 1 else 0.0)
            d = np.hypot(x - cx, y - cy)
            if np.isfinite(d):
                dists.append(d)
        val = float(np.mean(dists)) if dists else 0.0
        return {self.name: val}

    def end(self):
        return None


def make_spec(cfg: ChemotaxisBudConfig, world_factory) -> ExperimentSpec:
    S = cfg.state_size

    def interpreter_factory():
        # state: [0:S), move: [S:S+2), bud: [S+2]
        return SlotBasedInterpreter(
            {"state": slice(0, S), "move": slice(S, S + 2), "bud": S + 2}
        )

    # Emitters around origin (if >1, add small noise)
    def emitter_pos(i: int) -> Tuple[float, float]:
        if cfg.n_emitters == 1:
            return (0.0, 0.0)
        rng = np.random.default_rng(cfg.seed + i)
        xy = rng.normal(0, 0.25, size=2)
        return float(xy[0]), float(xy[1])

    # Followers on a ring
    R = 1.5

    def follower_pos(i: int) -> Tuple[float, float]:
        ang = 2.0 * np.pi * i / max(1, cfg.n_followers)
        return float(R * np.cos(ang)), float(R * np.sin(ang))

    follower_field_layout = {f"field:{cfg.field_name}:grad": 2}

    pops = []
    pops.append(
        PopulationSpec(
            count=cfg.n_emitters,
            positioner=emitter_pos,
            genome_factory=lambda: EmitterContinuous(
                S, field_key=f"emit_field:{cfg.field_name}"
            ),
            recv_layout={},
            field_layout={f"field:{cfg.field_name}:val": 1},
            energy_init=cfg.energy_init,
            energy_max=cfg.energy_max,
            max_neighbors=0,
        )
    )
    pops.append(
        PopulationSpec(
            count=cfg.n_followers,
            positioner=follower_pos,
            genome_factory=lambda: FollowerChemotaxisAndBud(
                state_size=S,
                field_grad_key=f"field:{cfg.field_name}:grad",
                grad_gain=cfg.grad_gain,
                layout=None,  # genomes internally create layout if needed
            ),
            recv_layout={},
            field_layout=follower_field_layout,
            energy_init=cfg.energy_init,
            energy_max=cfg.energy_max,
            max_neighbors=0,
        )
    )

    def policy_factory():
        return ParentChildLinkWrapper(
            SimpleBudding(), weight=cfg.link_weight, bidirectional=cfg.bidirectional
        )

    mean_r = MeanRadiusHook(center=(0.0, 0.0), name="mean_radius")

    return ExperimentSpec(
        out_dir=cfg.out_dir,
        steps=cfg.steps,
        seed=cfg.seed,
        world_factory=world_factory,
        interpreter_factory=interpreter_factory,
        field_channels=[
            FieldChannelSpec(name=cfg.field_name, sigma=cfg.sigma, decay=cfg.decay)
        ],
        populations=pops,
        policy_factory=policy_factory,
        use_fields=True,
        use_neighbors=False,
        dump_frames=True,
        sample_every=cfg.sample_every,
        log_events=True,
        metric_hooks=[mean_r],
    )
