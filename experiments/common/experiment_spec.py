# Purpose: pure data to describe an experiment; runner_generic consumes this.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.common.metrics_hook import MetricsHook

# --- Types are intentionally loose to avoid tight coupling ---
InterpreterFactory = Callable[[], Any]
GenomeFactory = Callable[[], Any]
WorldFactory = Callable[..., Any]  # must accept keyword args used below
PolicyFactory = Callable[[], Any]


@dataclass
class FieldChannelSpec:
    name: str
    dim_space: int = 2
    sigma: float = 1.0
    decay: float = 0.0


@dataclass
class PopulationSpec:
    """How to seed a group of cells."""

    count: int
    positioner: Callable[[int], Tuple[float, float]]  # i -> (x,y)
    genome_factory: GenomeFactory
    recv_layout: Dict[str, int] = field(default_factory=dict)
    field_layout: Dict[str, int] = field(default_factory=dict)
    energy_init: float = 1.0
    energy_max: float = 1.0
    max_neighbors: int = 0


@dataclass
class ExperimentSpec:
    out_dir: str
    steps: int
    seed: int
    world_factory: WorldFactory
    interpreter_factory: InterpreterFactory
    field_channels: List[FieldChannelSpec] = field(default_factory=list)
    populations: List[PopulationSpec] = field(default_factory=list)
    policy_factory: Optional[PolicyFactory] = (
        None  # e.g., ParentChildLinkWrapper(SimpleBudding)
    )
    use_fields: bool = True
    use_neighbors: bool = False
    dump_frames: bool = True
    sample_every: int = 1
    log_events: bool = True
    metric_hooks: List[MetricsHook] = field(default_factory=list)
