import os
from dataclasses import dataclass


@dataclass
class ChemotaxisBudConfig:
    steps: int = 200
    n_emitters: int = 1
    n_followers: int = 20
    seed: int = 0
    # Field channel params
    field_name: str = "pher"
    sigma: float = 1.0
    decay: float = 0.95
    # Parent-child link
    link_weight: float = 0.8
    bidirectional: bool = False
    # Movement gain (scale gradient)
    grad_gain: float = 1.0
    # Initial energy given to every cell (if used by policies)
    energy_init: float = 1.0
    energy_max: float = 10.0
    # Output directory
    out_dir: str | os.PathLike = "runs/chemotaxis_bud"
