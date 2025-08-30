from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class ConstantMaintenance:
    """Global per-step basal metabolism.
    Set maintenance=0.0 to start with no passive drain.
    """

    maintenance: float = 0.0

    def per_step(self, cell) -> float:
        # Could be made cell-dependent later if needed.
        return float(self.maintenance)


@dataclass(frozen=True)
class SimpleBudding:
    """Minimal budding policy with a scalar threshold and cost."""

    threshold: float = 0.6  # min energy required to attempt budding
    cost: float = 0.5  # energy paid by parent at bud time
    init_energy: float = 0.4  # newborn initial energy
    offset_sigma: float = 0.2  # jitter if no offset is provided
    init_child: Optional[Callable[["Cell", "Cell", "World"], None]] = None

    def apply(self, world, parent, value, spawn_fn):
        """Interpret 'value' and spawn a single offspring if conditions hold.

        Accepted shapes:
          - scalar: gate in (0..1); offset sampled ~ N(0, sigma)
          - D-vector: treated as offset; gate=1.0
          - [gate, *offset(D)]: explicit gate and offset
        """
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        D = int(parent.position.shape[0])

        # Parse gate and offset
        if arr.size == 1:
            gate = float(arr[0])
            rng = getattr(parent, "rng", None)
            sigma = float(self.offset_sigma)
            offset = (
                rng.normal(0.0, sigma, size=D)
                if rng is not None
                else np.zeros(D, dtype=float)
            )
        elif arr.size == D:
            gate, offset = 1.0, arr[:D]
        else:
            gate = float(arr[0])
            offset = arr[1 : 1 + D]

        # Check energy/threshold
        if gate <= 0.5:
            return
        if float(parent.energy) < float(self.threshold):
            return

        # Pay cost and spawn
        parent.energy = max(0.0, float(parent.energy) - float(self.cost))

        Baby = parent.__class__
        baby = Baby(
            position=(parent.position + offset).tolist(),
            genome=parent.genome,
            state_size=parent.state_size,
            interpreter=parent.interpreter,
            # Inherit energy cap; newborn gets init_energy
            energy_init=float(self.init_energy),
            energy_max=float(getattr(parent, "energy_max", 1.0)),
        )
        # RNG will be attached by the world; newborn does not pay maintenance this frame.
        spawn_fn(baby)

        # Call optional birth hook (after child exists, before step ends)
        if self.init_child is not None:
            self.init_child(baby, parent, world)


@dataclass(frozen=True)
class NoDeath:
    """Lifecycle: never blocks acting; never removes."""

    def can_act(self, cell) -> bool:
        return True

    def should_remove(self, cell) -> bool:
        return False


@dataclass(frozen=True)
class KillAtZero:
    """Lifecycle: blocks acting at energy<=0 and removes at energy<=0."""

    def can_act(self, cell) -> bool:
        return cell.energy > 0.0

    def should_remove(self, cell) -> bool:
        return cell.energy <= 0.0
