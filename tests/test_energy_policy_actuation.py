# tests/test_energy_policy_actuation_like.py
import numpy as np

from simulation.cell import Cell
from simulation.policies import SimpleBudding


class _MoveGenome:
    # Emits: [state(S)=0...] + move=[3,4]  (just for example)
    def __init__(self, S=4):
        self.S = S

    def activate(self, x):
        out = np.zeros(self.S + 2, dtype=float)
        out[self.S : self.S + 2] = [3.0, 4.0]
        return out


class ActuationEnergyPolicy:
    """
    Test-only energy policy.
    Subtracts L2 norm of selected output slots (e.g., 'move') times a scale,
    then applies a constant maintenance drain. This lives ONLY in tests.
    """

    def __init__(
        self, scale: float = 0.1, maintenance: float = 0.0, slot_names=("move",)
    ):
        self.scale = float(scale)
        self.maintenance = float(maintenance)
        self.slot_names = tuple(slot_names)

    def per_step(self, cell):
        slots = getattr(cell, "output_slots", None) or {}
        cost = 0.0
        for k in self.slot_names:
            v = slots.get(k)
            if v is not None:
                vv = np.asarray(v, dtype=float)
                cost += float(np.sqrt(np.dot(vv, vv)))  # L2
        return self.scale * cost + self.maintenance


def test_testonly_actuation_cost(world_factory, interpreter4):
    c = Cell([0, 0], _MoveGenome(), state_size=4, interpreter=interpreter4)
    c.energy = 1.0
    w = world_factory(
        [c],
        energy_policy=ActuationEnergyPolicy(scale=0.1, maintenance=0.0),
        reproduction_policy=SimpleBudding(),  # unused here
    )
    w.step()
    # L2([3,4])=5 → cost=0.5 → E=0.5
    assert abs(c.energy - 0.5) < 1e-6
