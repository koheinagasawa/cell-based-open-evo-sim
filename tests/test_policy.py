# tests/test_policy_minimal.py
# Verifies: (1) budding via injected policy, (2) energy accounting with global maintenance,
# (3) newborn does not pay maintenance on its birth step.

import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ConstantMaintenance, KillAtZero, SimpleBudding


def test_bud_and_maintenance_minimal(world_factory):
    # --- Build a tiny interpreter with state/move/bud slots ---
    S = 4
    interp = SlotBasedInterpreter(
        {
            "state": slice(0, S),
            "move": slice(S, S + 2),
            "bud": S + 2,  # scalar gate; >0.5 => bud attempt
        }
    )

    # --- Genome that always triggers bud and never moves ---
    class AlwaysBudGenome:
        def __init__(self):
            self.output_size = S + 3

        def activate(self, inputs, rng=None):
            # [state S zeros] + [move (0,0)] + [bud gate = 1.0]
            return [0.0] * S + [0.0, 0.0] + [1.0]

    g = AlwaysBudGenome()

    # Parent starts full energy
    parent = Cell(
        position=[0.0, 0.0],
        genome=g,
        state_size=S,
        interpreter=interp,
        energy_init=1.0,
        energy_max=1.0,
    )

    # Policies: maintenance>0, deterministic bud offset (sigma=0)
    energy_pol = ConstantMaintenance(maintenance=0.1)
    repro_pol = SimpleBudding(
        threshold=0.6, cost=0.5, init_energy=0.4, offset_sigma=0.0
    )

    # World via fixture factory (constructor-injected policies)
    w = world_factory(
        [parent], seed=123, energy_policy=energy_pol, reproduction_policy=repro_pol
    )

    # --- Step 1: bud happens (energy>=threshold); parent pays cost + maintenance ---
    assert len(w.cells) == 1
    w.step()
    assert len(w.cells) == 2

    # Identify parent/newborn by id
    by_id = {c.id: c for c in w.cells}
    par = by_id[parent.id]
    baby = next(c for c in w.cells if c.id != parent.id)

    # Energy after step-1:
    # parent: 1.0 - 0.5 (bud cost) - 0.1 (maintenance) = 0.4
    # baby:   0.4 (init) and pays no maintenance on birth step
    assert np.isclose(par.energy, 0.4)
    assert np.isclose(baby.energy, 0.4)

    # Newborn spawned at parent's position (sigma=0)
    np.testing.assert_allclose(baby.position, par.position, atol=1e-12)

    # --- Step 2: both pay maintenance; no further buds (energies below threshold) ---
    w.step()
    assert np.isclose(by_id[parent.id].energy, 0.3)  # 0.4 - 0.1
    # Rebuild lookup because w.cells may be reordered by add_cell
    by_id2 = {c.id: c for c in w.cells}
    assert np.isclose(by_id2[baby.id].energy, 0.3)
    # Both cells' energy are below the bud gate, so no new cell was born
    assert len(w.cells) == 2


# Validates: parent hits zero energy on Step 1 and is removed; newborn survives and did not pay maintenance.
def test_kill_at_zero_removes_parent_newborn_survives(world_factory):
    S = 4
    interp = SlotBasedInterpreter(
        {
            "state": slice(0, S),
            "move": slice(S, S + 2),
            "bud": S + 2,
        }
    )

    class AlwaysBud:
        def __init__(self):
            self.output_size = S + 3

        def activate(self, inputs, rng=None):
            return [0.0] * S + [0.0, 0.0] + [1.0]

    parent = Cell(
        [0.0, 0.0],
        genome=AlwaysBud(),
        state_size=S,
        interpreter=interp,
        energy_init=1.0,
        energy_max=1.0,
    )

    # Make energy hit exactly zero after Step 1:
    # start 1.0 -> bud cost 0.5 -> maintenance 0.5 -> 0.0
    energy = ConstantMaintenance(maintenance=0.5)
    repro = SimpleBudding(threshold=0.5, cost=0.5, init_energy=0.1, offset_sigma=0.0)

    w = world_factory(
        [parent],
        seed=123,
        energy_policy=energy,
        reproduction_policy=repro,
        lifecycle_policy=KillAtZero(),
    )

    w.step()

    # Parent removed at zero; newborn remains and did NOT pay maintenance on birth step.
    assert len(w.cells) == 1
    baby = w.cells[0]
    assert np.isclose(baby.energy, 0.1)


def test_testonly_actuation_cost(world_factory, interpreter4):
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
