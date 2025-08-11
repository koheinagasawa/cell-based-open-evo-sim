# tests/test_policy_minimal.py
# Verifies: (1) budding via injected policy, (2) energy accounting with global maintenance,
# (3) newborn does not pay maintenance on its birth step.

import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ConstantMaintenance, SimpleBudding


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
