import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import ProfiledInterpreter, SlotBasedInterpreter


class ConstantGenome:
    def __init__(self, output_vector):
        self.output_vector = np.array(output_vector, dtype=float)

    def activate(self, inputs):
        return self.output_vector


def test_profile_switches_interpretation():
    # 1. Setup Interpreters for two profiles: "mover" and "emitter"
    # Genome output size = 4.
    # Mover uses indices [0,1] for move. Emitter uses [2,3] for emit.
    # Both use [0,4] (overlapping) for state, just for complexity.

    mover_interp = SlotBasedInterpreter(
        {"state": slice(0, 2), "move": slice(0, 2)}  # Uses first 2 dims as move
    )

    emitter_interp = SlotBasedInterpreter(
        {"state": slice(0, 2), "emit_field:A": slice(2, 4)}  # Uses last 2 dims as emit
    )

    # 2. Create ProfiledInterpreter
    p_interp = ProfiledInterpreter(
        profiles={"mover": mover_interp, "emitter": emitter_interp},
        default_profile="mover",
    )

    # 3. Create Genome outputting [1.0, 1.0, 5.0, 5.0]
    # Mover should see move=[1,1]. Emitter should see emit=[5,5].
    genome = ConstantGenome([1.0, 1.0, 5.0, 5.0])

    # 4. Create Cells with different profiles
    cell_m = Cell([0, 0], genome, interpreter=p_interp, profile="mover", state_size=2)
    cell_e = Cell([0, 0], genome, interpreter=p_interp, profile="emitter", state_size=2)
    cell_d = Cell(
        [0, 0], genome, interpreter=p_interp, profile="unknown", state_size=2
    )  # Default -> mover

    # 5. Step (Act)
    # We can call act() directly with dummy inputs
    cell_m.act([])
    cell_e.act([])
    cell_d.act([])

    # 6. Verify Output Slots
    # Mover
    assert "move" in cell_m.output_slots
    assert "emit_field:A" not in cell_m.output_slots
    np.testing.assert_array_equal(cell_m.output_slots["move"], [1.0, 1.0])

    # Emitter
    assert "move" not in cell_e.output_slots
    assert "emit_field:A" in cell_e.output_slots
    np.testing.assert_array_equal(cell_e.output_slots["emit_field:A"], [5.0, 5.0])

    # Default (should be Mover)
    assert "move" in cell_d.output_slots
    np.testing.assert_array_equal(cell_d.output_slots["move"], [1.0, 1.0])
