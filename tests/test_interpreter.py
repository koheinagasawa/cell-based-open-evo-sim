import numpy as np
import pytest
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter, ProfiledInterpreter

def test_profiled_interpreter_switching():
    """
    Test that ProfiledInterpreter switches between different SlotBasedInterpreters
    based on the 'profile' provided by the cell.
    """
    # 1. Define interpretation rules for different profiles
    
    # Profile A: "Mover" (Uses first 2 outputs for movement)
    # Genome Output: [v0, v1, v2, v3] -> move=[v0, v1]
    interp_mover = SlotBasedInterpreter({
        "state": slice(0, 0), # dummy
        "move": slice(0, 2)
    })

    # Profile B: "Emitter" (Uses last output for field emission)
    # Genome Output: [v0, v1, v2, v3] -> emit_field:pher=[v3]
    interp_emitter = SlotBasedInterpreter({
        "state": slice(0, 0), # dummy
        "move": slice(0, 0), # no move
        "emit_field:pher": 3  # index 3
    })

    # 2. Construct ProfiledInterpreter
    # It switches usage based on the profile key.
    profiles = {
        "mover": interp_mover,
        "emitter": interp_emitter
    }
    # Set default to "mover" for unknown profiles
    interpreter = ProfiledInterpreter(profiles, default_profile="mover")

    # 3. Create a Dummy Genome (Fixed Output)
    # Output: [1.0, 2.0, 3.0, 4.0]
    class FixedGenome:
        def activate(self, inputs, rng=None):
            return np.array([1.0, 2.0, 3.0, 4.0])

    genome = FixedGenome()

    # 4. Verify Case 1: Profile "mover"
    cell_m = Cell(
        position=[0,0], 
        genome=genome, 
        interpreter=interpreter, 
        profile="mover",
        state_size=0
    )
    # Trigger act() which calls interpreter.interpret(..., profile="mover")
    cell_m.act([]) 
    
    # Check: "move" should be [1.0, 2.0]
    assert "move" in cell_m.output_slots
    np.testing.assert_array_equal(cell_m.output_slots["move"], [1.0, 2.0])
    # Check: "emit_field:pher" should NOT be present
    assert "emit_field:pher" not in cell_m.output_slots

    # 5. Verify Case 2: Profile "emitter"
    cell_e = Cell(
        position=[0,0], 
        genome=genome, 
        interpreter=interpreter, 
        profile="emitter",
        state_size=0
    )
    cell_e.act([])

    # Check: "emit_field:pher" should be [4.0] (index 3)
    assert "emit_field:pher" in cell_e.output_slots
    np.testing.assert_array_equal(cell_e.output_slots["emit_field:pher"], [4.0])
    # Check: "move" should be empty (slice(0,0))
    assert "move" in cell_e.output_slots
    assert len(cell_e.output_slots["move"]) == 0

    # 6. Verify Case 3: Unknown Profile (Default Fallback)
    cell_d = Cell(
        position=[0,0], 
        genome=genome, 
        interpreter=interpreter, 
        profile="unknown_profile",
        state_size=0
    )
    cell_d.act([])

    # Check: Should fallback to "mover" (default_profile)
    assert "move" in cell_d.output_slots
    np.testing.assert_array_equal(cell_d.output_slots["move"], [1.0, 2.0])