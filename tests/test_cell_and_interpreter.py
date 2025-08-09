
import numpy as np
from simulation.cell import Cell
from simulation.world import World

# Dummy genome that emits increasing integers to verify interpreter slicing
class DummyGenome:
    def __init__(self, output_size):
        self.output_size = output_size
    def activate(self, inputs):
        return list(range(self.output_size))

def test_cell_step_and_output_action(interpreter4):
    """Verify that state/move slices are applied correctly for a single cell."""
    state_size, action_size = 4, 2
    g = DummyGenome(state_size + action_size)
    cell = Cell(position=[0, 0], genome=g, state_size=state_size, interpreter=interpreter4)
    world = World([cell])
    world.step()

    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)
    # Expected move = [4, 5]
    assert np.array_equal(cell.position, [4, 5])

def test_multiple_cells_with_different_output_sizes(interpreter4, interpreter4_skip4):
    """Two cells with different interpreter mappings should both move per their mapping."""
    c1 = Cell(position=[0, 0], genome=DummyGenome(6), state_size=4, interpreter=interpreter4)      # 4 + 2
    c2 = Cell(position=[10, 10], genome=DummyGenome(7), state_size=4, interpreter=interpreter4_skip4)  # 4 + 3

    world = World([c1, c2])
    world.step()

    assert np.array_equal(c1.position, [4, 5])
    assert np.array_equal(c2.position, [15, 16])
