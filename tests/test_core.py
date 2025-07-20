from simulation.cell import Cell
from simulation.world import World
import numpy as np

class DummyGenome:
    def __init__(self, output_size):
        """
        output_size: total number of outputs = state_size + action_size
        """
        self.output_size = output_size

    def activate(self, inputs):
        """
        Returns deterministic outputs: a list of increasing integers.
        This helps verify how the outputs are sliced into state and action.
        """
        return list(range(self.output_size))

def test_cell_step_and_output_action():
    # state_size = 4, action_size = 2 → total output = 6
    state_size = 4
    action_size = 2
    g = DummyGenome(state_size + action_size)

    # Create cell at origin
    cell = Cell(position=[0, 0], genome=g, state_size=state_size)

    # Create world with just this cell (no neighbors)
    world = World([cell])
    world.step()

    # Check state
    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)

    # Check output_action
    expected_action = [4, 5]
    assert cell.output_action == expected_action

def test_multiple_cells_with_different_output_sizes():
    c1 = Cell(position=[0, 0], genome=DummyGenome(6), state_size=4)  # 4 + 2
    c2 = Cell(position=[10, 0], genome=DummyGenome(7), state_size=4)  # 4 + 3

    world = World([c1, c2])
    world.step()

    assert c1.output_action == [4, 5]
    assert c2.output_action == [4, 5, 6]

