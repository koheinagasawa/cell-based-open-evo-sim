import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter


class MoveXYOnly:
    """Always moves +1 in x, +2 in y; no z component provided."""

    def __init__(self, state_size):
        self.S = state_size

    def activate(self, inputs):
        return [0.0] * self.S + [1.0, 2.0]  # move len=2


def test_2d_move_in_3d_world_keeps_z(world_factory):
    S = 4
    D = 3
    interp = SlotBasedInterpreter(
        {"state": slice(0, S), "move": slice(S, S + 2)}
    )  # move(2)
    cell = Cell(
        position=[0.0, 0.0, 5.0], genome=MoveXYOnly(S), state_size=S, interpreter=interp
    )
    w = world_factory([cell], seed=0)
    w.step()
    # Expect x+=1, y+=2, z unchanged
    np.testing.assert_allclose(cell.position, np.array([1.0, 2.0, 5.0]), atol=1e-12)
