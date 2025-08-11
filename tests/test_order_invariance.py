import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter


class FirstNeighborChaserGenome2D:
    """Move toward the first neighbor's relative position."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def activate(self, inputs):
        # inputs = [self_pos(2), self_state(S), n0_rel(2), n0_state(S), ...]
        dx, dy = inputs[2 + self.state_size : 2 + self.state_size + 2]
        return [0.0] * self.state_size + [dx, dy]


def test_two_phase_is_order_invariant(world_line_factory):
    w1 = world_line_factory(
        order="normal", genome_builder=lambda i: FirstNeighborChaserGenome2D(4, 2)
    )
    w2 = world_line_factory(
        order="reversed", genome_builder=lambda i: FirstNeighborChaserGenome2D(4, 2)
    )
    w1.step()
    w2.step()
    pos1 = np.stack([c.position for c in w1.cells], axis=0)
    pos2 = np.stack([c.position for c in w2.cells], axis=0)
    np.testing.assert_allclose(pos1, pos2, atol=1e-12)


class FirstNeighborChaserGenomeND:
    """Chase first neighbor using the first two relative coords (works for any D>=2)."""

    def __init__(self, state_size, pos_dim=2):
        self.S = state_size
        self.D = pos_dim

    def activate(self, inputs):
        # inputs = [self_pos(D), self_state(S), n0_rel(D), n0_state(S), ...]
        off = self.D + self.S  # start of n0_rel
        dx, dy = inputs[off : off + 2]  # use x,y components only
        return [0.0] * self.S + [dx, dy]


def _interp(S, move_dim):
    return SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + move_dim)})


@pytest.mark.parametrize("D", [2, 3])
def test_two_phase_order_invariance_nd(D, world_factory):
    S = 4
    interp = _interp(S, 2)  # genome outputs move(2); World will pad if D=3

    # Build 3 cells on a line along x; pad zeros for higher D
    def pos(x, D):
        return [x] + [0.0] * (D - 1)

    cells1 = [
        Cell(
            position=pos(0.0, D),
            genome=FirstNeighborChaserGenomeND(S),
            state_size=S,
            interpreter=interp,
        ),
        Cell(
            position=pos(1.0, D),
            genome=FirstNeighborChaserGenomeND(S),
            state_size=S,
            interpreter=interp,
        ),
        Cell(
            position=pos(2.0, D),
            genome=FirstNeighborChaserGenomeND(S),
            state_size=S,
            interpreter=interp,
        ),
    ]
    cells2 = list(reversed([c for c in cells1]))  # reversed order build

    w1, w2 = world_factory(cells1, seed=123), world_factory(cells2, seed=123)
    w1.step()
    w2.step()
    p1 = np.stack([c.position for c in w1.cells])
    p2 = np.stack([c.position for c in w2.cells])

    # Compare order-invariant by sorting on all coordinates lexicographically
    idx1 = np.lexsort(tuple(p1[:, k] for k in reversed(range(D))))
    idx2 = np.lexsort(tuple(p2[:, k] for k in reversed(range(D))))
    np.testing.assert_allclose(p1[idx1], p2[idx2], atol=1e-12)
