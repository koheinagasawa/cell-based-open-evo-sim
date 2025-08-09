import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World


# Use a genome that consumes rng
class RandomWalkGenome:
    def __init__(self, state_size=4, sigma=0.5):
        self.state_size = state_size
        self.sigma = sigma
        self.rng = None

    def activate(self, inputs, rng=None):
        g = rng or self.rng
        dx, dy = g.normal(0.0, self.sigma, size=2) if g else (0.0, 0.0)
        return [0.0] * self.state_size + [dx, dy]


def _interp(S=4):
    return SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + 2)})


def _make_world(seed, order="normal"):
    S = 4
    interp = SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + 2)})
    g = RandomWalkGenome(state_size=S, sigma=0.5)

    assert interp.slot_defs["state"].start == 0 and interp.slot_defs["state"].stop == S
    assert (
        interp.slot_defs["move"].start == S and interp.slot_defs["move"].stop == S + 2
    )

    cells = [
        # Always use keyword args to avoid positional mistakes
        Cell(position=[0.0, 0.0], genome=g, state_size=S, interpreter=interp),
        Cell(position=[1.0, 0.0], genome=g, state_size=S, interpreter=interp),
        Cell(position=[2.0, 0.0], genome=g, state_size=S, interpreter=interp),
    ]
    if order == "reversed":
        cells = list(reversed(cells))
    return World(cells, seed=seed)


def _stack_positions_sorted_by_xy(world):
    """Return positions stacked and sorted by x then y for order-invariant compare."""
    import numpy as np

    arr = np.stack([c.position for c in world.cells])  # (N, 2)
    idx = np.lexsort((arr[:, 1], arr[:, 0]))  # sort by x, then y
    return arr[idx]


def test_same_seed_same_result_regardless_of_order():
    w1 = _make_world(1234, "normal")
    w2 = _make_world(1234, "reversed")
    w1.step()
    w2.step()
    p1 = _stack_positions_sorted_by_xy(w1)
    p2 = _stack_positions_sorted_by_xy(w2)
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_different_seed_different_result():
    w1 = _make_world(1111)
    w2 = _make_world(2222)
    w1.step()
    w2.step()
    assert not np.allclose(
        np.stack([c.position for c in w1.cells]),
        np.stack([c.position for c in w2.cells]),
    )
