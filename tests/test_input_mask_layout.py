import numpy as np

from simulation.cell import Cell


class PassthroughGenome:
    """Returns zeros; we only care about sense() layout in this test."""

    def activate(self, inputs):
        return np.zeros(8, dtype=float)


def test_input_appends_mask_and_count(interpreter4, world_factory):
    # 2D, S=4, K=3 neighbors
    S, K = 4, 3
    cells = [
        Cell(
            [0.0, 0.0],
            PassthroughGenome(),
            state_size=S,
            interpreter=interpreter4,
            max_neighbors=K,
        ),
        Cell(
            [1.0, 0.0],
            PassthroughGenome(),
            state_size=S,
            interpreter=interpreter4,
            max_neighbors=K,
        ),
        Cell(
            [2.0, 0.0],
            PassthroughGenome(),
            state_size=S,
            interpreter=interpreter4,
            max_neighbors=K,
        ),
    ]
    w = world_factory(cells)
    neighbors = w.get_neighbors(cells[1], radius=10.0)
    x = cells[1].sense(neighbors)

    pos_dim = 2
    base = pos_dim + S
    per_nb = pos_dim + S
    expected_min = base + K * per_nb  # padded blocks
    # time features unknown here -> ignore; we only assert the final tail exists
    assert len(x) >= expected_min + K + 1  # +mask(K) +num_neighbors(1)

    mask = x[-(K + 1) : -1]
    num = int(x[-1])
    # There are 2 neighbors within radius in this line (left & right)
    assert num == 2
    assert list(mask)[:2] == [1.0, 1.0]
    assert list(mask)[2] == 0.0
