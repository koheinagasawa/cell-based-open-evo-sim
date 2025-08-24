import numpy as np

from simulation.cell import Cell


class _G:
    def activate(self, x):
        return np.zeros(8, dtype=float)


def _sense_vec(c, neighbors):
    x = c.sense(neighbors)
    pos_dim, S, K = c.pos_dim, c.state_size, c.max_neighbors
    base = pos_dim + S
    per_nb = pos_dim + S
    start = base + K * per_nb  # aggregation starts here when enabled
    return x, start, pos_dim, S


def test_agg_mean_simple():
    S, K = 2, 3
    c = Cell(
        [0, 0],
        _G(),
        state_size=S,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
        neighbor_aggregation="mean",
    )
    n1 = Cell([1, 0], _G(), state_size=S)
    n1.state = np.array([2, 4], float)
    n2 = Cell([-1, 0], _G(), state_size=S)
    n2.state = np.array([6, 0], float)
    x, start, D, S = _sense_vec(c, [n1, n2])
    rel_mean = x[start : start + D]
    st_mean = x[start + D : start + D + S]
    np.testing.assert_allclose(rel_mean, [0, 0])
    np.testing.assert_allclose(st_mean, [4, 2])


def test_agg_max_simple():
    S, K = 1, 2
    c = Cell(
        [0, 0],
        _G(),
        state_size=S,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
        neighbor_aggregation="max",
    )
    n1 = Cell([1, -2], _G(), state_size=S)
    n1.state = np.array([0.1])
    n2 = Cell([-3, 5], _G(), state_size=S)
    n2.state = np.array([0.7])
    x, start, D, S = _sense_vec(c, [n1, n2])
    rel_max = x[start : start + D]
    st_max = x[start + D : start + D + S]
    np.testing.assert_allclose(rel_max, [1, 5])  # element-wise max
    np.testing.assert_allclose(st_max, [0.7])
