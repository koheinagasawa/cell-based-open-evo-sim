# tests/test_cell_connections_m1.py
import numpy as np

from simulation.cell import Cell


class _ZeroG:
    def activate(self, x):
        return np.zeros(6, float)


def _cells(interpreter4):
    g = _ZeroG()
    a = Cell([0, 0], g, state_size=4, interpreter=interpreter4)
    b = Cell([1, 0], g, state_size=4, interpreter=interpreter4)
    c = Cell([0, 1], g, state_size=4, interpreter=interpreter4)
    return a, b, c


def test_set_and_resolve_connections(interpreter4):
    a, b, c = _cells(interpreter4)
    a.set_connections([(b.id, 0.9), (c.id, 0.2)])
    reg = {x.id: x for x in (a, b, c)}
    pairs = a.connected_pairs(reg)
    ids = [cell.id for cell, _ in pairs]
    wts = [w for _, w in pairs]
    assert ids == [b.id, c.id] and wts == [0.9, 0.2]


def test_default_weight_and_order(interpreter4):
    a, b, c = _cells(interpreter4)
    a.set_connections([c.id, (b.id, 0.5), (c.id, 0.7), (b.id, 0.7)])  # last wins
    ids = a.connected_ids()
    assert ids == sorted([b.id, c.id])  # both 0.7 -> tie by id
