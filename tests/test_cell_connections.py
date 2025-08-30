# tests/test_cell_connections_m1.py
import numpy as np

from simulation.cell import Cell
from simulation.messaging import MessageRouter
from simulation.policies import ConstantMaintenance, SimpleBudding


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


class EmitGenome:
    """Writes 'emit:a' with a fixed 2D vector; keeps state unchanged."""

    def __init__(self, S=4, v=(1.0, 2.0)):
        self.S = S
        self.v = np.array(v, float)

    def activate(self, inputs):
        # state(S) zeros + move(2) zeros (adapt to your interpreter schema)
        out = np.zeros(self.S + 2, float)
        return {"state": out[: self.S], "move": out[self.S :], "emit:a": self.v}


class ReadGenome:
    """Copies recv:a into state[0:2] for easy assertion."""

    def __init__(self, S=4):
        self.S = S

    def activate(self, inputs):
        # inputs tail contains recv:a (2 dim) per recv_layout
        # Dummy: state = zeros except first 2 set to a small marker; we will check via sense route.
        return {"state": np.zeros(self.S, float), "move": np.zeros(2, float)}


def test_emit_route_recv(world_factory, interpreter4):
    S = 4
    g_src = EmitGenome(S=S, v=(3.0, 4.0))
    g_dst = ReadGenome(S=S)

    a = Cell(
        [0, 0], g_src, state_size=S, interpreter=interpreter4, recv_layout={}
    )  # sender doesn't need recv
    b = Cell(
        [1, 0], g_dst, state_size=S, interpreter=interpreter4, recv_layout={"recv:a": 2}
    )  # receiver expects 2 dims

    # A -> B with weight 0.5
    a.set_connections([(b.id, 0.5)])

    router = MessageRouter()
    w = world_factory(
        [a, b],
        message_router=router,
        energy_policy=ConstantMaintenance(0.0),
        reproduction_policy=SimpleBudding(),
    )

    # step 1: a emits, routed to b.next_inbox; then inbox swap
    w.step()

    # step 2: b.sense() should include recv:a == 0.5 * [3,4] at the tail;
    # we check by directly building sense() and inspecting the tail slice.
    x_b = b.sense([])  # neighbors unused in this test
    tail = x_b[-2:]
    np.testing.assert_allclose(tail, [1.5, 2.0])
