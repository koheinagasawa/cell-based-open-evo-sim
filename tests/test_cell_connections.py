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


class ParamEmitGenome:
    """Emit arbitrary keyed vectors via {'emit:<k>': vec} and keep state zero."""

    def __init__(self, emits: dict[str, np.ndarray], S: int = 4):
        self.S = int(S)
        # store copies as float ndarrays
        self.emits = {
            f"emit:{k}": np.asarray(v, dtype=float).ravel() for k, v in emits.items()
        }

    def activate(self, inputs):
        out = {"state": np.zeros(self.S, dtype=float), "move": np.zeros(2, dtype=float)}
        out.update(self.emits)
        return out


def test_multi_sources_aggregate_and_dim_guard(world_factory, interpreter4):
    S = 4

    # Sender A: emit a=[1,2], b=[5]
    gA = ParamEmitGenome(emits={"a": [1.0, 2.0], "b": [5.0]}, S=S)
    A = Cell([0, 0], gA, state_size=S, interpreter=interpreter4)

    # Sender C: emit a=[10,20,30] (longer), b=[7,8] (shorter than recv dim we will declare)
    gC = ParamEmitGenome(emits={"a": [10.0, 20.0, 30.0], "b": [7.0, 8.0]}, S=S)
    C = Cell([1, 0], gC, state_size=S, interpreter=interpreter4)

    # Receiver B expects recv:a (2 dims) and recv:b (3 dims).
    # We also turn off legacy spatial tails to keep indexing simple.
    B = Cell(
        [2, 0],
        genome=ParamEmitGenome(emits={}, S=S),  # dummy receiver genome
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": 2, "recv:b": 3},
    )
    # Reduce legacy spatial features noise (optional; safe defaults in your setup may already do this)
    B.max_neighbors = 0
    B.include_neighbor_mask = False
    B.include_num_neighbors = False

    # Wire: A -> B (w=0.5), C -> B (w=0.1)
    A.set_connections([(B.id, 0.5)])
    C.set_connections([(B.id, 0.1)])

    router = MessageRouter()
    w = world_factory(
        [A, C, B],
        energy_policy=ConstantMaintenance(0.0),
        reproduction_policy=SimpleBudding(),
        message_router=router,
    )

    # Step once: route emit(t) -> B.inbox(t+1)
    w.step()

    # Build B's sense input; the last 2+3 entries are recv:a (2) then recv:b (3) by key sort order.
    xB = B.sense([])  # neighbors unused
    tail = xB[-(2 + 3) :]
    recv_a = tail[:2]
    recv_b = tail[2:]

    # Expected:
    # recv:a dim=2: 0.5*[1,2] + 0.1*truncate([10,20,30]->[10,20]) = [0.5,1] + [1,2] = [1.5, 3.0]
    np.testing.assert_allclose(recv_a, [1.5, 3.0], atol=1e-12)

    # recv:b dim=3: 0.5*pad([5]->[5,0,0]) + 0.1*pad([7,8]->[7,8,0]) = [2.5,0,0] + [0.7,0.8,0] = [3.2,0.8,0]
    np.testing.assert_allclose(recv_b, [3.2, 0.8, 0.0], atol=1e-12)


class CopyRecvGenome:
    """
    Genome that:
      - emits current self_state as 'emit:<key>'
      - sets next state to the received 'recv:<key>' vector (read from input tail)
    Assumes sense() builds inputs as:
      [ self_pos(D), self_state(S), ... (legacy spatial stuff) ..., recv:<key>(R) ]
    Here we use R == S and a single recv key for simplicity.
    """

    def __init__(self, S: int, D: int = 2, key: str = "a"):
        self.S = int(S)
        self.D = int(D)
        self.key = str(key)

    def activate(self, inputs):
        x = np.asarray(inputs, dtype=float).ravel()
        # Parse self_state from the known slot after position
        self_state = x[self.D : self.D + self.S]
        # Read recv:<key> from the tail (dimension == S)
        recv = x[-self.S :] if self.S > 0 else np.zeros(0, dtype=float)
        # Outputs:
        return {
            "state": recv.copy(),  # next state = received vector
            "move": np.zeros(2, dtype=float),  # no movement
            f"emit:{self.key}": self_state.copy(),  # emit own state
        }


def _mk_world(cells, world_factory):
    return world_factory(
        cells,
        energy_policy=ConstantMaintenance(0.0),
        reproduction_policy=SimpleBudding(),
        message_router=MessageRouter(),
    )


def test_connected_copy_two_cells_swap(interpreter4, world_factory):
    """
    Two cells A <-> B (weight 1.0). Each emits its own state and copies recv.
    With two-phase routing, after two steps states should swap.
    """
    S, D = 4, 2
    g = CopyRecvGenome(S=S, D=D, key="a")

    A = Cell(
        [-1.0, 0.0],
        g,
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": S},
    )
    B = Cell(
        [+1.0, 0.0],
        g,
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": S},
    )

    A.state = np.array([1, 2, 3, 4], dtype=float)
    B.state = np.array([10, 20, 30, 40], dtype=float)

    # Wire both ways, weight 1.0
    A.set_connections([(B.id, 1.0)])
    B.set_connections([(A.id, 1.0)])

    w = _mk_world([A, B], world_factory)

    # Step 1: emit routed into inbox(t+1). States unchanged yet.
    w.step()
    np.testing.assert_allclose(A.state, [0, 0, 0, 0])
    np.testing.assert_allclose(B.state, [0, 0, 0, 0])

    # Step 2: each copies recv (the other's previous state) -> swap
    w.step()
    np.testing.assert_allclose(A.state, [10, 20, 30, 40])
    np.testing.assert_allclose(B.state, [1, 2, 3, 4])


def test_connected_average_three_cells(interpreter4, world_factory):
    """
    Three cells fully connected (no self loops). Each receiver has 2 incoming.
    Set each edge weight = 0.5 so that the sum equals the mean of the two sources.
    After two steps, each state's next value becomes the mean of the other two's previous states.
    """
    S, D = 4, 2
    g = CopyRecvGenome(S=S, D=D, key="a")

    C1 = Cell(
        [-1.0, 0.0],
        g,
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": S},
    )
    C2 = Cell(
        [+1.0, 0.0],
        g,
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": S},
    )
    C3 = Cell(
        [0.0, +1.0],
        g,
        state_size=S,
        interpreter=interpreter4,
        recv_layout={"recv:a": S},
    )

    # Distinct initial states (one-hot-ish on first 3 dims)
    C1.state = np.array([1, 0, 0, 0], dtype=float)
    C2.state = np.array([0, 1, 0, 0], dtype=float)
    C3.state = np.array([0, 0, 1, 0], dtype=float)

    # Fully connect (no self): each incoming weight 0.5 so sum == mean
    C1.set_connections([(C2.id, 0.5), (C3.id, 0.5)])
    C2.set_connections([(C1.id, 0.5), (C3.id, 0.5)])
    C3.set_connections([(C1.id, 0.5), (C2.id, 0.5)])

    w = _mk_world([C1, C2, C3], world_factory)

    # Step 1: only routing happens
    w.step()

    # Step 2: states become the mean of the other two's previous states
    w.step()
    np.testing.assert_allclose(C1.state, [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(C2.state, [0.5, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(C3.state, [0.5, 0.5, 0.0, 0.0])
