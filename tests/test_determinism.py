import numpy as np
import pytest

from experiments.chemotaxis_bud.config import ChemotaxisBudConfig
from experiments.chemotaxis_bud.runner import make_spec
from experiments.common.runner_generic import run_experiment
from simulation.cell import Cell
from simulation.fields import FieldChannel, FieldRouter
from simulation.input_layout import InputLayout
from simulation.interpreter import SlotBasedInterpreter


def test_two_phase_is_order_invariant(world_line_factory):
    class FirstNeighborChaserGenome2D:
        """Move toward the first neighbor's relative position."""

        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size

        def activate(self, inputs):
            # inputs = [self_pos(2), self_state(S), n0_rel(2), n0_state(S), ...]
            dx, dy = inputs[2 + self.state_size : 2 + self.state_size + 2]
            return [0.0] * self.state_size + [dx, dy]

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


@pytest.mark.parametrize("D", [2, 3])
def test_two_phase_order_invariance_nd(D, world_factory):
    def _interp(S, move_dim):
        return SlotBasedInterpreter(
            {"state": slice(0, S), "move": slice(S, S + move_dim)}
        )

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


def _build_random_walk_cells(order="normal"):
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
    return cells if order == "normal" else list(reversed(cells))


def _stack_positions_sorted_by_xy(world):
    """Return positions stacked and sorted by x then y for order-invariant compare."""
    import numpy as np

    arr = np.stack([c.position for c in world.cells])  # (N, 2)
    idx = np.lexsort((arr[:, 1], arr[:, 0]))  # sort by x, then y
    return arr[idx]


def test_same_seed_same_result_regardless_of_order(world_factory):
    w1 = world_factory(_build_random_walk_cells("normal"), seed=1234)
    w2 = world_factory(_build_random_walk_cells("reversed"), seed=1234)
    w1.step()
    w2.step()
    p1 = _stack_positions_sorted_by_xy(w1)
    p2 = _stack_positions_sorted_by_xy(w2)
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_different_seed_different_result(world_factory):
    w1 = world_factory(_build_random_walk_cells(), seed=1111)
    w2 = world_factory(_build_random_walk_cells(), seed=2222)
    w1.step()
    w2.step()
    assert not np.allclose(
        np.stack([c.position for c in w1.cells]),
        np.stack([c.position for c in w2.cells]),
    )


def test_tail_order_is_key_sorted(world_factory, interpreter4):
    # Prepare router & channel but we only care about *placement* in inputs
    fr = FieldRouter(
        {
            "b": FieldChannel(name="b", dim_space=2),
            "a": FieldChannel(name="a", dim_space=2),
        }
    )

    class Idle:
        def activate(self, inputs):
            return {"state": np.zeros(4), "move": np.zeros(2)}

    c = Cell(
        [0.0, 0.0],
        Idle(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={"recv:z": 3, "recv:a": 2},
        field_layout={"field:b:val": 1, "field:a:grad": 2},
    )
    w = world_factory([c], field_router=fr, use_fields=True)

    # Manually preload inbox and field_inputs to known values
    c.inbox["recv:a"] = np.array([10, 11])
    c.inbox["recv:z"] = np.array([20, 21, 22])
    # field inputs are populated by world.step() -> sample_into_cell
    w.step()  # fills field_inputs (zeros initially)

    x = c.sense([])  # K=0, tail appended in key-sorted order
    layout = InputLayout.from_cell(c)
    tail = layout.split_tail(x)

    # Keys must appear sorted by key string
    assert list(tail.keys()) == ["recv:a", "recv:z", "field:a:grad", "field:b:val"]

    # And the values must match what we loaded (recv) or zeros (field)
    assert np.allclose(tail["recv:a"], [10, 11])
    assert np.allclose(tail["recv:z"], [20, 21, 22])
    assert tail["field:a:grad"].shape == (2,) and np.allclose(
        tail["field:a:grad"], [0.0, 0.0]
    )
    assert tail["field:b:val"].shape == (1,) and float(tail["field:b:val"][0]) == 0.0


def test_fields_only_determinism(world_factory, interpreter4):
    # Simple emitter-follower; check positions are identical across two runs.
    fr1 = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=1.0, decay=0.95)}
    )
    fr2 = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=1.0, decay=0.95)}
    )

    class Emitter:
        def activate(self, inputs):
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:pher": [1.0]}

    class Follower:
        def __init__(self, layout):
            self.layout = layout

        def activate(self, inputs):
            grad = self.layout.get_vector(inputs, "field:pher:grad")
            return {"state": np.zeros(4), "move": grad}

    field_layout = {"field:pher:grad": 2}

    A1 = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout=field_layout,
    )
    B1 = Cell(
        [1.0, 0.0],
        Follower(layout=InputLayout.from_dicts({}, field_layout)),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout=field_layout,
    )

    A2 = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:grad": 2},
    )
    B2 = Cell(
        [1.0, 0.0],
        Follower(layout=InputLayout.from_dicts({}, field_layout)),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout=field_layout,
    )

    w1 = world_factory(
        [A1, B1], field_router=fr1, use_fields=True, use_neighbors=False, seed=123
    )
    w2 = world_factory(
        [A2, B2], field_router=fr2, use_fields=True, use_neighbors=False, seed=123
    )

    for _ in range(8):
        w1.step()
        w2.step()

    assert np.allclose(B1.position, B2.position)


def test_experiment_determinism(world_factory, test_output_dir):
    cfg = ChemotaxisBudConfig(
        steps=40,
        n_emitters=1,
        n_followers=10,
        seed=123,
        sigma=1.0,
        decay=0.95,
        grad_gain=1.0,
        link_weight=0.8,
        bidirectional=False,
        out_dir=str(test_output_dir / "det_run_a"),
    )
    spec = make_spec(cfg, world_factory)
    res_a = run_experiment(spec)

    # Run again with the same config to a different directory
    cfg.out_dir = str(test_output_dir / "det_run_b")
    res_b = run_experiment(spec)

    # These arrays must be identical bitwise (exclude step_ms which is wall time)
    keys = ["t", "births", "alive", "mean_energy", "mean_degree", "mean_radius"]
    for k in keys:
        assert np.array_equal(res_a[k], res_b[k]), f"Mismatch in key '{k}'"

    # Causal sanity: on steps where births occur, mean_degree should be non-decreasing
    births = res_a["births"]
    deg = res_a["mean_degree"]
    for i in range(1, len(births)):
        if births[i] > 0:
            assert deg[i] >= deg[i - 1] - 1e-12
