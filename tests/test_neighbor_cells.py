import numpy as np

from simulation.cell import Cell


def test_sense_neighbor_cells(interpreter4, run_env_factory, world_factory):
    def default_time_encoding(t):
        """Small multi-period sinusoidal time features."""
        alphas = [0.5, 0.05, 0.005]  # periods ~ 12.6, 125.6, 1256.6 (scaled)
        return [v for a in alphas for v in (np.sin(t * a), np.cos(t * a))]

    class InputDependentGenome:
        def __init__(self, output_size, time_dim=6):
            self.output_size = output_size
            self.time_dim = time_dim

        def activate(self, inputs):
            # Extract first neighbor's relative position
            dx, dy = inputs[6], inputs[7]
            # Time features come from the tail
            time_features = np.array(inputs[-self.time_dim :])
            base = np.array(inputs[0:4])
            offset = np.array([dx + time_features[0], dy + time_features[1]])
            return np.concatenate([base, offset]).tolist()

    state_size = 4
    steps = 10

    run_config, recorder = run_env_factory(
        {
            "genome": "InputDependentGenome",
            "state_size": state_size,
            "action_size": 2,
            "steps": steps,
        }
    )

    cells = [
        Cell(
            position=[0, 0],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter4,
            time_encoding_fn=default_time_encoding,
        ),
        Cell(
            position=[1, 0],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter4,
            time_encoding_fn=default_time_encoding,
        ),
        Cell(
            position=[-1, 1],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter4,
            time_encoding_fn=default_time_encoding,
        ),
    ]

    world = world_factory(cells)
    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(i, cell)

    recorder.save_all()
    # Keep plotting disabled for CI speed
    # vis.plot_state_trajectories(recorder, False)
    # vis.plot_2D_position_trajectories(recorder, False)


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


# --- Test-only Genome: uses neighbor "state" as a shared memory -------------
class NeighborSharedStateGenome:
    """
    A minimal genome that reads neighbor states from the fixed sense layout
    and writes the (copy/mean/max) of those states back into its own 'state' slot.

    It does NOT require any cell-level 'tag' field; everything lives in 'state'.
    This keeps the design aligned with "state as shared memory".
    """

    def __init__(
        self,
        state_size: int,
        pos_dim: int,
        max_neighbors: int,
        mode: str = "mean",
        action_out_dim: int = 2,
    ):
        self.S = int(state_size)
        self.D = int(pos_dim)
        self.K = int(max_neighbors)
        self.mode = str(mode).lower()  # 'copy'|'mean'|'max'
        self.action_out_dim = int(action_out_dim)

    def activate(self, inputs):
        x = np.asarray(inputs, dtype=float)

        # Input layout assumed (no mask/count/time for simplicity in tests):
        # [ self_pos(D), self_state(S),
        #   n0_relpos(D), n0_state(S),
        #   n1_relpos(D), n1_state(S),
        #   ... up to K neighbors (zero-padded if absent) ]
        base = self.D + self.S
        per_nb = self.D + self.S

        nb_states = []
        for i in range(self.K):
            off = base + i * per_nb
            nb_st = x[off + self.D : off + self.D + self.S]
            nb_states.append(nb_st)

        nb_states = np.stack(nb_states, axis=0) if len(nb_states) > 0 else None

        if nb_states is None:
            target = np.zeros(self.S, dtype=float)
        else:
            if self.mode == "copy":
                # Copy the first neighbor's state (with K=1 this is pure imitation)
                target = nb_states[0]
            elif self.mode == "max":
                # Element-wise max across neighbors' states
                target = nb_states.max(axis=0)
            else:
                # Default: arithmetic mean across neighbors' states
                target = nb_states.mean(axis=0)

        # Build full output: [state(S), move(2) ...]
        out = np.zeros(self.S + self.action_out_dim, dtype=float)
        out[: self.S] = target
        return out


def test_shared_state_copy_two_cells(interpreter4, world_factory):
    """
    Two cells, each sees exactly one neighbor (K=1).
    'copy' mode makes them imitate the neighbor's *previous* state,
    so after one step their states swap.
    """
    S, D, K = 4, 2, 1
    g = NeighborSharedStateGenome(state_size=S, pos_dim=D, max_neighbors=K, mode="copy")

    a = Cell(
        position=[-1.0, 0.0],
        genome=g,
        state_size=S,
        interpreter=interpreter4,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
    )
    b = Cell(
        position=[+1.0, 0.0],
        genome=g,
        state_size=S,
        interpreter=interpreter4,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
    )

    # Initialize states to distinct values
    a.state = np.array([1, 2, 3, 4], dtype=float)
    b.state = np.array([10, 20, 30, 40], dtype=float)

    w = world_factory([a, b])  # zero maintenance & no reproduction in this fixture
    w.step()

    # After one step, each has copied the other's previous state
    np.testing.assert_allclose(a.state, [10, 20, 30, 40])
    np.testing.assert_allclose(b.state, [1, 2, 3, 4])


def test_shared_state_average_three_cells(interpreter4, world_factory):
    """
    Three cells, each sees the other two (K=2).
    'mean' mode makes a cell's next state become the average of neighbors' states.
    """
    S, D, K = 4, 2, 2
    g = NeighborSharedStateGenome(state_size=S, pos_dim=D, max_neighbors=K, mode="mean")

    c1 = Cell(
        position=[-1.0, 0.0],
        genome=g,
        state_size=S,
        interpreter=interpreter4,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
    )
    c2 = Cell(
        position=[+1.0, 0.0],
        genome=g,
        state_size=S,
        interpreter=interpreter4,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
    )
    c3 = Cell(
        position=[0.0, +1.0],
        genome=g,
        state_size=S,
        interpreter=interpreter4,
        max_neighbors=K,
        include_neighbor_mask=False,
        include_num_neighbors=False,
    )

    # Distinct initial states (one-hot-ish on first 3 dims)
    c1.state = np.array([1, 0, 0, 0], dtype=float)
    c2.state = np.array([0, 1, 0, 0], dtype=float)
    c3.state = np.array([0, 0, 1, 0], dtype=float)

    w = world_factory([c1, c2, c3])
    w.step()

    # Each cell averages the other two's previous states
    np.testing.assert_allclose(
        c1.state, 0.5 * (c2.state * 0 + np.array([0, 1, 0, 0]) + np.array([0, 0, 1, 0]))
    )
    np.testing.assert_allclose(
        c2.state, 0.5 * (np.array([1, 0, 0, 0]) + np.array([0, 0, 1, 0]))
    )
    np.testing.assert_allclose(
        c3.state, 0.5 * (np.array([1, 0, 0, 0]) + np.array([0, 1, 0, 0]))
    )

    # Alternatively, assert by values directly:
    np.testing.assert_allclose(c1.state, [0, 0.5, 0.5, 0])
    np.testing.assert_allclose(c2.state, [0.5, 0, 0.5, 0])
    np.testing.assert_allclose(c3.state, [0.5, 0.5, 0, 0])
