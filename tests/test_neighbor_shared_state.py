import numpy as np

from simulation.cell import Cell


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


# ------------------------------ Tests ---------------------------------------


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
