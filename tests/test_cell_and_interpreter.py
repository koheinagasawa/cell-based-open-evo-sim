import numpy as np

from simulation.cell import Cell
from tests.utils.test_utils import _build_interpreter


# Dummy genome that emits increasing integers to verify interpreter slicing
class DummyGenome:
    def __init__(self, output_size):
        self.output_size = output_size

    def activate(self, inputs):
        return list(range(self.output_size))


def test_cell_step_and_output_action(interpreter4, world_factory):
    """Verify that state/move slices are applied correctly for a single cell."""
    state_size, action_size = 4, 2
    g = DummyGenome(state_size + action_size)
    cell = Cell(
        position=[0, 0], genome=g, state_size=state_size, interpreter=interpreter4
    )
    world = world_factory([cell])
    world.step()

    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)
    # Expected move = [4, 5]
    assert np.array_equal(cell.position, [4, 5])


def test_cell_step_and_output_action_via_config(run_env_factory, world_factory):
    """Build the Interpreter from config (import path + kwargs) and verify slicing works."""
    state_size, action_size = 4, 2

    # 1) Prepare run via config-driven interpreter spec (interpreter-agnostic RunConfig)
    run_config, recorder = run_env_factory(
        {
            "genome": "Dummy",  # just provenance
            "state_size": state_size,
            "action_size": action_size,
            "steps": 1,
            "interpreter": {
                # Fully-qualified import path to the interpreter class
                "class": "simulation.interpreter.SlotBasedInterpreter",
                # JSON-friendly kwargs; slot_defs will be auto-coerced to Python slice
                "kwargs": {
                    "slot_defs": {
                        "state": [0, state_size],
                        "move": [state_size, state_size + action_size],
                    }
                },
            },
        }
    )

    # 2) Instantiate the interpreter from the spec in run_config
    interp = _build_interpreter(run_config.interpreter)

    # 3) Run a single cell with that interpreter
    g = DummyGenome(state_size + action_size)
    cell = Cell(position=[0, 0], genome=g, state_size=state_size, interpreter=interp)
    world = world_factory([cell])
    world.step()

    # 4) Assertions
    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)
    assert np.array_equal(cell.position, [4, 5])  # move slice = [4, 5]


def test_multiple_cells_with_different_output_sizes(
    interpreter4, interpreter4_skip4, world_factory
):
    """Two cells with different interpreter mappings should both move per their mapping."""
    c1 = Cell(
        position=[0, 0], genome=DummyGenome(6), state_size=4, interpreter=interpreter4
    )  # 4 + 2
    c2 = Cell(
        position=[10, 10],
        genome=DummyGenome(7),
        state_size=4,
        interpreter=interpreter4_skip4,
    )  # 4 + 3

    world = world_factory([c1, c2])
    world.step()

    assert np.array_equal(c1.position, [4, 5])
    assert np.array_equal(c2.position, [15, 16])


def test_input_appends_mask_and_count(interpreter4, world_factory):
    class PassthroughGenome:
        """Returns zeros; we only care about sense() layout in this test."""

        def activate(self, inputs):
            return np.zeros(8, dtype=float)

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


def test_vector_path_backcompat(interpreter4):
    vec = np.arange(6, dtype=float)
    out = interpreter4.interpret(vec)
    assert "state" in out


def test_dict_passthrough(interpreter4):
    raw = {
        "state": np.array([1, 2, 3, 4], float),
        "move": np.array([0.1, -0.2], float),
    }
    out = interpreter4.interpret(raw)
    assert set(out.keys()) >= {"state", "move"}
    np.testing.assert_allclose(out["state"], [1, 2, 3, 4])
    np.testing.assert_allclose(out["move"], [0.1, -0.2])


def test_cell_with_keyed_genome(interpreter4, world_factory):
    class DictGenome:
        def __init__(self, S=4):
            self.S = S

        def activate(self, inputs):
            return {
                "state": np.zeros(self.S, dtype=float) + 7.0,  # easy to assert
                "move": np.array([0.05, -0.05], dtype=float),
            }

    g = DictGenome(S=4)
    c = Cell([0, 0], g, state_size=4, interpreter=interpreter4)
    w = world_factory([c])
    w.step()  # two-phase commit applies 'state' after step
    assert np.allclose(c.state, [7, 7, 7, 7])
    # keyed slot is present in output_slots
    assert "move" in getattr(c, "output_slots", {})


def test_interpreter_index_array(interpreter4):
    # Suppose slot_defs maps "foo" -> [0,2,4] and "bar" -> slice(1,3)
    interpreter4.slot_defs["foo"] = [0, 2, 4]
    vec = np.array([10, 11, 12, 13, 14], float)
    out = interpreter4.interpret(vec)
    np.testing.assert_allclose(out["foo"], [10, 12, 14])
