import numpy as np

from simulation.cell import Cell
from simulation.world import World
from tests.utils.test_utils import _build_interpreter


# Dummy genome that emits increasing integers to verify interpreter slicing
class DummyGenome:
    def __init__(self, output_size):
        self.output_size = output_size

    def activate(self, inputs):
        return list(range(self.output_size))


def test_cell_step_and_output_action(interpreter4):
    """Verify that state/move slices are applied correctly for a single cell."""
    state_size, action_size = 4, 2
    g = DummyGenome(state_size + action_size)
    cell = Cell(
        position=[0, 0], genome=g, state_size=state_size, interpreter=interpreter4
    )
    world = World([cell])
    world.step()

    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)
    # Expected move = [4, 5]
    assert np.array_equal(cell.position, [4, 5])


def test_cell_step_and_output_action_via_config(run_env_factory):
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
    world = World([cell])
    world.step()

    # 4) Assertions
    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)
    assert np.array_equal(cell.position, [4, 5])  # move slice = [4, 5]


def test_multiple_cells_with_different_output_sizes(interpreter4, interpreter4_skip4):
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

    world = World([c1, c2])
    world.step()

    assert np.array_equal(c1.position, [4, 5])
    assert np.array_equal(c2.position, [15, 16])
