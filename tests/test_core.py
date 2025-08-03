import matplotlib.pyplot as plt
import numpy as np

import tests.utils.test_utils as tu
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World


class DummyGenome:
    def __init__(self, output_size):
        """
        output_size: total number of outputs = state_size + action_size
        """
        self.output_size = output_size

    def activate(self, inputs):
        """
        Returns deterministic outputs: a list of increasing integers.
        This helps verify how the outputs are sliced into state and action.
        """
        return list(range(self.output_size))


def test_cell_step_and_output_action():
    # state_size = 4, action_size = 2 → total output = 6
    state_size = 4
    action_size = 2
    g = DummyGenome(state_size + action_size)

    # Create an interpreter
    interpreter = SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(4, 6),
        }
    )

    # Create cell at origin
    cell = Cell(
        position=[0, 0], genome=g, state_size=state_size, interpreter=interpreter
    )

    # Create world with just this cell (no neighbors)
    world = World([cell])
    world.step()

    # Check state
    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)

    # Check output_action
    expected_action = [4, 5]
    assert np.array_equal(cell.position, expected_action)


def test_multiple_cells_with_different_output_sizes():
    # Create interpreters
    interpreter1 = SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(4, 6),
        }
    )
    interpreter2 = SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(5, 7),  # skip 4
        }
    )

    c1 = Cell(
        position=[0, 0], genome=DummyGenome(6), state_size=4, interpreter=interpreter1
    )  # 4 + 2
    c2 = Cell(
        position=[10, 10], genome=DummyGenome(7), state_size=4, interpreter=interpreter2
    )  # 4 + 3

    world = World([c1, c2])
    world.step()

    assert np.array_equal(c1.position, [4, 5])
    assert np.array_equal(c2.position, [15, 16])


def default_time_encoding(t):
    alphas = [0.5, 0.05, 0.005]  # Periods ≈ 125, 1250, 12500
    return [v for a in alphas for v in (np.sin(t * a), np.cos(t * a))]


class InputDependentGenome:
    def __init__(self, output_size, time_dim=6):
        self.output_size = output_size
        self.time_dim = time_dim

    def activate(self, inputs):
        # Extract first neighbor's relative position
        dx, dy = inputs[6], inputs[7]

        # Extract time features from input tail
        time_features = np.array(inputs[-self.time_dim :])

        base = np.array(inputs[0:4])
        offset = np.array([dx + time_features[0], dy + time_features[1]])

        return np.concatenate([base, offset]).tolist()


def test_sense_neighbor_cells():
    state_size = 4
    num_steps = 10

    config_dict = {
        "genome": "InputDependentGenome",
        "state_size": state_size,
        "action_size": 2,
        "steps": num_steps,
    }

    run_config, recorder = tu.prepare_run(config_dict)

    # Create an interpreter
    interpreter = SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(4, 6),
        }
    )

    # Set up cells in fixed positions
    cells = [
        Cell(
            position=[0, 0],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter,
            time_encoding_fn=default_time_encoding,
        ),
        Cell(
            position=[1, 0],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter,
            time_encoding_fn=default_time_encoding,
        ),
        Cell(
            position=[-1, 1],
            genome=InputDependentGenome(6),
            state_size=state_size,
            interpreter=interpreter,
            time_encoding_fn=default_time_encoding,
        ),
    ]

    world = World(cells)

    for _ in range(num_steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(i, cell)

    recorder.save_all()
    tu.plot_state_trajectories(recorder, False)
    tu.plot_2D_position_trajectories(recorder, False)


class ConstantGenome:
    def __init__(self, output_size, value=1.0):
        self.output = [value] * output_size

    def activate(self, inputs):
        return self.output


class IncrementGenome:
    def __init__(self, output_size):
        self.output_size = output_size
        self.counter = 0

    def activate(self, inputs):
        out = [(self.counter + i) % 10 for i in range(self.output_size)]
        self.counter += 1
        return out


class NeighborEchoGenome:
    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size

    def activate(self, inputs):
        # Echo first neighbor's state if available, else zero
        echo = (
            inputs[4 : 4 + self.state_size]
            if len(inputs) >= 4 + self.state_size
            else [0] * self.state_size
        )
        echo = echo.tolist() if isinstance(echo, np.ndarray) else echo
        return echo + [0] * (self.output_size - self.state_size)


def test_multiple_genomes_interaction():
    state_size = 2
    action_size = 2
    output_size = state_size + action_size
    steps = 10

    config_dict = {
        "genome": "Mixed",
        "state_size": state_size,
        "action_size": action_size,
        "steps": steps,
    }

    run_config, recorder = tu.prepare_run(config_dict)

    # Create an interpreter
    interpreter = SlotBasedInterpreter(
        {
            "state": slice(0, 2),
            "move": slice(2, 4),
        }
    )

    cells = [
        Cell(
            position=[0, 0],
            genome=ConstantGenome(output_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
        Cell(
            position=[1, 0],
            genome=IncrementGenome(output_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
        Cell(
            position=[-1, 1],
            genome=NeighborEchoGenome(output_size, state_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
    ]

    world = World(cells)

    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(t, cell)

    recorder.save_all()
    tu.plot_state_trajectories(recorder, False)
    tu.plot_2D_position_trajectories(recorder, False)


class RandomMoveGenome:
    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size

    def activate(self, inputs):
        state = np.random.uniform(-1, 1, self.state_size)
        action = np.random.uniform(-0.5, 0.5, self.output_size - self.state_size)
        return np.concatenate([state, action]).tolist()


class OscillatingGenome:
    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size
        self.t = 0

    def activate(self, inputs):
        state = np.array(
            [
                np.sin(self.t * 0.1),
                np.cos(self.t * 0.1),
                np.sin(self.t * 0.2),
                np.cos(self.t * 0.2),
            ]
        )[: self.state_size]

        action = np.array([np.sin(self.t * 0.2), np.cos(self.t * 0.2)])
        self.t += 1
        return np.concatenate([state, action]).tolist()


# Uses internal state memory to generate directional movement (e.g. drift)
class DirectionalMemoryGenome:
    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size

    def activate(self, inputs):
        # Keep internal state increasing, use it for movement
        memory = np.array(inputs[2 : 2 + self.state_size])
        new_state = memory + 1
        direction = new_state[:2] * 0.1
        return np.concatenate([new_state, direction]).tolist()


def test_cells_with_movement():
    state_size = 4
    action_size = 2
    output_size = state_size + action_size
    steps = 20

    config_dict = {
        "genome": "MovementTest",
        "state_size": state_size,
        "action_size": action_size,
        "steps": steps,
    }

    run_config, recorder = tu.prepare_run(config_dict)

    # Create an interpreter
    interpreter = SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(4, 6),
        }
    )

    cells = [
        Cell(
            position=[0, 0],
            genome=RandomMoveGenome(output_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
        Cell(
            position=[3, 0],
            genome=OscillatingGenome(output_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
        Cell(
            position=[0, 3],
            genome=DirectionalMemoryGenome(output_size),
            state_size=state_size,
            interpreter=interpreter,
        ),
    ]

    world = World(cells)

    for t in range(steps):
        world.step()
        for cell in cells:
            recorder.record(t, cell)

    recorder.save_all()
    tu.plot_state_trajectories(recorder, True)
    tu.plot_2D_position_trajectories(recorder, True)
