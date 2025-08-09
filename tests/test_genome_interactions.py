import numpy as np
import pytest

import tests.utils.test_utils as tu
from simulation.cell import Cell
from simulation.world import World


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


def test_multiple_genomes_interaction(interpreter_factory, run_env_factory):
    state_size, action_size = 2, 2
    output_size, steps = state_size + action_size, 10

    run_config, recorder = run_env_factory(
        {
            "genome": "Mixed",
            "state_size": state_size,
            "action_size": action_size,
            "steps": steps,
        }
    )

    # Use an interpreter consistent with (2 + 2)
    interp = interpreter_factory(state_size=state_size, action_size=action_size)

    cells = [
        Cell(
            position=[0, 0],
            genome=ConstantGenome(output_size),
            state_size=state_size,
            interpreter=interp,
        ),
        Cell(
            position=[1, 0],
            genome=IncrementGenome(output_size),
            state_size=state_size,
            interpreter=interp,
        ),
        Cell(
            position=[-1, 1],
            genome=NeighborEchoGenome(output_size, state_size),
            state_size=state_size,
            interpreter=interp,
        ),
    ]

    world = World(cells)
    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(t, cell)

    recorder.save_all()


# --- A second, slower integration-style test ---


class RandomMoveGenome:
    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size

    def activate(self, inputs):
        # NOTE: Replace with injected RNG (cell.rng) when available
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


class DirectionalMemoryGenome:
    """Uses its prior state (from inputs) to bias direction."""

    def __init__(self, output_size, state_size=4):
        self.output_size = output_size
        self.state_size = state_size

    def activate(self, inputs):
        # Read self-state from inputs (pos(2) + state(S) ...)
        memory = np.array(inputs[2 : 2 + self.state_size])
        new_state = memory + 1.0
        direction = new_state[:2] * 0.1
        return np.concatenate([new_state, direction]).tolist()


def test_multiple_genomes_interaction2(interpreter4, run_env_factory):
    state_size, action_size = 4, 2
    output_size, steps = state_size + action_size, 20

    run_config, recorder = run_env_factory(
        {
            "genome": "MovementTest",
            "state_size": state_size,
            "action_size": action_size,
            "steps": steps,
        }
    )

    cells = [
        Cell(
            position=[0, 0],
            genome=RandomMoveGenome(output_size),
            state_size=state_size,
            interpreter=interpreter4,
        ),
        Cell(
            position=[3, 0],
            genome=OscillatingGenome(output_size),
            state_size=state_size,
            interpreter=interpreter4,
        ),
        Cell(
            position=[0, 3],
            genome=DirectionalMemoryGenome(output_size),
            state_size=state_size,
            interpreter=interpreter4,
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
