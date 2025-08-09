import numpy as np
import pytest

import tests.utils.visualization as visualization
from simulation.cell import Cell
from simulation.world import World


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


def test_sense_neighbor_cells(interpreter4, run_env_factory):
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

    world = World(cells)
    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(i, cell)

    recorder.save_all()
    # Keep plotting disabled for CI speed
    # vis.plot_state_trajectories(recorder, False)
    # vis.plot_2D_position_trajectories(recorder, False)
