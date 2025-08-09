from pathlib import Path

import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World
from tests.utils.test_utils import prepare_run


class NullGenome:
    def activate(self, inputs):
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]  # 4 state + 2 zero-move


def test_simple_test_with_output_dir(run_env_factory):
    state_size = 1
    action_size = 1
    steps = 5
    run_config, recorder = run_env_factory(
        {
            "genome": "Mixed",
            "state_size": state_size,
            "action_size": action_size,
            "steps": steps,
        }
    )

    # --- Prepare simulation ---
    genome = NullGenome()
    interpreter = SlotBasedInterpreter(
        {
            "state": slice(0, 1),
        }
    )
    cell = Cell(
        position=[0.0, 0.0],
        genome=genome,
        state_size=state_size,
        interpreter=interpreter,
    )
    world = World([cell])

    for step in range(steps):
        world.step()
        recorder.record(step, cell)

    recorder.save_all()
