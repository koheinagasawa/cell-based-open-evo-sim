from pathlib import Path

import numpy as np

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World
from tests.utils.test_utils import prepare_run


class NullGenome:
    def activate(self, inputs):
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]  # 4 state + 2 zero-move


def test_simple_test_with_output_dir():
    config_dict = {
        "genome": "NullGenome",
        "state_size": 1,
        "action_size": 1,
        "steps": 5,
    }

    run_config, recorder = prepare_run(config_dict, commit="core-loop")

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
        state_size=config_dict["state_size"],
        interpreter=interpreter,
    )
    world = World([cell])

    for step in range(config_dict["steps"]):
        world.step()
        recorder.record(step, cell)

    recorder.save_all()
