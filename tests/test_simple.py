
import numpy as np
from pathlib import Path
from simulation.cell import Cell
from simulation.world import World
from tests.utils.test_utils import prepare_run

class NullGenome:
    def activate(self, inputs):
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]  # 4 state + 2 zero-move

def test_simple_test_with_output_dir():
    config_dict = {
        "genome": "NullGenome",
        "state_size": 4,
        "action_size": 2,
        "steps": 5
    }

    run_config, recorder = prepare_run(config_dict, commit="core-loop")

    # --- Prepare simulation ---
    genome = NullGenome()
    cell = Cell(position=[0.0, 0.0], genome=genome, state_size=config_dict["state_size"])
    world = World([cell])

    for step in range(config_dict["steps"]):
        world.step()
        dx, dy = cell.output_action[:2]
        cell.position += np.array([dx, dy])
        recorder.record(step, cell)

    recorder.save_all()
