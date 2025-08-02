from simulation.cell import Cell
from simulation.world import World
import numpy as np
import matplotlib.pyplot as plt
import tests.utils.test_utils as tu

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

    # Create cell at origin
    cell = Cell(position=[0, 0], genome=g, state_size=state_size)

    # Create world with just this cell (no neighbors)
    world = World([cell])
    world.step()

    # Check state
    expected_state = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(cell.state, expected_state)

    # Check output_action
    expected_action = [4, 5]
    assert cell.output_action == expected_action

def test_multiple_cells_with_different_output_sizes():
    c1 = Cell(position=[0, 0], genome=DummyGenome(6), state_size=4)  # 4 + 2
    c2 = Cell(position=[10, 0], genome=DummyGenome(7), state_size=4)  # 4 + 3

    world = World([c1, c2])
    world.step()

    assert c1.output_action == [4, 5]
    assert c2.output_action == [4, 5, 6]

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
        time_features = np.array(inputs[-self.time_dim:])

        base = np.array(inputs[0:4])
        offset = np.array([dx + time_features[0], dy + time_features[1]])

        return np.concatenate([base, offset]).tolist()
    
def test_sense_neighbor_cells():
    state_size = 4
    num_steps = 10

    config_dict = {
        "genome": "NullGenome",
        "state_size": state_size,
        "action_size": 2,
        "steps": num_steps
    }

    run_config, recorder = tu.prepare_run(config_dict)

    # Set up cells in fixed positions
    cells = [
        Cell(position=[0, 0], genome=InputDependentGenome(6), state_size=state_size, time_encoding_fn=default_time_encoding),
        Cell(position=[1, 0], genome=InputDependentGenome(6), state_size=state_size, time_encoding_fn=default_time_encoding),
        Cell(position=[-1, 1], genome=InputDependentGenome(6), state_size=state_size, time_encoding_fn=default_time_encoding),
    ]

    world = World(cells)

    # Log output_action over time
    logs = [[] for _ in cells]

    for _ in range(num_steps):
        world.step()
        for i, cell in enumerate(cells):
            logs[i].append(cell.output_action.copy())
            recorder.record(i, cell)

    recorder.save_all()

    # Plot first two dimensions of output_action over time
    for i, log in enumerate(logs):
        x = [step[0] for step in log]
        y = [step[1] for step in log]
        plt.plot(x, y, label=f'Cell {i}')
    plt.title("Output Action (dim 0 vs dim 1) over Time")
    plt.xlabel("Output 0")
    plt.ylabel("Output 1")
    plt.legend()
    plt.grid()
    plt.show()

    tu.plot_state_trajectories(recorder)
    
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
        echo = inputs[6+2 : 6+2+self.state_size] if len(inputs) >= 6+2+self.state_size else [0]*self.state_size
        echo = echo.tolist() if isinstance(echo, np.ndarray) else echo
        return echo + [0] * (self.output_size - self.state_size)
    
def test_multiple_genomes_interaction():
    state_size = 4
    action_size = 2
    output_size = state_size + action_size
    steps = 10

    config_dict = {
        "genome": "Mixed",
        "state_size": state_size,
        "action_size": action_size,
        "steps": steps
    }

    run_config, recorder = tu.prepare_run(config_dict)

    cells = [
        Cell(position=[0, 0], genome=ConstantGenome(output_size), state_size=state_size),
        Cell(position=[1, 0], genome=IncrementGenome(output_size), state_size=state_size),
        Cell(position=[-1, 1], genome=NeighborEchoGenome(output_size, state_size), state_size=state_size),
    ]

    world = World(cells)

    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            dx, dy = cell.output_action[:2]
            cell.position += np.array([dx, dy])
            recorder.record(t, cell)

    recorder.save_all()
    tu.plot_state_trajectories(recorder)

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
        state = np.array([
            np.sin(self.t * 0.1),
            np.cos(self.t * 0.1),
            np.sin(self.t * 0.2),
            np.cos(self.t * 0.2),
        ])[:self.state_size]

        action = np.array([
            np.sin(self.t * 0.2),
            np.cos(self.t * 0.2)
        ])
        self.t += 1
        return np.concatenate([state, action]).tolist()    
    
def test_cells_with_movement():
    state_size = 4
    action_size = 2
    output_size = state_size + action_size
    steps = 20

    config_dict = {
        "genome": "MovementTest",
        "state_size": state_size,
        "action_size": action_size,
        "steps": steps
    }

    run_config, recorder = tu.prepare_run(config_dict)

    cells = [
        Cell(position=[0, 0], genome=RandomMoveGenome(output_size), state_size=state_size),
        Cell(position=[3, 0], genome=OscillatingGenome(output_size), state_size=state_size),
    ]

    world = World(cells)

    for t in range(steps):
        world.step()
        for cell in cells:
            dx, dy = cell.output_action[:2]
            cell.position += np.array([dx, dy])
            recorder.record(t, cell)

    recorder.save_all()
    tu.plot_state_trajectories(recorder)
    tu.plot_position_trajectories(recorder)
