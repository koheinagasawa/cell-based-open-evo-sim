import numpy as np

import tests.utils.visualization2d as visualization2d
import tests.utils.visualization2d as vis
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ConstantMaintenance, NoDeath
from simulation.world import World
from tests.utils.test_utils import DummyBudPolicy


def test_multiple_genomes_interaction(
    interpreter_factory, run_env_factory, world_factory
):
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

    world = world_factory(cells)
    for t in range(steps):
        world.step()
        for i, cell in enumerate(cells):
            recorder.record(t, cell)

    recorder.save_all()


def test_multiple_genomes_interaction2(interpreter4, run_env_factory, world_factory):
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

    world = world_factory(cells)
    for t in range(steps):
        world.step()
        for cell in cells:
            recorder.record(t, cell)

    recorder.save_all()

    # visualization.plot_state_trajectories(recorder, True)
    # visualization.plot_2D_position_trajectories(recorder, True)
    # visualization.plot_quiver_last_step(recorder, True)
    # visualization.plot_quiver_along_trajectories(recorder, True)


def test_swarm_runs_and_records(world_random_factory):
    class SelfAvoidingGenome:
        """Escape from the closest neighbor while avoiding repeating the same escape direction.

        Input layout assumption (matches Cell.sense):
        inputs = [ self_pos(2), self_state(S), n0_rel(2), n0_state(S), n1_rel(2), n1_state(S), ... ]

        Behavior:
        - Read last 2 dims of self_state as 'memory' (previous escape).
        - Take the first neighbor's relative position as the threat.
        - Escape opposite to the neighbor, with a small bias away from recent memory.
        - Write new memory as EMA of the chosen move into the last 2 dims of state.
        - Outputs: [new_state(S), move(2)]
        """

        def __init__(
            self, state_size=4, step=0.3, memory_decay=0.9, avoid_strength=0.2
        ):
            assert state_size >= 2
            self.S = int(state_size)
            self.step = float(step)
            self.decay = float(memory_decay)
            self.avoid = float(avoid_strength)
            self.output_size = self.S + 2

        def activate(self, inputs, rng=None):
            x = np.asarray(inputs, dtype=float)
            S = self.S

            # previous memory from last 2 dims of self_state
            # self_pos(2) + self_state(S) => memory at [2+S-2 : 2+S]
            mem_prev = np.zeros(2)
            if x.shape[0] >= 2 + S:
                mem_prev = x[2 + S - 2 : 2 + S]

            # first neighbor relative vector at offset 2 + S
            rel = np.zeros(2)
            if x.shape[0] >= 2 + S + 2:
                rel = x[2 + S : 2 + S + 2]

            # helpers
            def _unit(v):
                n = float(np.linalg.norm(v))
                return v / n if n > 1e-12 else v

            escape = -_unit(rel)  # flee opposite to the neighbor
            bias = -self.avoid * mem_prev  # avoid repeating recent heading
            mv = _unit(escape + bias) * self.step  # chosen move (2D)

            # update memory (EMA) and write to last 2 dims of state
            mem_new = self.decay * mem_prev + (1.0 - self.decay) * mv
            state_out = [0.0] * (S - 2) + [float(mem_new[0]), float(mem_new[1])]
            return state_out + [float(mv[0]), float(mv[1])]

    w = world_random_factory(
        n=100,
        seed=42,
        box=(-8, 8, -8, 8),
        genome_builder=lambda i: SelfAvoidingGenome(state_size=4, step=0.3),
        energy_policy=ConstantMaintenance(0.0),
        reproduction_policy=DummyBudPolicy(),
    )

    # minimal recorder
    class R:
        positions = []

    r = R()
    r.positions = []
    for _ in range(20):
        w.step()
        for c in w.cells:
            r.positions.append(
                [w.time, str(c.id), float(c.position[0]), float(c.position[1])]
            )
    # optional visual checks (keep show=False in tests)
    # vis.plot_2D_position_trajectories(r, show=True)
    # vis.plot_quiver_along_trajectories(r, arrow_stride=5, show=True)
    # vis.animate_2D_position_trajectories(
    #     r, tail=40, legend="outside", legend_cols=2, label_shorten=6
    # )
    # vis.animate_quiver_2D(
    #     r, tail_steps=6, interval=50, save_path="quiver.gif", blit=True
    # )
