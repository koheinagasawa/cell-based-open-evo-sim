import numpy as np

import tests.utils.visualization as vis
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ConstantMaintenance, NoDeath
from simulation.world import World
from tests.utils.test_utils import DummyBudPolicy


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

    def __init__(self, state_size=4, step=0.3, memory_decay=0.9, avoid_strength=0.2):
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


def test_swarm_runs_and_records(world_random_factory):
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
    vis.animate_2D_position_trajectories(
        r, tail=40, legend="outside", legend_cols=2, label_shorten=6
    )
    # vis.animate_quiver_2D(
    #     r, tail_steps=6, interval=50, save_path="quiver.gif", blit=True
    # )
