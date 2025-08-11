import numpy as np

import tests.utils.visualization as vis
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter


class Orbit3DGenome:
    """Minimal 3D demo genome: circular motion in XY + gentle Z oscillation.
    Each cell instance keeps its own phase so please instantiate per-cell.
    """

    def __init__(self, state_size=4, amp_xy=0.3, amp_z=0.15, omega=0.2, phase=0.0):
        self.S = int(state_size)
        self.amp_xy = float(amp_xy)
        self.amp_z = float(amp_z)
        self.omega = float(omega)
        self.phase = float(phase)
        self.t = 0

    def activate(self, inputs, rng=None):
        # Parametric circle with a slow vertical component
        dx = self.amp_xy * np.cos(self.omega * self.t + self.phase)
        dy = self.amp_xy * np.sin(self.omega * self.t + self.phase)
        dz = self.amp_z * np.sin(self.omega * self.t * 0.5 + self.phase)
        self.t += 1
        return [0.0] * self.S + [dx, dy, dz]  # move has 3 dims


def _interp(S, move_dim):
    # Interpreter with 3-dim move slot
    return SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + move_dim)})


def _pos3(x, y, z):
    return [float(x), float(y), float(z)]


def test_demo_3d_orbits(run_env_factory, world_factory, tmp_path=None):
    S, steps = 4, 120
    interp = _interp(S, 3)

    # Three cells with different phases
    cells = [
        Cell(
            position=_pos3(0.0, 0.0, 0.0),
            genome=Orbit3DGenome(S, phase=0.0),
            state_size=S,
            interpreter=interp,
        ),
        Cell(
            position=_pos3(2.0, 0.0, 0.5),
            genome=Orbit3DGenome(S, phase=1.0),
            state_size=S,
            interpreter=interp,
        ),
        Cell(
            position=_pos3(-2.0, 0.0, 1.0),
            genome=Orbit3DGenome(S, phase=2.1),
            state_size=S,
            interpreter=interp,
        ),
    ]
    world = world_factory(cells, seed=123)

    run_config, recorder = run_env_factory(
        {
            "genome": "Orbit3DGenome",
            "state_size": S,
            "action_size": 3,
            "steps": steps,
        }
    )

    for t in range(steps):
        world.step()
        for i, c in enumerate(cells):
            recorder.record(t, c)

    recorder.save_all()

    # Optional: quick numerical smoke
    # z should vary for at least one cell
    zs = np.stack([row[4] for row in recorder.positions if row[1] == cells[0].id])
    assert np.std(zs) > 0.01

    vis.plot_3d_position_trajectories(recorder, show=True)
