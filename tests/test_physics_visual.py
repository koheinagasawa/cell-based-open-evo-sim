"""Visual smoke tests for physics: output GIFs to verify behavior."""
import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.physics.solver import PhysicsSolver
from tests.utils.visualization2d import animate_field_cells_connections


class _StaticGenome:
    """Genome that always returns zeros (no voluntary movement)."""

    def __init__(self, output_size):
        self._size = output_size

    def activate(self, inputs):
        return np.zeros(self._size)


def _interp():
    return SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})


def _make_cell(pos, pid, radius=0.5, conn_out=None):
    c = Cell(
        pos,
        _StaticGenome(6),
        id=pid,
        interpreter=_interp(),
        state_size=4,
        max_neighbors=0,
        radius=radius,
    )
    if conn_out:
        c.set_connections(conn_out)
    return c


def _record_frame(cells):
    """Capture cell positions and edges for one frame."""
    cell_frame = {}
    for c in cells:
        cell_frame[c.id] = (float(c.position[0]), float(c.position[1]))

    edges = []
    for c in cells:
        for dst_id, w in c.conn_out.items():
            edges.append((c.id, dst_id, float(w)))
    return cell_frame, edges


def _blank_field(h=32, w=32):
    return np.zeros((h, w), dtype=float)


class TestPhysicsVisualization:

    def test_repulsion_overlap_gif(self, world_factory, test_output_dir):
        """GIF: two overlapping unconnected cells pushed apart."""
        a = _make_cell([0.0, 0.0], "A", radius=0.5)
        b = _make_cell([0.4, 0.0], "B", radius=0.5)

        solver = PhysicsSolver(dt=0.05, repulsion_stiffness=2.0)
        w = world_factory([a, b], physics_solver=solver)

        field_frames, cell_frames, edge_frames = [], [], []
        steps = 80
        for _ in range(steps):
            cf, ef = _record_frame(w.cells)
            field_frames.append(_blank_field())
            cell_frames.append(cf)
            edge_frames.append(ef)
            w.step()
        # Record final frame
        cf, ef = _record_frame(w.cells)
        field_frames.append(_blank_field())
        cell_frames.append(cf)
        edge_frames.append(ef)

        out = test_output_dir / "physics_repulsion.gif"
        animate_field_cells_connections(
            out_path=str(out),
            field_frames=field_frames,
            cell_frames=cell_frames,
            edge_frames=edge_frames,
            fps=20,
            trail_len=40,
            figsize=(5, 5),
            cmap="gray",
            show_colorbar=False,
        )
        assert out.exists() and out.stat().st_size > 0

    def test_spring_equilibrium_gif(self, world_factory, test_output_dir):
        """GIF: two bonded cells converging to rest length from stretched state."""
        a = _make_cell([0.0, 0.0], "A", radius=0.5, conn_out={"B": 1.0})
        b = _make_cell([4.0, 0.0], "B", radius=0.5, conn_out={"A": 1.0})

        solver = PhysicsSolver(dt=0.05, spring_stiffness=2.0, repulsion_stiffness=2.0)
        w = world_factory([a, b], physics_solver=solver)

        field_frames, cell_frames, edge_frames = [], [], []
        steps = 120
        for _ in range(steps):
            cf, ef = _record_frame(w.cells)
            field_frames.append(_blank_field())
            cell_frames.append(cf)
            edge_frames.append(ef)
            w.step()
        cf, ef = _record_frame(w.cells)
        field_frames.append(_blank_field())
        cell_frames.append(cf)
        edge_frames.append(ef)

        out = test_output_dir / "physics_spring_equilibrium.gif"
        animate_field_cells_connections(
            out_path=str(out),
            field_frames=field_frames,
            cell_frames=cell_frames,
            edge_frames=edge_frames,
            fps=20,
            trail_len=60,
            figsize=(5, 5),
            cmap="gray",
            show_colorbar=False,
        )
        assert out.exists() and out.stat().st_size > 0

    def test_agent_body_gif(self, world_factory, test_output_dir):
        """GIF: 7-cell agent body (hexagonal) with springs relaxing from random offsets."""
        # Center cell + 6 surrounding cells (hexagonal layout with noise)
        rng = np.random.default_rng(42)
        center_id = "C0"

        # Create center
        cells = [_make_cell([0.0, 0.0], center_id, radius=0.4)]
        ring_ids = []
        for i in range(6):
            angle = i * np.pi / 3.0
            # Start positions with some random perturbation
            nominal = np.array([np.cos(angle), np.sin(angle)]) * 0.8
            noise = rng.normal(0, 0.3, size=2)
            pid = f"C{i+1}"
            ring_ids.append(pid)
            cells.append(_make_cell(nominal + noise, pid, radius=0.4))

        # Connect center to all ring cells, and ring cells to their neighbors
        cells[0].set_connections({rid: 1.0 for rid in ring_ids})
        for i, rid in enumerate(ring_ids):
            conns = {center_id: 1.0}
            # Connect to adjacent ring neighbors
            conns[ring_ids[(i - 1) % 6]] = 1.0
            conns[ring_ids[(i + 1) % 6]] = 1.0
            cells[i + 1].set_connections(conns)

        solver = PhysicsSolver(dt=0.03, spring_stiffness=3.0, repulsion_stiffness=3.0)
        w = world_factory(cells, physics_solver=solver)

        field_frames, cell_frames, edge_frames = [], [], []
        steps = 200
        for _ in range(steps):
            cf, ef = _record_frame(w.cells)
            field_frames.append(_blank_field())
            cell_frames.append(cf)
            edge_frames.append(ef)
            w.step()
        cf, ef = _record_frame(w.cells)
        field_frames.append(_blank_field())
        cell_frames.append(cf)
        edge_frames.append(ef)

        out = test_output_dir / "physics_agent_body.gif"
        animate_field_cells_connections(
            out_path=str(out),
            field_frames=field_frames,
            cell_frames=cell_frames,
            edge_frames=edge_frames,
            fps=20,
            trail_len=30,
            figsize=(6, 6),
            cmap="gray",
            show_colorbar=False,
        )
        assert out.exists() and out.stat().st_size > 0

    def test_two_agents_collision_gif(self, world_factory, test_output_dir):
        """GIF: two small agents (3 cells each) approaching, repelling each other."""
        # Agent 1: triangle on the left
        a1 = _make_cell([-2.0, 0.0], "L0", radius=0.4, conn_out={"L1": 1.0, "L2": 1.0})
        a2 = _make_cell([-2.0, 0.8], "L1", radius=0.4, conn_out={"L0": 1.0, "L2": 1.0})
        a3 = _make_cell([-2.0, -0.8], "L2", radius=0.4, conn_out={"L0": 1.0, "L1": 1.0})

        # Agent 2: triangle on the right
        b1 = _make_cell([2.0, 0.0], "R0", radius=0.4, conn_out={"R1": 1.0, "R2": 1.0})
        b2 = _make_cell([2.0, 0.8], "R1", radius=0.4, conn_out={"R0": 1.0, "R2": 1.0})
        b3 = _make_cell([2.0, -0.8], "R2", radius=0.4, conn_out={"R0": 1.0, "R1": 1.0})

        cells = [a1, a2, a3, b1, b2, b3]

        # Give cells initial velocity toward each other via a genome that pushes inward
        class _InwardGenome:
            def __init__(self, dx):
                self._dx = dx

            def activate(self, inputs):
                out = np.zeros(6)
                out[4] = self._dx  # move x
                return out

        # Override genomes for first few steps to give initial momentum
        for c in [a1, a2, a3]:
            c.genome = _InwardGenome(0.15)
        for c in [b1, b2, b3]:
            c.genome = _InwardGenome(-0.15)

        solver = PhysicsSolver(dt=0.05, spring_stiffness=3.0, repulsion_stiffness=4.0)
        w = world_factory(cells, physics_solver=solver)

        field_frames, cell_frames, edge_frames = [], [], []
        steps = 150
        for step in range(steps):
            # After 30 steps, stop voluntary movement
            if step == 30:
                static = _StaticGenome(6)
                for c in w.cells:
                    c.genome = static

            cf, ef = _record_frame(w.cells)
            field_frames.append(_blank_field())
            cell_frames.append(cf)
            edge_frames.append(ef)
            w.step()
        cf, ef = _record_frame(w.cells)
        field_frames.append(_blank_field())
        cell_frames.append(cf)
        edge_frames.append(ef)

        out = test_output_dir / "physics_two_agents_collision.gif"
        animate_field_cells_connections(
            out_path=str(out),
            field_frames=field_frames,
            cell_frames=cell_frames,
            edge_frames=edge_frames,
            fps=20,
            trail_len=40,
            figsize=(7, 5),
            cmap="gray",
            show_colorbar=False,
        )
        assert out.exists() and out.stat().st_size > 0
