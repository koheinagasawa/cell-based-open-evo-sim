"""Integration tests: PhysicsSolver wired into World.step() (with GIF output)."""
import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.physics.solver import PhysicsSolver
from tests.utils.test_utils import DummyEnergyPolicy, DummyBudPolicy
from tests.utils.visualization2d import animate_field_cells_connections


class _StaticGenome:
    """Genome that always returns zeros (no voluntary movement)."""

    def __init__(self, output_size):
        self._size = output_size

    def activate(self, inputs):
        return np.zeros(self._size)


def _make_cell(pos, pid, radius=0.5, conn_out=None):
    interp = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})
    c = Cell(
        pos,
        _StaticGenome(6),
        id=pid,
        interpreter=interp,
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


def _blank_field(h=1, w=1):
    return np.zeros((h, w), dtype=float)


def _run_and_record(world, steps):
    """Run world for N steps, return (field_frames, cell_frames, edge_frames)."""
    field_frames, cell_frames, edge_frames = [], [], []
    for _ in range(steps):
        cf, ef = _record_frame(world.cells)
        field_frames.append(_blank_field())
        cell_frames.append(cf)
        edge_frames.append(ef)
        world.step()
    # Final frame
    cf, ef = _record_frame(world.cells)
    field_frames.append(_blank_field())
    cell_frames.append(cf)
    edge_frames.append(ef)
    return field_frames, cell_frames, edge_frames


def _save_gif(test_output_dir, name, field_frames, cell_frames, edge_frames,
              *, fps=20, trail_len=40, figsize=(5, 5)):
    # Compute field extent from all cell positions
    all_x, all_y = [], []
    for cf in cell_frames:
        for (x, y) in cf.values():
            all_x.append(x)
            all_y.append(y)
    pad = 0.5
    field_extent = (min(all_x) - pad, max(all_x) + pad,
                    min(all_y) - pad, max(all_y) + pad)

    out = test_output_dir / f"{name}.gif"
    animate_field_cells_connections(
        out_path=str(out),
        field_frames=field_frames,
        cell_frames=cell_frames,
        edge_frames=edge_frames,
        fps=fps,
        trail_len=trail_len,
        figsize=figsize,
        cmap="gray",
        show_colorbar=False,
        field_extent=field_extent,
    )
    assert out.exists() and out.stat().st_size > 0


class TestWorldPhysicsIntegration:
    """World.step() should apply physics when a solver is provided."""

    def test_overlapping_cells_repelled(self, world_factory, test_output_dir):
        """Two overlapping cells are pushed apart. Outputs GIF."""
        a = _make_cell([0.0, 0.0], "A", radius=0.5)
        b = _make_cell([0.4, 0.0], "B", radius=0.5)
        pos_a_before = a.position.copy()
        pos_b_before = b.position.copy()

        solver = PhysicsSolver(dt=0.05, repulsion_stiffness=2.0)
        w = world_factory([a, b], physics_solver=solver)
        ff, cf, ef = _run_and_record(w, steps=80)

        # Cells should have been pushed apart
        assert a.position[0] < pos_a_before[0]
        assert b.position[0] > pos_b_before[0]

        #_save_gif(test_output_dir, "physics_repulsion", ff, cf, ef)

    def test_bonded_cells_equilibrium(self, world_factory, test_output_dir):
        """Bonded cells converge to rest length from stretched state. Outputs GIF."""
        a = _make_cell([0.0, 0.0], "A", radius=0.5, conn_out={"B": 1.0})
        b = _make_cell([4.0, 0.0], "B", radius=0.5, conn_out={"A": 1.0})

        solver = PhysicsSolver(dt=0.05, spring_stiffness=2.0, repulsion_stiffness=2.0)
        w = world_factory([a, b], physics_solver=solver)
        ff, cf, ef = _run_and_record(w, steps=200)

        dist = np.linalg.norm(b.position - a.position)
        np.testing.assert_allclose(dist, 1.0, atol=0.05)

        #_save_gif(test_output_dir, "physics_spring_equilibrium", ff, cf, ef,
        #          trail_len=60)

    def test_no_solver_no_physics(self, world_factory):
        """Without solver, overlapping cells stay where move puts them."""
        a = _make_cell([0.0, 0.0], "A")
        b = _make_cell([0.6, 0.0], "B")

        w = world_factory([a, b])  # no physics_solver
        w.step()

        # Static genome outputs zero move, so positions unchanged
        np.testing.assert_allclose(a.position, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(b.position, [0.6, 0.0], atol=1e-12)

    def test_physics_deterministic_across_steps(self, world_factory):
        """Two identical worlds produce identical trajectories."""

        def trial():
            a = _make_cell([0.0, 0.0], "A", conn_out={"B": 1.0})
            b = _make_cell([0.6, 0.0], "B", conn_out={"A": 1.0})
            w = world_factory([a, b], seed=42, physics_solver=PhysicsSolver())
            for _ in range(10):
                w.step()
            return a.position.copy(), b.position.copy()

        (a1, b1) = trial()
        (a2, b2) = trial()
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)

    def test_agent_body(self, world_factory, test_output_dir):
        """7-cell hexagonal agent body relaxes from noisy positions. Outputs GIF."""
        rng = np.random.default_rng(42)
        center_id = "C0"

        cells = [_make_cell([0.0, 0.0], center_id, radius=0.4)]
        ring_ids = []
        for i in range(6):
            angle = i * np.pi / 3.0
            nominal = np.array([np.cos(angle), np.sin(angle)]) * 0.8
            noise = rng.normal(0, 0.3, size=2)
            pid = f"C{i+1}"
            ring_ids.append(pid)
            cells.append(_make_cell(nominal + noise, pid, radius=0.4))

        cells[0].set_connections({rid: 1.0 for rid in ring_ids})
        for i, rid in enumerate(ring_ids):
            conns = {center_id: 1.0}
            conns[ring_ids[(i - 1) % 6]] = 1.0
            conns[ring_ids[(i + 1) % 6]] = 1.0
            cells[i + 1].set_connections(conns)

        solver = PhysicsSolver(dt=0.03, spring_stiffness=3.0, repulsion_stiffness=3.0)
        w = world_factory(cells, physics_solver=solver)
        ff, cf, ef = _run_and_record(w, steps=200)

        center_pos = cells[0].position.copy()
        ring_positions = np.array([cell.position.copy() for cell in cells[1:]])
        ring_centroid = ring_positions.mean(axis=0)

        # The center should remain near the centroid of the ring.
        np.testing.assert_allclose(center_pos, ring_centroid, atol=0.2)

        # The ring should relax to roughly the intended radius from the center.
        center_to_ring = np.linalg.norm(ring_positions - center_pos, axis=1)
        np.testing.assert_allclose(center_to_ring, np.full(6, 0.8), atol=0.2)

        # Neighboring ring cells should be spaced like a regular hexagon.
        neighbor_distances = np.array([
            np.linalg.norm(ring_positions[i] - ring_positions[(i + 1) % 6])
            for i in range(6)
        ])
        np.testing.assert_allclose(neighbor_distances, np.full(6, 0.8), atol=0.2)

        # Opposite ring cells should remain farther apart than neighbors,
        # guarding against collapse into a non-hexagonal shape.
        opposite_distances = np.array([
            np.linalg.norm(ring_positions[i] - ring_positions[(i + 3) % 6])
            for i in range(3)
        ])
        assert np.all(opposite_distances > 1.2)

        #_save_gif(test_output_dir, "physics_agent_body", ff, cf, ef,
        #          trail_len=30, figsize=(6, 6))

    def test_two_agents_collision(self, world_factory, test_output_dir):
        """Two 3-cell agents approach and repel each other. Outputs GIF."""
        a1 = _make_cell([-2.0, 0.0], "L0", radius=0.4, conn_out={"L1": 1.0, "L2": 1.0})
        a2 = _make_cell([-2.0, 0.8], "L1", radius=0.4, conn_out={"L0": 1.0, "L2": 1.0})
        a3 = _make_cell([-2.0, -0.8], "L2", radius=0.4, conn_out={"L0": 1.0, "L1": 1.0})

        b1 = _make_cell([2.0, 0.0], "R0", radius=0.4, conn_out={"R1": 1.0, "R2": 1.0})
        b2 = _make_cell([2.0, 0.8], "R1", radius=0.4, conn_out={"R0": 1.0, "R2": 1.0})
        b3 = _make_cell([2.0, -0.8], "R2", radius=0.4, conn_out={"R0": 1.0, "R1": 1.0})

        cells = [a1, a2, a3, b1, b2, b3]

        class _InwardGenome:
            def __init__(self, dx):
                self._dx = dx

            def activate(self, inputs):
                out = np.zeros(6)
                out[4] = self._dx
                return out

        for c in [a1, a2, a3]:
            c.genome = _InwardGenome(0.15)
        for c in [b1, b2, b3]:
            c.genome = _InwardGenome(-0.15)

        solver = PhysicsSolver(dt=0.05, spring_stiffness=3.0, repulsion_stiffness=4.0)
        w = world_factory(cells, physics_solver=solver)

        field_frames, cell_frames, edge_frames = [], [], []
        for step in range(150):
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

        #_save_gif(test_output_dir, "physics_two_agents_collision",
        #          field_frames, cell_frames, edge_frames,
        #          figsize=(7, 5))

        # --- Physics-behavior assertions ---

        a_cells = [a1, a2, a3]
        b_cells = [b1, b2, b3]

        # 1. The two groups should not have collapsed into a single blob;
        #    their centroids must be clearly separated after the simulation.
        centroid_a = np.mean([c.position for c in a_cells], axis=0)
        centroid_b = np.mean([c.position for c in b_cells], axis=0)
        centroid_sep = np.linalg.norm(centroid_b - centroid_a)
        assert centroid_sep > 0.5, (
            f"Group centroids collapsed too close after collision: "
            f"separation={centroid_sep:.3f}"
        )

        # 2. Verify the groups actually collided and moved through to
        #    opposite sides: group A (started at x=-2, moved right) should
        #    have its centroid at positive x, group B (started at x=2,
        #    moved left) at negative x.
        assert centroid_a[0] > 0.0, (
            f"Group A centroid x={centroid_a[0]:.3f} expected > 0 after crossing"
        )
        assert centroid_b[0] < 0.0, (
            f"Group B centroid x={centroid_b[0]:.3f} expected < 0 after crossing"
        )

        # 3. Outer cells of opposite groups (which are free to separate
        #    along x) should be pushed apart beyond their combined radii.
        combined_radius = a1.radius + b1.radius
        separation_tolerance = 0.05  # numerical settling allowance
        for ca in [a2, a3]:      # outer cells of group A
            for cb in [b2, b3]:  # outer cells of group B
                dist = np.linalg.norm(ca.position - cb.position)
                assert dist >= combined_radius - separation_tolerance, (
                    f"Outer cells {ca.id} and {cb.id} still overlap after "
                    f"repulsion: dist={dist:.3f} < "
                    f"{combined_radius - separation_tolerance:.3f}"
                )
