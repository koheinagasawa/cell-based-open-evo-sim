"""Tests for physics solver (force aggregation + position integration)."""
import numpy as np
import pytest


def _make_particle(pos, pid="a", radius=0.5, conn_out=None):
    class _P:
        pass

    p = _P()
    p.position = np.array(pos, dtype=float)
    p.radius = float(radius)
    p.id = pid
    p.conn_out = conn_out or {}
    return p


class TestSolver:
    """Tests for PhysicsSolver step."""

    def test_overlapping_unconnected_repelled(self):
        """Two overlapping unconnected particles are pushed apart."""
        from simulation.physics.solver import PhysicsSolver

        a = _make_particle([0.0, 0.0], pid="a")
        b = _make_particle([0.6, 0.0], pid="b")
        pos_before_a = a.position.copy()
        pos_before_b = b.position.copy()

        solver = PhysicsSolver()
        solver.step([a, b])

        # a moved in -x, b moved in +x
        assert a.position[0] < pos_before_a[0]
        assert b.position[0] > pos_before_b[0]

    def test_stretched_bond_contracts(self):
        """Connected particles farther than rest length move closer."""
        from simulation.physics.solver import PhysicsSolver

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([5.0, 0.0], pid="b")
        dist_before = np.linalg.norm(b.position - a.position)

        solver = PhysicsSolver()
        solver.step([a, b])

        dist_after = np.linalg.norm(b.position - a.position)
        assert dist_after < dist_before

    def test_equilibrium_no_movement(self):
        """Particles at rest (no overlap, bonds at rest length) don't move."""
        from simulation.physics.solver import PhysicsSolver

        a = _make_particle([0.0, 0.0], pid="a", radius=0.5, conn_out={"b": 1.0})
        b = _make_particle([1.0, 0.0], pid="b", radius=0.5)
        pos_a = a.position.copy()
        pos_b = b.position.copy()

        solver = PhysicsSolver()
        solver.step([a, b])

        np.testing.assert_allclose(a.position, pos_a, atol=1e-15)
        np.testing.assert_allclose(b.position, pos_b, atol=1e-15)

    def test_dt_scales_displacement(self):
        """Larger dt produces larger displacement."""
        from simulation.physics.solver import PhysicsSolver

        def trial(dt):
            a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
            b = _make_particle([5.0, 0.0], pid="b")
            solver = PhysicsSolver(dt=dt)
            solver.step([a, b])
            return np.linalg.norm(b.position - np.array([5.0, 0.0]))

        disp_small = trial(0.01)
        disp_large = trial(0.1)
        assert disp_large > disp_small

    def test_combined_spring_and_repulsion(self):
        """Spring and repulsion act together within a single step."""
        from simulation.physics.solver import PhysicsSolver

        # Three particles in a line: a--b (bonded, overlapping), c (far, unconnected)
        a = _make_particle([0.0, 0.0], pid="a", radius=0.5, conn_out={"b": 1.0})
        b = _make_particle([0.4, 0.0], pid="b", radius=0.5)  # overlapping with a
        c = _make_particle([10.0, 0.0], pid="c")  # far away

        pos_c_before = c.position.copy()

        solver = PhysicsSolver()
        solver.step([a, b, c])

        # c should not move (no connections, no overlap)
        np.testing.assert_allclose(c.position, pos_c_before, atol=1e-15)
        # a and b should have moved (spring + repulsion both push them apart here)
        assert a.position[0] < 0.0
        assert b.position[0] > 0.4

    def test_empty_particle_list(self):
        """Solver handles empty input gracefully."""
        from simulation.physics.solver import PhysicsSolver

        solver = PhysicsSolver()
        solver.step([])  # should not raise

    def test_single_particle_no_movement(self):
        """A lone particle doesn't move."""
        from simulation.physics.solver import PhysicsSolver

        a = _make_particle([3.0, 1.0], pid="a")
        pos = a.position.copy()
        solver = PhysicsSolver()
        solver.step([a])
        np.testing.assert_allclose(a.position, pos, atol=1e-15)

    def test_deterministic(self):
        """Same configuration yields identical results across runs."""
        from simulation.physics.solver import PhysicsSolver

        def trial():
            a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
            b = _make_particle([0.6, 0.0], pid="b", conn_out={"a": 1.0})
            c = _make_particle([0.3, 0.8], pid="c")
            solver = PhysicsSolver()
            solver.step([a, b, c])
            return np.array([a.position, b.position, c.position])

        r1 = trial()
        r2 = trial()
        np.testing.assert_array_equal(r1, r2)

    def test_3d_solver(self):
        """Solver works with 3D positions."""
        from simulation.physics.solver import PhysicsSolver

        a = _make_particle([0.0, 0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([3.0, 0.0, 0.0], pid="b")
        dist_before = np.linalg.norm(b.position - a.position)

        solver = PhysicsSolver()
        solver.step([a, b])

        dist_after = np.linalg.norm(b.position - a.position)
        assert dist_after < dist_before
