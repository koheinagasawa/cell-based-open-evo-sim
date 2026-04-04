"""Tests for spring (bond) forces between connected particles."""
import numpy as np
import pytest


def _make_particle(pos, pid="a", radius=0.5, conn_out=None):
    """Minimal particle with connections for spring tests."""

    class _P:
        pass

    p = _P()
    p.position = np.array(pos, dtype=float)
    p.radius = float(radius)
    p.id = pid
    p.conn_out = conn_out or {}
    return p


class TestComputeSpringForces:
    """Unit tests for compute_spring_forces()."""

    def test_stretched_bond_pulls_together(self):
        """Particles farther than rest length are attracted."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([3.0, 0.0], pid="b")
        # rest_length = r_a + r_b = 1.0, distance = 3.0 -> stretched

        forces = compute_spring_forces([a, b])
        # a pulled toward +x, b pulled toward -x
        assert forces[0, 0] > 0.0
        assert forces[1, 0] < 0.0
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)

    def test_compressed_bond_pushes_apart(self):
        """Particles closer than rest length are repelled."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([0.5, 0.0], pid="b")
        # rest_length = 1.0, distance = 0.5 -> compressed

        forces = compute_spring_forces([a, b])
        # a pushed toward -x, b pushed toward +x
        assert forces[0, 0] < 0.0
        assert forces[1, 0] > 0.0

    def test_at_rest_length_no_force(self):
        """Particles at exactly rest length produce zero force."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", radius=0.5, conn_out={"b": 1.0})
        b = _make_particle([1.0, 0.0], pid="b", radius=0.5)
        # rest_length = 0.5 + 0.5 = 1.0, distance = 1.0

        forces = compute_spring_forces([a, b])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_no_connections_no_force(self):
        """Unconnected particles produce zero spring force."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a")
        b = _make_particle([0.5, 0.0], pid="b")

        forces = compute_spring_forces([a, b])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_bidirectional_connection(self):
        """Both a->b and b->a edges: forces accumulate symmetrically."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([3.0, 0.0], pid="b", conn_out={"a": 1.0})

        forces = compute_spring_forces([a, b])
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)
        # Force should be double compared to unidirectional
        a2 = _make_particle([0.0, 0.0], pid="a2", conn_out={"b2": 1.0})
        b2 = _make_particle([3.0, 0.0], pid="b2")
        forces_uni = compute_spring_forces([a2, b2])
        np.testing.assert_allclose(
            np.abs(forces[0]), np.abs(forces_uni[0]) * 2.0, atol=1e-12
        )

    def test_3d_spring(self):
        """Spring works in 3D."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0, 0.0], pid="a", radius=0.5, conn_out={"b": 1.0})
        b = _make_particle([2.0, 0.0, 0.0], pid="b", radius=0.5)

        forces = compute_spring_forces([a, b])
        assert forces.shape == (2, 3)
        assert forces[0, 0] > 0.0  # a pulled toward +x
        assert forces[0, 1] == 0.0
        assert forces[0, 2] == 0.0

    def test_coincident_connected_particles(self):
        """Connected particles at same position don't produce NaN."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([1.0, 1.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([1.0, 1.0], pid="b")

        forces = compute_spring_forces([a, b])
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isinf(forces))

    def test_stiffness_parameter(self):
        """Custom stiffness scales spring force."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0})
        b = _make_particle([3.0, 0.0], pid="b")

        f1 = compute_spring_forces([a, b], stiffness=1.0)
        f2 = compute_spring_forces([a, b], stiffness=3.0)
        np.testing.assert_allclose(f2, f1 * 3.0, atol=1e-12)

    def test_deterministic(self):
        """Same input produces identical output across calls."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"b": 1.0, "c": 1.0})
        b = _make_particle([2.0, 0.0], pid="b", conn_out={"a": 1.0})
        c = _make_particle([0.0, 2.0], pid="c")

        f1 = compute_spring_forces([a, b, c])
        f2 = compute_spring_forces([a, b, c])
        np.testing.assert_array_equal(f1, f2)

    def test_dangling_connection_ignored(self):
        """Connection to non-existent particle ID is silently ignored."""
        from simulation.physics.spring import compute_spring_forces

        a = _make_particle([0.0, 0.0], pid="a", conn_out={"z": 1.0})
        b = _make_particle([0.5, 0.0], pid="b")

        forces = compute_spring_forces([a, b])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)
