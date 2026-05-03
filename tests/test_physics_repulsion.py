"""Tests for soft-repulsion force between cells."""
import numpy as np
import pytest


def _make_particle(pos, radius=0.5):
    """Minimal particle-like object for physics tests."""

    class _P:
        pass

    p = _P()
    p.position = np.array(pos, dtype=float)
    p.radius = float(radius)
    return p


class TestComputeRepulsionForces:
    """Unit tests for compute_repulsion_forces()."""

    def test_overlapping_pair_pushed_apart(self):
        """Two overlapping particles receive equal and opposite forces."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([0.0, 0.0], radius=0.5)
        b = _make_particle([0.6, 0.0], radius=0.5)
        # overlap = (0.5+0.5) - 0.6 = 0.4 > 0

        forces = compute_repulsion_forces([a, b])
        assert forces.shape == (2, 2)

        # a should be pushed in -x, b in +x
        assert forces[0, 0] < 0.0
        assert forces[1, 0] > 0.0
        # Newton's third law
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)

    def test_no_overlap_no_force(self):
        """Particles that don't overlap produce zero force."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([0.0, 0.0], radius=0.5)
        b = _make_particle([2.0, 0.0], radius=0.5)
        # distance=2.0, sum_radii=1.0 -> no overlap

        forces = compute_repulsion_forces([a, b])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_exactly_touching_no_force(self):
        """Particles touching but not overlapping produce zero force."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([0.0, 0.0], radius=0.5)
        b = _make_particle([1.0, 0.0], radius=0.5)

        forces = compute_repulsion_forces([a, b])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_single_particle_no_force(self):
        """A lone particle gets zero force."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([3.0, 1.0])
        forces = compute_repulsion_forces([a])
        np.testing.assert_allclose(forces, 0.0, atol=1e-15)

    def test_coincident_particles_handled(self):
        """Two particles at the exact same position don't produce NaN."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([1.0, 1.0], radius=0.5)
        b = _make_particle([1.0, 1.0], radius=0.5)

        forces = compute_repulsion_forces([a, b])
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isinf(forces))
        # Forces should still be equal and opposite
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)

    def test_3d_particles(self):
        """Repulsion works in 3D."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([0.0, 0.0, 0.0], radius=0.5)
        b = _make_particle([0.3, 0.4, 0.0], radius=0.5)
        # distance = 0.5, sum_radii = 1.0, overlap = 0.5

        forces = compute_repulsion_forces([a, b])
        assert forces.shape == (2, 3)
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)
        # Force should be along the direction from b to a (for particle a)
        direction = np.array([0.0, 0.0, 0.0]) - np.array([0.3, 0.4, 0.0])
        direction /= np.linalg.norm(direction)
        force_dir = forces[0] / np.linalg.norm(forces[0])
        np.testing.assert_allclose(force_dir, direction, atol=1e-12)

    def test_force_proportional_to_overlap(self):
        """Larger overlap produces larger force magnitude."""
        from simulation.physics.repulsion import compute_repulsion_forces

        # Small overlap
        a1 = _make_particle([0.0, 0.0], radius=0.5)
        b1 = _make_particle([0.9, 0.0], radius=0.5)  # overlap = 0.1
        f_small = compute_repulsion_forces([a1, b1])

        # Large overlap
        a2 = _make_particle([0.0, 0.0], radius=0.5)
        b2 = _make_particle([0.3, 0.0], radius=0.5)  # overlap = 0.7
        f_large = compute_repulsion_forces([a2, b2])

        assert np.linalg.norm(f_large[0]) > np.linalg.norm(f_small[0])

    def test_deterministic(self):
        """Same input produces identical output across calls."""
        from simulation.physics.repulsion import compute_repulsion_forces

        particles = [
            _make_particle([0.0, 0.0], radius=0.5),
            _make_particle([0.6, 0.0], radius=0.5),
            _make_particle([0.3, 0.8], radius=0.5),
        ]

        f1 = compute_repulsion_forces(particles)
        f2 = compute_repulsion_forces(particles)
        np.testing.assert_array_equal(f1, f2)

    def test_stiffness_parameter(self):
        """Custom stiffness scales force magnitude."""
        from simulation.physics.repulsion import compute_repulsion_forces

        a = _make_particle([0.0, 0.0], radius=0.5)
        b = _make_particle([0.6, 0.0], radius=0.5)

        f1 = compute_repulsion_forces([a, b], stiffness=1.0)
        f2 = compute_repulsion_forces([a, b], stiffness=2.0)

        np.testing.assert_allclose(f2, f1 * 2.0, atol=1e-12)
