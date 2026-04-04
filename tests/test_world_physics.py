"""Integration tests: PhysicsSolver wired into World.step()."""
import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.physics.solver import PhysicsSolver
from tests.utils.test_utils import DummyEnergyPolicy, DummyBudPolicy


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


class TestWorldPhysicsIntegration:
    """World.step() should apply physics when a solver is provided."""

    def test_overlapping_cells_repelled_after_step(self, world_factory):
        a = _make_cell([0.0, 0.0], "a")
        b = _make_cell([0.6, 0.0], "b")
        pos_a_before = a.position.copy()
        pos_b_before = b.position.copy()

        w = world_factory([a, b], physics_solver=PhysicsSolver())
        w.step()

        # Cells should have been pushed apart
        assert a.position[0] < pos_a_before[0]
        assert b.position[0] > pos_b_before[0]

    def test_bonded_cells_pulled_together(self, world_factory):
        a = _make_cell([0.0, 0.0], "a", conn_out={"b": 1.0})
        b = _make_cell([5.0, 0.0], "b")
        dist_before = np.linalg.norm(b.position - a.position)

        w = world_factory([a, b], physics_solver=PhysicsSolver())
        w.step()

        dist_after = np.linalg.norm(b.position - a.position)
        assert dist_after < dist_before

    def test_no_solver_no_physics(self, world_factory):
        """Without solver, overlapping cells stay where move puts them."""
        a = _make_cell([0.0, 0.0], "a")
        b = _make_cell([0.6, 0.0], "b")

        w = world_factory([a, b])  # no physics_solver
        w.step()

        # Static genome outputs zero move, so positions unchanged
        np.testing.assert_allclose(a.position, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(b.position, [0.6, 0.0], atol=1e-12)

    def test_physics_deterministic_across_steps(self, world_factory):
        """Two identical worlds produce identical trajectories."""

        def trial():
            a = _make_cell([0.0, 0.0], "a", conn_out={"b": 1.0})
            b = _make_cell([0.6, 0.0], "b", conn_out={"a": 1.0})
            w = world_factory([a, b], seed=42, physics_solver=PhysicsSolver())
            for _ in range(10):
                w.step()
            return a.position.copy(), b.position.copy()

        (a1, b1) = trial()
        (a2, b2) = trial()
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)

    def test_equilibrium_reached(self, world_factory):
        """Bonded overlapping cells converge toward rest length."""
        a = _make_cell([0.0, 0.0], "a", radius=0.5, conn_out={"b": 1.0})
        b = _make_cell([0.3, 0.0], "b", radius=0.5, conn_out={"a": 1.0})
        # rest_length = 1.0, initial distance = 0.3

        w = world_factory([a, b], physics_solver=PhysicsSolver(dt=0.05))
        for _ in range(200):
            w.step()

        dist = np.linalg.norm(b.position - a.position)
        np.testing.assert_allclose(dist, 1.0, atol=0.05)
