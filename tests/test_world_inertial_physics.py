from dataclasses import replace

import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.messaging import MessageRouter
from simulation.physics.inertial import (
    InertialPhysicsConfig,
    InertialPhysicsSolver,
    PhysicalBond,
    PhysicsFailure,
)
from simulation.physics.solver import PhysicsSolver
from simulation.policies import ConstantMaintenance, SimpleBudding
from simulation.world import World


class _StateAndMessageGenome:
    def __init__(self):
        self.calls = 0

    def activate(self, inputs, rng=None):
        self.calls += 1
        return np.array([inputs[2] + 1, 0, 0, 0, 1], dtype=float)


def _cells():
    return [
        Cell(
            position,
            _StateAndMessageGenome(),
            id=cell_id,
            max_neighbors=0,
            interpreter=SlotBasedInterpreter({"state": slice(0, 4), "emit:signal": 4}),
            recv_layout={"recv:signal": 1},
        )
        for cell_id, position in (("a", (-0.6, 0)), ("b", (0.6, 0)))
    ]


def _world(cells, solver, substeps=10, *, maintenance=0):
    return World(
        cells,
        seed=7,
        use_neighbors=False,
        energy_policy=ConstantMaintenance(maintenance),
        reproduction_policy=SimpleBudding(),
        message_router=MessageRouter(),
        physics_solver=solver,
        physics_substeps=substeps,
    )


@pytest.mark.parametrize(("dt", "substeps"), [(0.01, 10), (0.005, 20)])
def test_one_control_update_runs_only_physics_at_substep_frequency(dt, substeps):
    cells, reference = _cells(), _cells()
    bonds = [PhysicalBond("a", "b")]
    config = replace(InertialPhysicsConfig(), dt=dt)
    world = _world(cells, InertialPhysicsSolver(bonds, config=config), substeps)
    reference_solver = InertialPhysicsSolver(bonds, config=config)
    world.step()
    for _ in range(substeps):
        reference_solver.step(reference)
    assert world.time == 1
    for actual, expected in zip(cells, reference):
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.velocity, expected.velocity)
        np.testing.assert_array_equal(actual.state, [1, 0, 0, 0])
        assert actual.age == actual.genome.calls == 1
    assert "time_phase3_5_physics" in world.perf_stats


def test_mutating_message_edges_changes_delivery_but_not_physical_trajectory():
    first, second = _cells(), _cells()
    left = _world(first, InertialPhysicsSolver([PhysicalBond("a", "b")]))
    right = _world(second, InertialPhysicsSolver([PhysicalBond("a", "b")]))
    for step in range(10):
        if step < 3:
            second[0].set_connections({"b": 4})
        elif step < 6:
            second[0].set_connections({})
            second[1].set_connections({"a": -2})
        else:
            second[0].set_connections({"b": 100})
            second[1].set_connections({"a": 0.25})
        left.step()
        right.step()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.position, b.position)
            np.testing.assert_array_equal(a.velocity, b.velocity)
            assert a.mass == b.mass == 1
        assert not first[0].inbox and not first[1].inbox
        if step < 3:
            np.testing.assert_array_equal(second[1].inbox["recv:signal"], [4])
        elif step < 6:
            np.testing.assert_array_equal(second[0].inbox["recv:signal"], [-2])
            assert not second[1].inbox
        else:
            np.testing.assert_array_equal(second[1].inbox["recv:signal"], [100])
            np.testing.assert_array_equal(second[0].inbox["recv:signal"], [0.25])
    assert left.physics_solver.bonds == right.physics_solver.bonds


def test_message_only_and_physical_only_links_are_different():
    message_cells, physical_cells = _cells(), _cells()
    message_cells[0].set_connections({"b": 1})
    message_world = _world(message_cells, InertialPhysicsSolver())
    physical_world = _world(
        physical_cells, InertialPhysicsSolver([PhysicalBond("a", "b")])
    )
    initial = message_cells[0].position.copy()
    message_world.step()
    physical_world.step()
    np.testing.assert_array_equal(message_cells[0].position, initial)
    np.testing.assert_array_equal(message_cells[1].inbox["recv:signal"], [1])
    assert physical_cells[0].position[0] > initial[0]
    assert not physical_cells[0].conn_out and not physical_cells[1].conn_out
    assert not physical_cells[0].inbox and not physical_cells[1].inbox


def test_maintenance_and_message_commit_remain_once_per_control_interval():
    cells = _cells()
    cells[0].set_connections({"b": 3})
    world = _world(
        cells, InertialPhysicsSolver([PhysicalBond("a", "b")]), maintenance=0.1
    )
    world.step()
    for cell in cells:
        assert cell.energy == pytest.approx(0.9)
        assert cell.age == cell.genome.calls == 1
    np.testing.assert_array_equal(cells[1].inbox["recv:signal"], [3])
    assert len(world.cells) == 2


def test_default_world_keeps_one_legacy_physics_step():
    cells, reference = _cells(), _cells()
    cells[0].set_connections({"b": 1})
    reference[0].set_connections({"b": 1})
    world = World(
        cells,
        energy_policy=ConstantMaintenance(),
        reproduction_policy=SimpleBudding(),
        physics_solver=PhysicsSolver(),
        use_neighbors=False,
    )
    world.step()
    PhysicsSolver().step(reference)
    assert world.physics_substeps == 1
    for actual, expected in zip(cells, reference):
        np.testing.assert_array_equal(actual.position, expected.position)


@pytest.mark.parametrize("substeps", [0, -1, 1.5, True, None, "10"])
def test_invalid_substep_counts_are_rejected(substeps):
    with pytest.raises(ValueError, match="physics_substeps"):
        _world(_cells(), InertialPhysicsSolver(), substeps)


def test_substeps_without_solver_are_rejected():
    with pytest.raises(ValueError, match="physics_solver"):
        _world(_cells(), None, 10)


def test_dangling_physical_link_fails_instead_of_using_a_message_link():
    cells = _cells()
    cells[0].set_connections({"b": 1})
    world = _world(cells, InertialPhysicsSolver([PhysicalBond("a", "missing")]))
    with pytest.raises(ValueError, match="dangling"):
        world.step()
    assert world.time == 0


def test_substep_failure_stops_before_maintenance_and_message_commit():
    class _FailingSolver:
        def __init__(self):
            self.calls = 0
            self.solver = InertialPhysicsSolver([PhysicalBond("a", "b")])

        def step(self, cells):
            self.calls += 1
            if self.calls == 3:
                raise PhysicsFailure(
                    "speed_limit",
                    "before",
                    tuple(c.id for c in cells),
                    np.array([c.position for c in cells]),
                    np.array([c.velocity for c in cells]),
                )
            self.solver.step(cells)

    cells = _cells()
    cells[0].set_connections({"b": 1})
    solver = _FailingSolver()
    world = _world(cells, solver, maintenance=0.1)
    with pytest.raises(PhysicsFailure, match="speed_limit"):
        world.step()
    assert solver.calls == 3
    assert world.time == 0
    assert cells[0].energy == cells[1].energy == 1
    assert not cells[1].inbox
    np.testing.assert_array_equal(cells[0].state, [1, 0, 0, 0])
