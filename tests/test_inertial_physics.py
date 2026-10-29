from dataclasses import replace

import numpy as np
import pytest

from simulation.cell import Cell
from simulation.physics.inertial import (
    InertialPhysicsConfig,
    InertialPhysicsSolver,
    PhysicalBond,
    PhysicsFailure,
)


def particle(cell_id, position, velocity=(0.0, 0.0)):
    return Cell(position, genome=None, id=cell_id, velocity=velocity)


def test_velocity_is_owned_by_each_cell():
    velocity = np.array([1.0, 2.0])
    a = particle("a", (0, 0), velocity)
    b = particle("b", (2, 0), velocity)
    velocity[:] = 5
    a.velocity[0] = 9
    np.testing.assert_array_equal(b.velocity, [1, 2])
    assert a.mass == b.mass == 1
    np.testing.assert_array_equal(particle("c", (0, 0)).velocity, [0, 0])


def test_approved_physics_defaults():
    config = InertialPhysicsConfig()
    assert config.dt == 0.01
    assert config.drag == 1
    assert config.spring_stiffness == 10
    assert config.repulsion_stiffness == 100
    assert config.max_strain == 0.2
    assert config.max_cells == 64
    assert config.max_speed == 10
    assert config.max_distance == 100
    assert config.max_overlap == 0.2
    assert config.max_bond_ratio == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dt", 0),
        ("dt", np.nan),
        ("dt", True),
        ("drag", -1),
        ("spring_stiffness", np.inf),
        ("repulsion_stiffness", 0),
        ("max_strain", 1),
        ("max_strain", -0.1),
        ("max_cells", 0),
        ("max_cells", 1.5),
        ("max_cells", True),
        ("max_speed", 0),
        ("max_distance", np.inf),
        ("max_overlap", 1),
        ("max_overlap", -0.1),
        ("max_bond_ratio", 1),
    ],
)
def test_invalid_config_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        replace(InertialPhysicsConfig(), **{name: value})


def test_semi_implicit_euler_uses_new_velocity():
    a = particle("a", (0, 0), (1, 0))
    result = InertialPhysicsSolver().step([a])
    np.testing.assert_allclose(a.velocity, [0.99, 0], atol=0, rtol=0)
    np.testing.assert_allclose(a.position, [0.0099, 0], atol=1e-17)
    assert result.max_speed == 1


def test_neutral_physical_bond_does_not_depend_on_message_edges():
    def trial(message_edges):
        a = particle("a", (-0.6, 0))
        b = particle("b", (0.6, 0))
        a.conn_out = message_edges
        b.conn_out = {"a": 20}
        InertialPhysicsSolver([PhysicalBond("a", "b")]).step([a, b])
        return np.array([a.position, b.position]), np.array([a.velocity, b.velocity])

    positions, velocities = trial({})
    np.testing.assert_allclose(velocities, [[0.02, 0], [-0.02, 0]])
    other_positions, other_velocities = trial({"b": 999, "missing": -1})
    np.testing.assert_array_equal(positions, other_positions)
    np.testing.assert_array_equal(velocities, other_velocities)
    a = particle("a", (0, 0))
    b = particle("b", (1.2, 0))
    a.conn_out["b"] = 1
    InertialPhysicsSolver().step([a, b])
    np.testing.assert_array_equal(a.velocity, [0, 0])


def test_commands_are_averaged_and_set_target_length_without_accumulation():
    a = particle("a", (0, 0))
    b = particle("b", (1, 0))
    solver = InertialPhysicsSolver([PhysicalBond("a", "b")])
    solver.step([a, b], {"a": 1, "b": -1})
    np.testing.assert_array_equal(a.velocity, [0, 0])
    solver.step([a, b], {"a": 1})
    np.testing.assert_allclose(a.velocity, [-0.01, 0])
    np.testing.assert_allclose(b.velocity, [0.01, 0])
    a.position[:] = [0, 0]
    b.position[:] = [1, 0]
    a.velocity[:] = b.velocity[:] = 0
    solver.step([a, b], {"a": 1})
    np.testing.assert_allclose(a.velocity, [-0.01, 0])


def test_soft_contact_has_no_tangential_friction():
    a = particle("a", (-0.475, 0), (0, 1))
    b = particle("b", (0.475, 0), (0, -1))
    result = InertialPhysicsSolver().step([a, b])
    np.testing.assert_allclose(a.velocity, [-0.05, 0.99])
    np.testing.assert_allclose(b.velocity, [0.05, -0.99])
    assert result.max_overlap == pytest.approx(0.05)


def test_forces_are_simultaneous_and_order_independent():
    def trial(reverse):
        cells = [
            particle("a", (-1, 0), (0.2, 0)),
            particle("b", (0, 0)),
            particle("c", (1, 0)),
        ]
        bonds = [PhysicalBond("a", "b"), PhysicalBond("b", "c")]
        if reverse:
            cells.reverse()
            bonds = [PhysicalBond(b.second, b.first) for b in reversed(bonds)]
        solver = InertialPhysicsSolver(bonds)
        for _ in range(10):
            solver.step(cells, {"b": 0.7})
        return {c.id: (c.position.copy(), c.velocity.copy()) for c in cells}

    first, second = trial(False), trial(True)
    for cell_id in first:
        np.testing.assert_array_equal(first[cell_id], second[cell_id])


@pytest.mark.parametrize(
    "bonds",
    [
        [PhysicalBond("a", "b"), PhysicalBond("b", "a")],
        [PhysicalBond("a", "missing")],
    ],
)
def test_duplicate_or_dangling_physical_bonds_are_rejected(bonds):
    with pytest.raises(ValueError):
        InertialPhysicsSolver(bonds).step(
            [particle("a", (0, 0)), particle("b", (1, 0))]
        )


def test_self_bond_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        PhysicalBond("a", "a")
    with pytest.raises(ValueError, match="id"):
        InertialPhysicsSolver().step([particle("a", (0, 0)), particle("a", (1, 0))])


@pytest.mark.parametrize("commands", [{"a": np.nan}, {"a": np.inf}, {"a": 2}, {"x": 0}])
def test_solver_rejects_invalid_interpreted_commands(commands):
    with pytest.raises(ValueError):
        InertialPhysicsSolver().step([particle("a", (0, 0))], commands)


@pytest.mark.parametrize(
    ("positions", "velocity", "bonds", "reason"),
    [
        ([(0, 0), (0.79, 0)], (0, 0), (), "overlap_limit"),
        ([(0, 0), (0, 0)], (0, 0), (), "overlap_limit"),
        ([(0, 0), (2.01, 0)], (0, 0), (PhysicalBond("a", "b"),), "bond_limit"),
        ([(0, 0)], (10.01, 0), (), "speed_limit"),
        ([(100.01, 0)], (0, 0), (), "distance_limit"),
        ([(np.nan, 0)], (0, 0), (), "nonfinite_state"),
    ],
)
def test_initial_unsafe_states_fail_explicitly(positions, velocity, bonds, reason):
    cells = [
        particle(chr(ord("a") + i), position, velocity)
        for i, position in enumerate(positions)
    ]
    with pytest.raises(PhysicsFailure) as caught:
        InertialPhysicsSolver(bonds).step(cells)
    assert caught.value.reason == reason
    assert caught.value.stage == "before"
    assert caught.value.cell_ids


def test_safety_bounds_are_inclusive():
    solver = InertialPhysicsSolver()
    solver.inspect([particle("a", (100, 0), (10, 0))])
    solver.inspect([particle("a", (0, 0)), particle("b", (0.8, 0))])
    InertialPhysicsSolver([PhysicalBond("a", "b")]).inspect(
        [particle("a", (0, 0)), particle("b", (2, 0))]
    )


def test_candidate_failure_keeps_last_safe_state_and_exposes_candidate():
    a = particle("a", (99.99, 0), (2, 0))
    with pytest.raises(PhysicsFailure) as caught:
        InertialPhysicsSolver().step([a])
    failure = caught.value
    assert failure.reason == "distance_limit"
    assert failure.stage == "after"
    np.testing.assert_array_equal(a.position, [99.99, 0])
    assert failure.positions[0, 0] > 100
    assert failure.velocities[0, 0] == pytest.approx(1.98)


def test_cell_count_limit_and_empty_state():
    config = replace(InertialPhysicsConfig(), max_cells=1)
    solver = InertialPhysicsSolver(config=config)
    assert solver.step([]).max_speed == 0
    with pytest.raises(PhysicsFailure) as caught:
        solver.step([particle("a", (0, 0)), particle("b", (1, 0))])
    assert caught.value.reason == "cell_limit"


@pytest.mark.parametrize(
    "attribute,value", [("mass", 0), ("radius", -1), ("mass", np.inf)]
)
def test_invalid_particle_parameters(attribute, value):
    a = particle("a", (0, 0))
    setattr(a, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        InertialPhysicsSolver().step([a])


def test_nonfinite_calculation_is_not_saturated_or_hidden():
    a = particle("a", (0, 0), (1, 0))
    a.mass = 1e-310
    with pytest.raises(PhysicsFailure) as caught:
        InertialPhysicsSolver().step([a])
    assert caught.value.reason == "nonfinite_calculation"
    np.testing.assert_array_equal(a.position, [0, 0])
    np.testing.assert_array_equal(a.velocity, [1, 0])


def test_four_physical_neighbor_limit_is_independent_of_messages():
    cells = [particle("center", (0, 0))] + [
        particle(f"leaf{i}", (1, 0)) for i in range(5)
    ]
    bonds = [PhysicalBond("center", c.id) for c in cells[1:]]
    with pytest.raises(ValueError, match="degree"):
        InertialPhysicsSolver(bonds).inspect(cells)


def test_failed_post_update_exposes_unclipped_speed():
    a = particle("a", (0, 0))
    b = particle("b", (2, 0))
    solver = InertialPhysicsSolver(
        [PhysicalBond("a", "b")],
        config=replace(InertialPhysicsConfig(), dt=1),
    )
    with pytest.raises(PhysicsFailure) as caught:
        solver.step([a, b], {"a": -1, "b": -1})
    assert caught.value.reason == "speed_limit"
    assert caught.value.stage == "after"
    assert np.max(np.abs(caught.value.velocities)) > 10
    np.testing.assert_array_equal(a.velocity, [0, 0])
