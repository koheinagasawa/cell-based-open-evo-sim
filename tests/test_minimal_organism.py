import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from simulation.cell import Cell
from simulation.cppn import ConnectionGene, CPPNGenome, NodeGene
from simulation.minimal_organism import (
    MinimalOrganismInterpreter,
    MinimalOrganismWorld,
)
from simulation.physics.inertial import (
    InertialPhysicsSolver,
    PhysicalBond,
    PhysicsDiagnostics,
    PhysicsFailure,
)


def genome(*, gate=1, direction=(1, 0), contraction=0, directional_state=False):
    biases = [1, 0, 0, 0, gate, *direction, contraction]
    nodes = tuple(NodeGene(18 + i, "identity", b) for i, b in enumerate(biases))
    connections = [ConnectionGene(2, 18, 1)]
    if directional_state:
        connections += [ConnectionGene(2, 23, 1), ConnectionGene(3, 24, 1)]
    return CPPNGenome(18, 8, nodes, connections, tuple(range(18, 26)))


def births(world, status):
    return [e for e in world.events if e["event"] == "birth" and e["status"] == status]


def trace_metadata(world):
    root = Path(__file__).resolve().parents[1]
    simulation = Path("simulation")
    paths = [
        simulation / name
        for name in (
            "minimal_organism.py",
            "world.py",
            "cell.py",
            "agent.py",
            "cppn.py",
            "interpreter.py",
            "policies.py",
        )
    ] + [
        simulation / "physics" / "inertial.py",
        simulation / "physics" / "repulsion.py",
        Path("tests") / "test_minimal_organism.py",
    ]
    sources = {str(path): (root / path).read_text(encoding="utf-8") for path in paths}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "genotype": world.agent.genome.to_dict(),
        "interpreter": world.agent.interpreter.to_metadata(),
        "seed": world.seed,
        "drive_off": world.drive_off,
        "physics": asdict(world.physics_solver.config),
        "revision": revision,
        "command": sys.argv,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "source_hashes": {
            path: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for path, text in sources.items()
        },
        "source_snapshot": sources,
        "purpose": "deterministic integration fixture, not evolutionary evidence",
    }


def test_seed_initialization_and_shared_rules():
    g = genome()
    world = MinimalOrganismWorld(g)
    seed = world.cells[0]
    assert len(world.cells) == len(world.agent.cells) == 1
    assert seed.genome is world.agent.genome is g
    assert seed.interpreter is world.agent.interpreter
    assert seed.profile == 0
    assert seed.energy == seed.energy_max == 63
    assert seed.mass == 1 and seed.radius == 0.5
    np.testing.assert_array_equal(seed.position, [0, 0])
    np.testing.assert_array_equal(seed.velocity, [0, 0])
    np.testing.assert_array_equal(seed.state, [0, 0, 0, 0])
    np.testing.assert_array_equal(world.inputs_for(seed.id), np.zeros(18))
    assert not world.physics_solver.bonds and not seed.conn_out
    with pytest.raises(AttributeError):
        seed.profile = 1


def test_births_share_parameters_but_reset_and_separate_cell_state(test_output_dir):
    world = MinimalOrganismWorld(genome(contraction=5), drive_off=True)
    trace = [world.snapshot()]
    world.step()
    seed, child = world.cells
    assert world.time == 1 and world.physics_steps == 10
    assert seed.energy == child.energy == 31
    assert child.birth_order == 1
    assert child.age == 0 and seed.age == 1
    assert child.genome is seed.genome
    assert child.interpreter is seed.interpreter
    assert child.profile == seed.profile == 0
    np.testing.assert_array_equal(seed.state, [1, 0, 0, 0])
    np.testing.assert_array_equal(child.state, [0, 0, 0, 0])
    np.testing.assert_array_equal(child.position, [1, 0])
    assert world.physics_solver.bonds == (PhysicalBond(seed.id, child.id),)
    assert not child.conn_out and not seed.conn_out
    assert seed.output_slots["contraction"] == 1
    trace.append(world.snapshot())
    world.step()
    assert len(world.cells) == 3
    np.testing.assert_array_equal(world.cells[0].state, [2, 0, 0, 0])
    np.testing.assert_array_equal(world.cells[1].state, [1, 0, 0, 0])
    np.testing.assert_array_equal(world.cells[2].state, [0, 0, 0, 0])
    assert [c.energy for c in world.cells] == [31, 15, 15]
    np.testing.assert_array_equal(
        [c.position for c in world.cells], [[0, 0], [1, 0], [2, 0]]
    )
    assert births(world, "rejected")[-1]["reason"] == "birth_overlap"
    trace.append(world.snapshot())
    (test_output_dir / "growth_trace.json").write_text(
        json.dumps(
            {**trace_metadata(world), "frames": trace, "events": world.events},
            allow_nan=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    child.state[0] = 100
    assert seed.state[0] == 2
    child.velocity[0] = 10
    assert seed.velocity[0] == 0


def test_birth_uses_interval_end_parent_position_and_velocity():
    world = MinimalOrganismWorld(genome())
    parent = world.cells[0]
    parent.velocity[:] = [1, 0.5]
    world.step()
    child = world.cells[1]
    np.testing.assert_allclose(parent.velocity, np.array([1, 0.5]) * 0.99**10)
    np.testing.assert_array_equal(child.velocity, parent.velocity)
    np.testing.assert_allclose(child.position, parent.position + [1, 0])
    assert child.development_position == (1, 0)
    assert parent.development_position == (0, 0)
    np.testing.assert_array_equal(world.inputs_for(child.id)[:6], [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(world.inputs_for(child.id)[6:8], [-1, 0])


def test_input_slots_follow_connection_creation_not_current_distance_or_storage_order():
    world = MinimalOrganismWorld(genome(directional_state=True))
    parent = world.cells[0]
    for state in ((0, 0), (-1, 1), (-2, 0), (-1, -1)):
        for child in world.cells[1:]:
            child.energy = 0
        parent.state[:2] = state
        world.step()
    assert len(world.cells) == 5
    world.cells[1].position[:] = [1.1, 0]
    world.cells[2].position[:] = [0, 1.2]
    world.cells[3].position[:] = [-1.3, 0]
    world.cells[4].position[:] = [0, -1.4]
    parent.state[:] = [4, 3, 2, 1]
    parent.conn_out = {"unknown": 999}
    expected = [0, 0, 4, 3, 2, 1, 1.1, 0, 0, 1.2, -1.3, 0, 0, -1.4, 1, 1, 1, 1]
    world.cells.reverse()
    world.physics_solver.bonds = tuple(reversed(world.physics_solver.bonds))
    np.testing.assert_allclose(world.inputs_for(parent.id), expected)
    parent.state[:2] = [0, 0]
    for cell in world.cells:
        if cell is not parent:
            cell.energy = 0
    world.step()
    assert len(world.cells) == 5
    assert any(e.get("reason") == "degree_limit" for e in births(world, "rejected"))


@pytest.mark.parametrize("direction", [(0, 0), (1e-300, 1e-300), (1e308, 1e308)])
def test_direction_normalization_is_finite_and_zero_does_not_spend(direction):
    world = MinimalOrganismWorld(genome(direction=direction))
    world.step()
    if direction == (0, 0):
        assert len(world.cells) == 1 and world.cells[0].energy == 63
        assert births(world, "rejected")[0]["reason"] == "zero_direction"
    else:
        np.testing.assert_allclose(world.cells[1].position, np.sqrt(0.5) * np.ones(2))


@pytest.mark.parametrize(("energy", "allowed"), [(0, False), (0.5, False), (1, True)])
def test_budget_boundary_and_no_energy_death(energy, allowed):
    world = MinimalOrganismWorld(genome())
    world.cells[0].energy = energy
    world.step()
    assert len(world.cells) == (2 if allowed else 1)
    assert world.cells[0].state[0] == 1
    if allowed:
        assert all(c.energy == 0 for c in world.cells)
    else:
        assert world.cells[0].energy == energy
        assert births(world, "rejected")[0]["reason"] == "energy_limit"


@pytest.mark.parametrize(
    ("distance", "allowed"), [(0.95, True), (0.95 - 0.5e-9, True), (0.95 - 2e-9, False)]
)
def test_birth_overlap_tolerance_is_distinct_from_physical_safety(
    distance, allowed, monkeypatch
):
    world = MinimalOrganismWorld(genome())
    world.step()
    world.cells[1].position[:] = [1 + distance, 0]
    world.cells[1].energy = 0
    old_energy = world.cells[0].energy
    monkeypatch.setattr(
        InertialPhysicsSolver,
        "step",
        lambda self, cells, commands=None: PhysicsDiagnostics(),
    )
    world.step()
    assert len(world.cells) == (3 if allowed else 2)
    if not allowed:
        assert world.cells[0].energy == old_energy
        assert births(world, "rejected")[-1]["reason"] == "birth_overlap"
    else:
        assert len(world.physics_solver.bonds) == 2


def test_failed_physics_does_not_commit_reserved_births_or_fees(monkeypatch):
    world = MinimalOrganismWorld(genome())

    def fail(cells, commands=None):
        raise PhysicsFailure(
            "speed_limit",
            "before",
            tuple(c.id for c in cells),
            np.array([c.position for c in cells]),
            np.array([c.velocity for c in cells]),
        )

    monkeypatch.setattr(world.physics_solver, "step", fail)
    with pytest.raises(PhysicsFailure):
        world.step()
    assert world.failed
    assert len(world.cells) == len(world.agent.cells) == 1
    assert world.cells[0].energy == 63
    assert world.time == world.physics_steps == 0
    assert not world.pending_births
    assert births(world, "unexecuted")[0]["reserved_cost"] == 1
    assert world.last_completed_snapshot["cells"][0]["state"] == [0, 0, 0, 0]
    assert world.failure["state"]["cells"][0]["state"] == [1, 0, 0, 0]
    with pytest.raises(RuntimeError, match="failed"):
        world.step()


def test_independent_growth_is_identical_despite_cell_storage_reversal():
    first = MinimalOrganismWorld(genome())
    second = MinimalOrganismWorld(CPPNGenome.from_dict(first.agent.genome.to_dict()))
    for _ in range(10):
        second.cells.reverse()
        first.step()
        second.step()
        assert first.snapshot() == second.snapshot()
        assert first.events == second.events
    assert len(first.cells) == 7
    assert first.physics_steps == 100
    assert first.cells[0] is not second.cells[0]


def test_cell_limit_counts_existing_and_reserved_children():
    world = MinimalOrganismWorld(genome())
    for i in range(1, 64):
        child = world._new_cell(i, (float(i), 0), (float(i), 0), 0, (0, 0))
        world.agent.cells.append(child)
        world.add_cell(child)
    world.physics_solver = InertialPhysicsSolver(
        [PhysicalBond(world.cells[i].id, world.cells[i + 1].id) for i in range(63)]
    )
    world.step()
    assert len(world.cells) == 64
    assert world.cells[0].energy == 63
    assert any(e.get("reason") == "cell_limit" for e in births(world, "rejected"))


def test_interpreter_is_strict_and_clips_only_the_command():
    interpreter = MinimalOrganismInterpreter()
    raw = np.array([1, 2, 3, 4, 0.7, 9, -8, 20], dtype=float)
    slots = interpreter.interpret(raw, profile=0)
    np.testing.assert_array_equal(slots["state"], [1, 2, 3, 4])
    np.testing.assert_array_equal(slots["bud"], [0.7, 9, -8])
    assert slots["contraction"] == 1
    raw[-1] = -20
    assert interpreter.interpret(raw, profile=0)["contraction"] == -1
    for invalid in (
        np.zeros(9),
        np.zeros((2, 4)),
        {"state": [0] * 4},
        np.full(8, np.nan),
        np.full(8, np.inf),
    ):
        with pytest.raises(ValueError):
            interpreter.interpret(invalid, profile=0)
    with pytest.raises(ValueError):
        interpreter.interpret(np.zeros(8), profile=1)


def test_world_rejects_nonmatching_genotype_and_invalid_dt():
    with pytest.raises(ValueError, match="18.*8"):
        MinimalOrganismWorld(CPPNGenome(1, 1, (NodeGene(1, "identity"),), (), (1,)))
    with pytest.raises(ValueError, match="physics_dt"):
        MinimalOrganismWorld(genome(), physics_dt=0.02)


def test_no_budding_on_exact_gate_threshold():
    world = MinimalOrganismWorld(genome(gate=0.5))
    world.step()
    assert len(world.cells) == 1
    assert world.cells[0].energy == 63
    assert not births(world, "reserved")


def test_same_interval_children_compete_in_parent_birth_order(monkeypatch):
    world = MinimalOrganismWorld(genome(directional_state=True))
    world.step()
    parent, first_child = world.cells
    parent.state[:2] = [-0.5, np.sqrt(0.75)]
    first_child.state[:2] = [-1.5, np.sqrt(0.75)]
    world.cells.reverse()
    monkeypatch.setattr(
        InertialPhysicsSolver,
        "step",
        lambda self, cells, commands=None: PhysicsDiagnostics(),
    )
    world.step()
    assert len(world.cells) == 3
    assert births(world, "born")[-1]["parent_id"] == parent.id
    assert births(world, "rejected")[-1]["parent_id"] == first_child.id
    assert births(world, "rejected")[-1]["reason"] == "birth_overlap"
    assert parent.energy == world.cells[2].energy == 15
    assert first_child.energy == 31
    assert set(world.physics_solver.bonds) == {
        PhysicalBond(parent.id, first_child.id),
        PhysicalBond(parent.id, world.cells[2].id),
    }


def test_reservations_consume_cell_capacity_before_birth():
    world = MinimalOrganismWorld(genome(direction=(-1, 0)))
    for i in range(1, 63):
        child = world._new_cell(i, (float(i), 0), (float(i), 0), 0, (0, 0))
        world.agent.cells.append(child)
        world.add_cell(child)
    world.cells[-1].energy = 1
    world._next_birth_order = 63
    world.physics_solver = InertialPhysicsSolver(
        [PhysicalBond(world.cells[i].id, world.cells[i + 1].id) for i in range(62)]
    )
    last_parent = world.cells[-1]
    world.step()
    assert len(world.cells) == 64
    assert last_parent.energy == 1
    assert any(
        e.get("parent_id") == last_parent.id and e.get("reason") == "cell_limit"
        for e in births(world, "rejected")
    )


def test_partial_physics_failure_records_physical_time_and_no_birth(monkeypatch):
    world = MinimalOrganismWorld(genome())
    original = world.physics_solver.step
    calls = 0

    def fail_on_third(cells, commands=None):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise PhysicsFailure(
                "speed_limit",
                "before",
                tuple(c.id for c in cells),
                np.array([c.position for c in cells]),
                np.array([c.velocity for c in cells]),
            )
        return original(cells, commands)

    monkeypatch.setattr(world.physics_solver, "step", fail_on_third)
    with pytest.raises(PhysicsFailure):
        world.step()
    assert world.time == 0 and world.physics_steps == 2
    assert len(world.cells) == 1 and world.cells[0].energy == 63
    assert world.failure["time"] == 0.02


def test_physical_neighbor_mask_is_evaluated_directly_by_the_cppn():
    g = genome()
    mask_gated = CPPNGenome(
        18,
        8,
        g.nodes,
        (*g.connections, ConnectionGene(14, 22, -1)),
        g.output_ids,
    )
    world = MinimalOrganismWorld(mask_gated)
    world.step()
    for _ in range(3):
        world.step()
    assert len(world.cells) == 2
    assert len(births(world, "born")) == 1
    assert all(c.raw_output[4] == 0 for c in world.cells)
    assert all(c.energy == 31 for c in world.cells)


def test_cppn_numerical_failure_is_terminal_and_not_a_default_action():
    g = genome()
    divergent = CPPNGenome(
        18,
        8,
        g.nodes,
        (ConnectionGene(2, 18, 1e308),),
        g.output_ids,
    )
    world = MinimalOrganismWorld(divergent)
    world.cells[0].state[0] = 2
    with pytest.raises(FloatingPointError):
        world.step()
    assert world.failed
    assert world.time == 0
    assert len(world.cells) == 1 and world.cells[0].energy == 63
    assert not births(world, "born")


def test_bounded_growth_trace_reaches_the_observation_limit(test_output_dir):
    world = MinimalOrganismWorld(genome())
    frames = [world.snapshot()]
    for _ in range(1000):
        world.step()
        frames.append(world.snapshot())
    assert len(world.cells) == 7
    assert world.time == 1000 and world.physics_steps == 10000
    assert sum(c.energy for c in world.cells) == 57
    assert len(world.physics_solver.bonds) == 6
    assert all(not c.conn_out for c in world.cells)
    assert not world.failed
    assert len(births(world, "born")) == 6
    (test_output_dir / "growth_trace.json").write_text(
        json.dumps(
            {
                **trace_metadata(world),
                "frames": frames,
                "events": world.events,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="observation limit"):
        world.step()


@pytest.mark.parametrize(("dt", "substeps"), [(0.01, 10), (0.005, 20)])
def test_interpreted_drive_is_held_and_matches_direct_physics(
    dt, substeps, monkeypatch
):
    world = MinimalOrganismWorld(genome(contraction=5), physics_dt=dt)
    world.step()
    parent, child = world.cells
    first = world.snapshot()
    assert first["cells"][0]["requested_contraction"] == 1
    assert first["cells"][0]["applied_contraction"] == 1
    assert first["cells"][1]["applied_contraction"] == 0
    reference_cells = [
        Cell(c.position.copy(), genome=None, id=c.id, velocity=c.velocity.copy())
        for c in world.cells
    ]
    reference = InertialPhysicsSolver(
        world.physics_solver.bonds, config=world.physics_solver.config
    )
    original = world.physics_solver.step
    received = []

    def record(cells, commands=None):
        received.append(dict(commands))
        result = original(cells, commands)
        # The physical substeps must not re-read mutable output slots.
        parent.output_slots["contraction"] = -1
        return result

    monkeypatch.setattr(world.physics_solver, "step", record)
    world.step()
    expected = {parent.id: 1.0, child.id: 1.0}
    assert received == [expected] * substeps
    for _ in range(substeps):
        reference.step(reference_cells, expected)
    for actual, direct in zip(world.cells[:2], reference_cells):
        np.testing.assert_array_equal(actual.position, direct.position)
        np.testing.assert_array_equal(actual.velocity, direct.velocity)
    assert parent.age == 2 and child.age == 1
    assert parent.genome is child.genome


def test_opposite_commands_are_clipped_per_cell_before_averaging():
    g = genome(contraction=5)
    opposite = CPPNGenome(
        18, 8, g.nodes, (*g.connections, ConnectionGene(0, 25, -10)), g.output_ids
    )
    world = MinimalOrganismWorld(opposite)
    world.step()
    world.step()
    parent, child = world.cells[:2]
    assert parent.raw_output[-1] == 5
    assert child.raw_output[-1] == -5
    np.testing.assert_array_equal(parent.position, [0, 0])
    np.testing.assert_array_equal(child.position, [1, 0])
    assert [c["applied_contraction"] for c in world.snapshot()["cells"][:2]] == [1, -1]


def test_drive_off_preserves_interpretation_and_neutral_growth():
    g = genome(contraction=5)
    driven = MinimalOrganismWorld(g)
    off = MinimalOrganismWorld(CPPNGenome.from_dict(g.to_dict()), drive_off=True)
    neutral = MinimalOrganismWorld(genome())
    for _ in range(3):
        driven.step()
        off.step()
        neutral.step()
    assert driven.cells[0].position[0] != off.cells[0].position[0]
    for actual, baseline in zip(off.cells, neutral.cells):
        np.testing.assert_array_equal(actual.position, baseline.position)
        np.testing.assert_array_equal(actual.state, baseline.state)
        assert actual.energy == baseline.energy
    assert all(c["applied_contraction"] == 0 for c in off.snapshot()["cells"])
    assert off.cells[0].output_slots["contraction"] == 1
    assert off.snapshot()["drive_off"] is True


@pytest.mark.parametrize("value", [1, 0, None, "off"])
def test_drive_off_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="drive_off"):
        MinimalOrganismWorld(genome(), drive_off=value)


def behavior_genome():
    g = genome()
    nodes = (*g.nodes, NodeGene(26, "sin"))
    connections = (
        *g.connections,
        ConnectionGene(14, 22, -1),  # Only the unconnected seed requests birth.
        ConnectionGene(2, 26, 0.1),
        ConnectionGene(26, 25, 0.5),
        ConnectionGene(6, 25, 0.1),
    )
    return CPPNGenome(18, 8, nodes, connections, g.output_ids)


def test_same_cppn_drives_growth_state_feedback_and_body_motion(test_output_dir):
    g = behavior_genome()
    active = MinimalOrganismWorld(g)
    replay = MinimalOrganismWorld(CPPNGenome.from_dict(g.to_dict()))
    off = MinimalOrganismWorld(CPPNGenome.from_dict(g.to_dict()), drive_off=True)
    frames = {"active": [active.snapshot()], "drive_off": [off.snapshot()]}
    lengths = []
    for interval in range(210):
        # Check the full current observation, including changing physical inputs.
        for cell in active.cells:
            expected = g.activate(active.inputs_for(cell.id))
            cell._expected = expected
        replay.cells.reverse()
        active.step()
        replay.step()
        off.step()
        assert active.snapshot() == replay.snapshot()
        assert active.events == replay.events
        assert len(active.cells) == len(off.cells) == 2
        for cell in active.cells:
            if hasattr(cell, "_expected"):
                np.testing.assert_array_equal(cell.raw_output, cell._expected)
            assert cell.genome is active.agent.genome
            assert cell.interpreter is active.agent.interpreter
        if interval:
            lengths.append(
                np.linalg.norm(active.cells[1].position - active.cells[0].position)
            )
        frames["active"].append(active.snapshot())
        frames["drive_off"].append(off.snapshot())
    assert np.ptp(lengths) > 0.01
    np.testing.assert_array_equal([c.position for c in off.cells], [[0, 0], [1, 0]])
    assert len(births(active, "born")) == len(births(off, "born")) == 1
    assert all(c.energy == 31 for c in active.cells)
    assert not active.failed and not off.failed
    assert active.time == off.time == 210
    assert [c.age for c in active.cells] == [210, 209]
    for mode, world in (("active", active), ("drive_off", off)):
        (test_output_dir / f"{mode}_trace.json").write_text(
            json.dumps(
                {
                    **trace_metadata(world),
                    "frames": frames[mode],
                    "events": world.events,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )


def test_drive_failure_preserves_held_command_and_reserved_birth(monkeypatch):
    world = MinimalOrganismWorld(genome(contraction=-5))
    received = []

    def fail(cells, commands=None):
        received.append(dict(commands))
        raise PhysicsFailure(
            "speed_limit",
            "before",
            tuple(c.id for c in cells),
            np.array([c.position for c in cells]),
            np.array([c.velocity for c in cells]),
        )

    monkeypatch.setattr(world.physics_solver, "step", fail)
    with pytest.raises(PhysicsFailure):
        world.step()
    assert received == [{world.cells[0].id: -1}]
    assert world.failure["state"]["cells"][0]["applied_contraction"] == -1
    assert world.cells[0].energy == 63
    assert len(world.cells) == 1
    assert births(world, "unexecuted")[0]["spent_cost"] == 0


def test_zero_growth_budget_does_not_disable_cppn_physical_drive():
    world = MinimalOrganismWorld(genome(contraction=0.5))
    world.step()
    for cell in world.cells:
        cell.energy = 0
    world.step()
    assert len(world.cells) == 2
    assert any(c.velocity[0] != 0 for c in world.cells)
    assert all(c.energy == 0 for c in world.cells)
    assert all(c["applied_contraction"] == 0.5 for c in world.snapshot()["cells"])
