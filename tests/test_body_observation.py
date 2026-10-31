import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.body_observation import runner
from simulation.cppn import CPPNGenome, NodeGene
from simulation.minimal_organism import MinimalOrganismWorld


def test_initialization_has_approved_topology_and_replays_rng():
    first = runner.initial_genome(0)
    assert first == runner.initial_genome(0)
    assert first != runner.initial_genome(1)
    assert first.input_size == 18 and first.output_size == 8
    assert [n.activation for n in first.nodes[:8]] == ["tanh"] * 4 + ["sin"] * 2 + [
        "gaussian"
    ] * 2
    assert [n.activation for n in first.nodes[8:]] == ["identity"] * 8
    assert len(first.connections) == 18 * 8 + 8 * 8
    assert all(-0.5 <= n.bias <= 0.5 for n in first.nodes)
    for edge in first.connections:
        n = 18 if edge.target < 26 else 8
        assert abs(edge.weight) <= 1 / np.sqrt(n)


def test_short_observations_preserve_all_30_trials_and_replay(test_output_dir):
    out = test_output_dir / "observations"
    config = replace(runner.ObservationConfig(), control_steps=3)
    runner.run_observations(out, config=config)
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["rollouts"]) == 30
    assert all(r["status"] == "completed" for r in summary["rollouts"])
    assert summary["replay_matches"]
    assert len(summary["comparisons"]) == 10
    assert (out / "observations.png").stat().st_size > 1000
    for seed in range(5):
        genotype = json.loads((out / f"seed_{seed}" / "genotype.json").read_text())
        assert CPPNGenome.from_dict(genotype) == runner.initial_genome(seed)
        for condition in ("active", "drive_off"):
            for run in ("normal", "replay", "half_step"):
                directory = out / f"seed_{seed}" / condition / run
                status = json.loads((directory / "status.json").read_text())
                assert status["completed_control_intervals"] == 3
                frames = [
                    json.loads(x)
                    for x in (directory / "frames.jsonl").read_text().splitlines()
                ]
                assert len(frames) == 4
                assert frames[0]["cells"][0]["energy"] == 63
                assert frames[0]["cells"][0]["state"] == [0] * 4
                assert (directory / "events.jsonl").exists()
                assert (directory / "metrics.csv").exists()


def test_failure_is_preserved_without_skipping_remaining_trials(
    test_output_dir, monkeypatch
):
    original = MinimalOrganismWorld.step
    calls = 0

    def fail_first(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("injected numerical failure")
        return original(self, **kwargs)

    monkeypatch.setattr(MinimalOrganismWorld, "step", fail_first)
    out = test_output_dir / "failed_trial"
    runner.run_observations(
        out, config=replace(runner.ObservationConfig(), control_steps=1)
    )
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["rollouts"]) == 30
    assert len([r for r in summary["rollouts"] if r["status"] == "failed"]) == 1
    assert (
        json.loads((out / "status.json").read_text())["status"]
        == "completed_with_failures"
    )
    assert not summary["replay_matches"]
    failed = out / "seed_0" / "active" / "normal"
    assert (failed / "failure.json").exists()
    assert (
        json.loads((failed / "status.json").read_text())["reason"]
        == "numerical_failure"
    )


def test_timeout_after_birth_reservation_does_not_spend_budget(monkeypatch):
    nodes = tuple(
        NodeGene(18 + i, "identity", b) for i, b in enumerate([0, 0, 0, 0, 1, 1, 0, 0])
    )
    world = MinimalOrganismWorld(CPPNGenome(18, 8, nodes, (), tuple(range(18, 26))))
    readings = iter([0.0, 2.0])
    monkeypatch.setattr(
        "simulation.minimal_organism.time.monotonic", lambda: next(readings)
    )
    with pytest.raises(TimeoutError):
        world.step(deadline=1)
    assert world.failed and world.failure["reason"] == "timeout"
    assert world.cells[0].energy == 63
    assert len(world.cells) == 1 and not world.pending_births
    assert any(e.get("status") == "unexecuted" for e in world.events)


def test_nonfinite_failure_evidence_is_tagged_not_replaced():
    encoded = runner.json_record({"value": float("nan"), "velocity": [float("inf")]})
    result = json.loads(encoded)
    assert result["value"] == {"nonfinite": "NaN"}
    assert result["velocity"] == [{"nonfinite": "Infinity"}]


def test_existing_directory_is_not_overwritten(test_output_dir):
    with pytest.raises(FileExistsError):
        runner.run_observations(test_output_dir)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("control_steps", 0),
        ("control_steps", 1001),
        ("control_steps", True),
        ("control_steps", 1.5),
        ("timeout_seconds", 0),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", True),
    ],
)
def test_invalid_observation_config_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        replace(runner.ObservationConfig(), **{name: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_deadline_is_rejected_before_control_changes(value):
    world = MinimalOrganismWorld(runner.initial_genome(0))
    before = world.snapshot()
    with pytest.raises(ValueError, match="deadline"):
        world.step(deadline=value)
    assert world.snapshot() == before


def test_replay_comparison_includes_failure_state(test_output_dir, monkeypatch):
    nodes = tuple(NodeGene(18 + i, "identity") for i in range(8))
    genotype = CPPNGenome(18, 8, nodes, (), tuple(range(18, 26))).to_dict()

    def fail(self, **kwargs):
        raise FloatingPointError("repeated injected failure: \u6570\u5024\u5931\u6557")

    monkeypatch.setattr(MinimalOrganismWorld, "step", fail)
    base = test_output_dir / "seed_0" / "active"
    for run, dt in runner.RUNS:
        runner._run_trial(
            genotype, 0, False, dt, base / run, runner.ObservationConfig()
        )
    assert runner._compare(test_output_dir, 0, "active")["replay_matches"]
    failure = json.loads((base / "replay" / "failure.json").read_text(encoding="utf-8"))
    failure["current_state"]["cells"][0]["state"][0] = 99
    (base / "replay" / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
    assert not runner._compare(test_output_dir, 0, "active")["replay_matches"]
