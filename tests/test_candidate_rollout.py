import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from experiments.body_observation import runner
from simulation.cppn import CPPNGenome, NodeGene


def growing_genome():
    values = [0, 0, 0, 0, 1, 1, 0, 0]
    return CPPNGenome(
        18,
        8,
        tuple(NodeGene(18 + i, "identity", value) for i, value in enumerate(values)),
        (),
        tuple(range(18, 26)),
    )


def test_candidate_result_aggregates_agent_and_preserves_artifacts(test_output_dir):
    out = test_output_dir / "candidate"
    result = runner.run_candidate(
        "candidate-1",
        growing_genome(),
        seed=7,
        out_dir=out,
        config=replace(runner.ObservationConfig(), control_steps=3),
    )
    assert isinstance(result, runner.CandidateResult)
    assert result.candidate_id == "candidate-1"
    assert result.status == "completed" and result.reason is None
    assert result.completed_control_intervals == 3
    assert result.births == 3
    assert result.final_cells == result.max_cells == 4
    assert result.growth_budget == 60
    assert result.physics_diagnostics.max_speed == 0
    assert result.out_dir == out
    with pytest.raises(FrozenInstanceError):
        result.final_cells = 999
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert saved == result.to_dict()
    assert (
        json.loads((out / "config.json").read_text())["candidate_id"] == "candidate-1"
    )
    assert (
        json.loads((out / "status.json").read_text())["candidate_id"] == "candidate-1"
    )
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "candidate"
    assert metadata["candidate_id"] == "candidate-1"
    assert metadata["seeds"] == [7]
    assert metadata["source_snapshot"]
    assert "no random initialization" in metadata["initialization"]["method"]
    assert (out / "frames.jsonl").exists() and (out / "events.jsonl").exists()
    assert not any("fitness" in name or "score" in name for name in saved)


def test_candidate_redevelopment_is_independent_and_does_not_mutate_parent(
    test_output_dir, monkeypatch
):
    parent = runner.initial_genome(0)
    original_genotype = parent.to_dict()
    worlds = []
    original = runner.MinimalOrganismWorld

    def capture(*args, **kwargs):
        world = original(*args, **kwargs)
        worlds.append(world)
        return world

    monkeypatch.setattr(runner, "MinimalOrganismWorld", capture)
    config = replace(runner.ObservationConfig(), control_steps=3)
    first = runner.run_candidate(
        "first", parent, seed=0, out_dir=test_output_dir / "first", config=config
    )
    second = runner.run_candidate(
        "second",
        CPPNGenome.from_dict(parent.to_dict()),
        seed=0,
        out_dir=test_output_dir / "second",
        config=config,
    )
    assert parent.to_dict() == original_genotype
    assert worlds[0] is not worlds[1]
    assert worlds[0].agent is not worlds[1].agent
    assert worlds[0].cells[0] is not worlds[1].cells[0]
    assert worlds[0].cells[0].interpreter is not worlds[1].cells[0].interpreter
    assert worlds[0].snapshot() == worlds[1].snapshot()
    assert worlds[0].events == worlds[1].events
    worlds[0].cells[0].state[0] = 99
    assert worlds[1].cells[0].state[0] != 99
    assert first.births == second.births
    assert first.physics_diagnostics == second.physics_diagnostics


def test_failed_candidate_is_an_explicit_result_not_an_evaluation_score(
    test_output_dir, monkeypatch
):
    def fail(self, **kwargs):
        raise FloatingPointError("injected numerical failure")

    monkeypatch.setattr(runner.MinimalOrganismWorld, "step", fail)
    out = test_output_dir / "failed"
    result = runner.run_candidate(
        "failed-candidate",
        growing_genome(),
        seed=0,
        out_dir=out,
        config=replace(runner.ObservationConfig(), control_steps=2),
    )
    assert result.status == "failed"
    assert result.reason == "numerical_failure"
    assert result.completed_control_intervals == 0
    assert result.births == 0
    assert result.final_cells == 1
    assert result.growth_budget == 63
    assert (out / "failure.json").exists()
    assert json.loads((out / "result.json").read_text())["status"] == "failed"


def test_observation_runner_uses_public_candidate_entry(test_output_dir, monkeypatch):
    original = runner.run_candidate
    inputs = []

    def capture(candidate_id, genotype, **kwargs):
        inputs.append((candidate_id, genotype.to_dict(), kwargs["seed"]))
        return original(candidate_id, genotype, **kwargs)

    monkeypatch.setattr(runner, "run_candidate", capture)
    runner.run_observations(
        test_output_dir / "observations",
        config=replace(runner.ObservationConfig(), control_steps=1),
    )
    assert len(inputs) == 30
    assert len({row[0] for row in inputs}) == 30
    for seed in range(5):
        assert all(
            data == runner.initial_genome(seed).to_dict()
            for _, data, s in inputs
            if s == seed
        )


@pytest.mark.parametrize("candidate_id", ["", "   ", None, 3])
def test_invalid_candidate_id_fails_before_creating_output(
    test_output_dir, candidate_id
):
    out = test_output_dir / "invalid"
    with pytest.raises(ValueError, match="candidate_id"):
        runner.run_candidate(candidate_id, growing_genome(), seed=0, out_dir=out)
    assert not out.exists()


def test_candidate_result_maxima_match_saved_metrics(test_output_dir):
    out = test_output_dir / "metrics"
    result = runner.run_candidate(
        "metrics",
        runner.initial_genome(0),
        seed=0,
        out_dir=out,
        config=replace(runner.ObservationConfig(), control_steps=5),
    )
    with np.load(out / "metrics.npz", allow_pickle=False) as metrics:
        assert result.max_cells == int(metrics["cells"].max())
        assert result.growth_budget == metrics["growth_budget"][-1]
        assert result.physics_diagnostics.max_speed == metrics["max_speed"].max()
        assert result.physics_diagnostics.max_overlap == metrics["max_overlap"].max()
        assert (
            result.physics_diagnostics.max_bond_length
            == metrics["max_bond_length"].max()
        )


def test_provenance_io_failure_propagates_with_execution_state_preserved(
    test_output_dir, monkeypatch
):
    def fail(*args, **kwargs):
        raise OSError("injected provenance write failure")

    monkeypatch.setattr(runner, "_provenance", fail)
    out = test_output_dir / "provenance_failure"
    with pytest.raises(OSError, match="provenance"):
        runner.run_candidate(
            "provenance",
            growing_genome(),
            seed=0,
            out_dir=out,
            config=replace(runner.ObservationConfig(), control_steps=1),
        )
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["execution_status"] == "completed"
    assert (out / "frames.jsonl").exists()


def test_unobserved_candidate_diagnostics_are_missing_not_zero(
    test_output_dir, monkeypatch
):
    def fail(*args, **kwargs):
        raise ValueError("injected initialization failure")

    monkeypatch.setattr(runner, "MinimalOrganismWorld", fail)
    result = runner.run_candidate(
        "initialization",
        growing_genome(),
        seed=0,
        out_dir=test_output_dir / "initialization",
        config=replace(runner.ObservationConfig(), control_steps=1),
    )
    assert result.status == "failed"
    assert result.physics_diagnostics is None
    assert result.growth_budget is None
