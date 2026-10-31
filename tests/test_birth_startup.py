import json
from dataclasses import replace

import pytest

from experiments.body_observation import runner


def test_startup_uses_one_hundred_unselected_seeds_and_stops_at_first_birth(
    test_output_dir,
):
    out = test_output_dir / "startup"
    runner.run_startup_diagnostic(
        out, config=replace(runner.ObservationConfig(), control_steps=2)
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["individuals"]) == 100
    assert [r["seed"] for r in summary["individuals"]] == list(range(100))
    assert all(r["status"] == "completed" for r in summary["individuals"])
    first = summary["individuals"][0]
    assert first["ended_by"] == "first_birth"
    assert first["births"] == 1
    assert first["completed_control_intervals"] == 1
    assert first["gate_evaluations"] == 1
    assert first["initial_gate"] == first["max_gate"]
    assert summary["startup_count"] >= 1
    assert summary["no_startup_count"] >= 1
    assert summary["failed_count"] == summary["not_run_count"] == 0
    assert summary["startup_fraction_of_completed"] == summary["startup_count"] / 100
    assert (out / "startup.png").stat().st_size > 1000
    assert (out / "startup.csv").exists()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "startup"
    assert metadata["seeds"] == list(range(100))
    assert metadata["conditions"] == ["active"]
    assert metadata["total_timeout_seconds"] == 300
    for row in summary["individuals"]:
        assert (out / f"seed_{row['seed']}" / "genotype.json").exists()
        assert row["max_gate"] is not None
        if row["ended_by"] == "first_birth":
            assert row["final_cells"] == 2
            assert row["gate_requests"] >= 1
        else:
            assert row["ended_by"] == "observation_limit"
            assert row["completed_control_intervals"] == 2


def test_global_timeout_marks_remaining_seeds_unexecuted(test_output_dir):
    out = test_output_dir / "timeout"
    runner.run_startup_diagnostic(
        out,
        config=replace(runner.ObservationConfig(), timeout_seconds=1e-12),
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["individuals"]) == 100
    assert summary["not_run_count"] == 100
    assert summary["startup_fraction_of_completed"] is None
    assert json.loads((out / "status.json").read_text())["status"] == "incomplete"
    assert all(r["reason"] == "global_timeout" for r in summary["individuals"])


def test_first_birth_mode_does_not_change_full_observation_behavior(test_output_dir):
    config = replace(runner.ObservationConfig(), control_steps=2)
    genotype = runner.initial_genome(0).to_dict()
    short = runner._run_trial(
        genotype,
        0,
        False,
        0.01,
        test_output_dir / "startup",
        config,
        stop_on_first_birth=True,
    )
    full = runner._run_trial(
        genotype,
        0,
        False,
        0.01,
        test_output_dir / "observation",
        config,
    )
    assert short["completed_control_intervals"] == 1
    assert full["completed_control_intervals"] == 2
    assert short["initial_gate"] == full["initial_gate"]
    assert full["birth_rejections"]["birth_overlap"] == 1


def test_startup_output_cannot_overwrite_previous_evidence(test_output_dir):
    with pytest.raises(FileExistsError):
        runner.run_startup_diagnostic(test_output_dir)
