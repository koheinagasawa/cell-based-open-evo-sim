import json

import numpy as np
import pytest

from experiments.physics_diagnostic import runner
from simulation.physics.inertial import PhysicsFailure


def test_approved_suite_saves_all_cases_comparisons_and_visuals(test_output_dir):
    out = test_output_dir / "diagnostics"
    assert runner.run_diagnostics(out) == out
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["completed_rollouts"] == 10
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    assert len(summary["cases"]) == 5
    for case in summary["cases"]:
        assert case["max_position_difference"] <= 0.01
        assert case["max_velocity_difference"] <= 0.05
        for step in ("dt_0.01", "dt_0.005"):
            directory = out / case["name"] / step
            with np.load(directory / "frames.npz", allow_pickle=False) as frames:
                np.testing.assert_allclose(frames["time"], np.arange(101) * 0.1)
                assert frames["position"].shape[0] == 101
                assert np.isfinite(frames["velocity"]).all()
                assert frames["command"].shape == frames["position"].shape[:2]
            assert (directory / "metrics.csv").stat().st_size > 0
            assert (directory / "events.jsonl").stat().st_size > 0
        with np.load(out / case["name"] / "comparison.npz", allow_pickle=False) as data:
            assert data["position_difference"].shape[0] == 101
    assert (out / "diagnostics.png").stat().st_size > 1000
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["physics"]["mass"] == 1
    assert metadata["physics"]["radius"] == 0.5
    assert metadata["timeout_seconds"] == 300
    assert metadata["source_hashes"]
    assert metadata["versions"]["numpy"] == np.__version__


def test_refuses_to_overwrite_an_existing_directory(test_output_dir):
    with pytest.raises(FileExistsError):
        runner.run_diagnostics(test_output_dir)


def test_physics_failure_is_saved_and_raised(test_output_dir, monkeypatch):
    def fail(self, cells, commands=None):
        raise PhysicsFailure(
            "speed_limit",
            "before",
            tuple(c.id for c in cells),
            np.array([c.position for c in cells]),
            np.array([c.velocity for c in cells]),
        )

    monkeypatch.setattr(runner.InertialPhysicsSolver, "step", fail)
    out = test_output_dir / "failure"
    with pytest.raises(PhysicsFailure, match="speed_limit"):
        runner.run_diagnostics(out)
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["reason"] == "speed_limit"
    directory = out / "free_decay" / "dt_0.01"
    with np.load(directory / "failure.npz", allow_pickle=False) as data:
        assert data["last_safe_position"].shape == (1, 2)
        assert data["failed_velocity"].shape == (1, 2)
    events = [
        json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "failure"
    assert events[-1]["stage"] == "before"


def test_timeout_is_not_reported_as_a_success(test_output_dir):
    out = test_output_dir / "timeout"
    with pytest.raises(TimeoutError):
        runner.run_diagnostics(out, timeout_seconds=1e-12)
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["reason"] == "timeout"


def test_discrete_control_schedule_does_not_shift_at_substep_boundaries():
    assert runner.drive_for_interval("active_spring", 29) == 1
    assert runner.drive_for_interval("active_spring", 30) == -1
    assert runner.drive_for_interval("active_spring", 59) == -1
    assert runner.drive_for_interval("active_spring", 60) == 0
    assert runner.drive_for_interval("active_spring_off", 0) == 0
    with pytest.raises(ValueError):
        runner.drive_for_interval("unknown", 0)


def test_independent_case_runs_reproduce_semantic_frames(test_output_dir):
    for name in ("first", "second"):
        runner._run_case("active_spring", 0.01, test_output_dir / name, 300)
    with (
        np.load(test_output_dir / "first" / "frames.npz", allow_pickle=False) as first,
        np.load(
            test_output_dir / "second" / "frames.npz", allow_pickle=False
        ) as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])


def test_initial_inspection_failure_preserves_evidence(test_output_dir, monkeypatch):
    def fail(self, cells):
        raise PhysicsFailure(
            "overlap_limit",
            "before",
            tuple(c.id for c in cells),
            np.array([c.position for c in cells]),
            np.array([c.velocity for c in cells]),
        )

    monkeypatch.setattr(runner.InertialPhysicsSolver, "inspect", fail)
    out = test_output_dir / "initial_failure"
    with pytest.raises(PhysicsFailure, match="overlap_limit"):
        runner.run_diagnostics(out)
    with np.load(
        out / "free_decay" / "dt_0.01" / "failure.npz", allow_pickle=False
    ) as data:
        assert data["last_completed_position"].shape == (0, 2)
        assert data["failed_position"].shape == (1, 2)


def test_failed_acceptance_is_not_a_completed_run(test_output_dir, monkeypatch):
    original = runner._compare_case

    def reject(name, out):
        result = original(name, out)
        result["passed"] = False
        return result

    monkeypatch.setattr(runner, "_compare_case", reject)
    out = test_output_dir / "failed_acceptance"
    with pytest.raises(FloatingPointError, match="acceptance criteria"):
        runner.run_diagnostics(out)
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert not summary["passed"]


def test_timeout_saves_latest_completed_physics_state(test_output_dir, monkeypatch):
    readings = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(readings))
    out = test_output_dir / "partial_timeout"
    with pytest.raises(TimeoutError):
        runner._run_case("free_decay", 0.01, out, 1)
    with np.load(out / "failure.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["last_safe_position"], [[0.0099, 0]])
        np.testing.assert_array_equal(
            data["last_safe_position"], data["failed_position"]
        )
    events = [
        json.loads(row) for row in (out / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["last_safe_time"] == 0.01
