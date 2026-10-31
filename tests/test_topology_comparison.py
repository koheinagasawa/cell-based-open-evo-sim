import json
from itertools import combinations

import numpy as np
import pytest

from experiments.physics_diagnostic import runner


def test_topology_initial_conditions_and_b_only_drive_are_explicit():
    expected = {
        "chain": {("A", "B"), ("B", "C"), ("C", "F"), ("E", "F"), ("D", "E")},
        "branched": {("A", "B"), ("B", "C"), ("A", "D"), ("B", "E"), ("C", "F")},
    }
    for body, edges in expected.items():
        for suffix in ("", "_off"):
            case = "topology_" + body + suffix
            cells, bonds = runner._initial_case(case)
            np.testing.assert_array_equal(
                [c.position for c in cells],
                [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            )
            assert [c.id for c in cells] == list("ABCDEF")
            assert {(b.first, b.second) for b in bonds} == edges
            assert all(
                np.linalg.norm(
                    cells[ord(b.first) - 65].position
                    - cells[ord(b.second) - 65].position
                )
                == 1
                for b in bonds
            )
            for interval, command in (
                (0, 1),
                (29, 1),
                (30, -1),
                (59, -1),
                (60, 0),
                (99, 0),
            ):
                commands = runner.commands_for_interval(case, interval, cells)
                assert commands == {
                    c.id: (command if c.id == "B" and not suffix else 0) for c in cells
                }


def test_topology_suite_saves_eight_runs_and_exact_response_comparison(test_output_dir):
    out = test_output_dir / "topology"
    runner.run_diagnostics(out, suite="topology")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["completed_rollouts"] == 8
    assert status["status"] == "completed"
    assert len(summary["cases"]) == 4
    assert summary["passed"] and summary["mechanism"]["passed"]
    responses = {}
    for body in ("chain", "branched"):
        for dt in ("dt_0.01", "dt_0.005"):
            positions = []
            for suffix in ("", "_off"):
                directory = out / ("topology_" + body + suffix) / dt
                with np.load(directory / "frames.npz", allow_pickle=False) as frames:
                    assert frames["position"].shape == (101, 6, 2)
                    assert frames["velocity"].shape == (101, 6, 2)
                    assert frames["command"].shape == (101, 6)
                    assert not frames["command"][:, [0, 2, 3, 4, 5]].any()
                    positions.append(frames["position"])
                assert (directory / "metrics.csv").exists()
                assert (directory / "events.jsonl").exists()
            pairs = list(combinations(range(6), 2))
            distances = [
                np.stack(
                    [np.linalg.norm(p[:, j] - p[:, i], axis=1) for i, j in pairs],
                    axis=1,
                )
                for p in positions
            ]
            responses[body, dt] = distances[0] - distances[1]
    with np.load(out / "responses.npz", allow_pickle=False) as data:
        assert data["pairs"].shape == (15, 2)
        for (body, dt), response in responses.items():
            np.testing.assert_array_equal(data[body + "_" + dt], response)
        difference = responses["chain", "dt_0.005"] - responses["branched", "dt_0.005"]
        np.testing.assert_array_equal(data["body_difference"], difference)
    noise = sum(
        np.max(np.abs(responses[body, "dt_0.01"] - responses[body, "dt_0.005"]))
        for body in ("chain", "branched")
    )
    signal = np.max(np.abs(difference))
    assert summary["mechanism"]["max_body_response_difference"] == signal
    assert summary["mechanism"]["sum_step_response_differences"] == noise
    assert signal > 0.01 and signal > 2 * noise
    assert (out / "topology_comparison.png").stat().st_size > 1000
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["suite"] == "topology"
    assert len(metadata["cases"]) == 4


def test_topology_acceptance_failure_is_saved_not_ignored(test_output_dir, monkeypatch):
    original = runner._topology_summary

    def reject(out):
        result = original(out)
        result["passed"] = False
        return result

    monkeypatch.setattr(runner, "_topology_summary", reject)
    out = test_output_dir / "rejected"
    with pytest.raises(FloatingPointError, match="acceptance criteria"):
        runner.run_diagnostics(out, suite="topology")
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
    assert not json.loads((out / "summary.json").read_text())["passed"]
    assert (out / "responses.npz").exists()


def test_unknown_suite_is_rejected_before_creating_output(test_output_dir):
    out = test_output_dir / "unknown"
    with pytest.raises(ValueError, match="suite"):
        runner.run_diagnostics(out, suite="unknown")
    assert not out.exists()
