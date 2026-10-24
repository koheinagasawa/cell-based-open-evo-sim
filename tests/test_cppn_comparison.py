import csv
import json
import shutil
from dataclasses import replace

import matplotlib.image as mpimg
import numpy as np
import pytest

from experiments.cppn_diagnostic import compare, runner
from experiments.cppn_diagnostic.compare import compare_suite, execute_suite


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def saved_suite(run_output_root):
    root = run_output_root / "cppn_comparison_fixture"
    template = replace(
        runner.DiagnosticConfig(), population_size=2, generations=1, selected_parents=1
    )
    execute_suite(root, seeds=(0,), template=template)
    return root


@pytest.fixture
def copied_suite(saved_suite, test_output_dir):
    root = test_output_dir / "suite"
    shutil.copytree(saved_suite, root)
    return root


def test_small_comparison_validates_artifacts_and_plot_without_more_fitness_evaluations(
    saved_suite, test_output_dir, monkeypatch
):
    def unexpected_evaluation(genome):
        raise AssertionError("comparison must not spend additional fitness budget")

    monkeypatch.setattr(runner, "evaluate_genome", unexpected_evaluation)
    out = test_output_dir / "analysis"
    summary = compare_suite(saved_suite, out)
    assert summary["status"] == "completed"
    assert summary["approved_budget"] is False
    assert summary["trials"] == 6
    assert summary["repeat_pairs"] == 3
    assert summary["individual_evaluations"] == 24
    assert summary["input_evaluations"] == 600
    assert summary["restored_genotypes"] == 24
    assert summary["replayed_offspring"] == 12
    assert summary["repeat_matches"] is True
    assert summary["initial_populations_match"] is True
    assert len(summary["per_trial"]) == 3
    no_mutation = next(
        row for row in summary["per_trial"] if row["condition"] == "no_mutation"
    )
    assert no_mutation["genotype_changed_offspring"] == 0
    assert no_mutation["error_changed_offspring"] == 0
    assert no_mutation["mutation_records"] == 0
    assert summary == read_json(out / "summary.json")
    with (out / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [float(row["final_median_error"]) for row in rows] == [
        row["final_median_error"] for row in summary["per_trial"]
    ]
    with (out / "paired.csv").open(encoding="utf-8", newline="") as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 1
    errors = {
        row["condition"]: row["final_median_error"] for row in summary["per_trial"]
    }
    assert float(paired[0]["selection_minus_random"]) == (
        errors["selection"] - errors["random_selection"]
    )
    image = mpimg.imread(out / "error_curves.png")
    assert image.shape[0] >= 600
    assert image.shape[1] >= 1000
    assert read_json(out / "status.json")["status"] == "completed"
    assert read_json(out / "analysis_provenance.json")["source_sha256"]


def test_approved_matrix_has_exactly_thirty_separate_trials(
    test_output_dir, monkeypatch
):
    calls = []

    def record_trial(config, out):
        calls.append((config, out))
        return out

    monkeypatch.setattr(runner, "run_diagnostic", record_trial)
    root = test_output_dir / "planned"
    execute_suite(root)
    assert len(calls) == 30
    assert len({out for _, out in calls}) == 30
    for config, out in calls:
        assert config.population_size == 16
        assert config.generations == 25
        assert config.seed in range(5)
        assert config.condition in ("selection", "random_selection", "no_mutation")
        assert out.parts[-3] in ("first", "repeat")
    manifest = read_json(root / "suite.json")
    assert manifest["status"] == "trials_completed"
    assert manifest["completed_trials"] == 30
    assert manifest["approved_budget"] is True
    assert manifest["failure"] is None


def test_reanalysis_allows_new_revision_when_core_sources_are_identical(
    saved_suite, test_output_dir, monkeypatch
):
    original = runner._provenance

    def newer_revision(deadline):
        result = original(deadline)
        result["git_revision"] = "f" * 40
        return result

    monkeypatch.setattr(runner, "_provenance", newer_revision)
    summary = compare_suite(saved_suite, test_output_dir / "analysis")
    assert summary["repeat_matches"] is True


def test_suite_stops_on_failure_without_retry_or_replacement(
    test_output_dir, monkeypatch
):
    calls = []

    def failing_trial(config, out):
        calls.append(out)
        if len(calls) == 2:
            raise TimeoutError("fixture timeout")
        return out

    monkeypatch.setattr(runner, "run_diagnostic", failing_trial)
    root = test_output_dir / "failed"
    with pytest.raises(TimeoutError, match="fixture timeout"):
        execute_suite(root)
    assert len(calls) == 2
    manifest = read_json(root / "suite.json")
    assert manifest["status"] == "failed"
    assert manifest["completed_trials"] == 1
    assert "fixture timeout" in manifest["failure"]["message"]


@pytest.mark.parametrize(
    "change", ["missing", "incomplete", "budget", "source", "mutation"]
)
def test_comparison_rejects_missing_incomplete_or_corrupt_artifacts(
    copied_suite, test_output_dir, change
):
    trial = copied_suite / "repeat" / "selection" / "seed_0"
    if change == "missing":
        (trial / "candidates.jsonl").unlink()
    elif change in ("incomplete", "budget"):
        status = read_json(trial / "status.json")
        if change == "incomplete":
            status["status"] = "timeout"
        else:
            status["completed_evaluations"] -= 1
        write_json(trial / "status.json", status)
    elif change == "source":
        provenance = read_json(trial / "provenance.json")
        provenance["sources"][0]["sha256"] = "0" * 64
        write_json(trial / "provenance.json", provenance)
    else:
        path = trial / "candidates.jsonl"
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        row = next(row for row in rows if row["mutations"])
        row["mutations"][0]["after"] = 999.0
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    out = test_output_dir / "analysis"
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_suite(copied_suite, out)
    status = read_json(out / "status.json")
    assert status["status"] == "failed"
    assert status["failure"]["message"]
    assert not (out / "summary.json").exists()


def test_repeat_score_difference_is_not_hidden_by_aggregate_metrics(
    copied_suite, test_output_dir
):
    trial = copied_suite / "repeat" / "selection" / "seed_0"
    path = trial / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["error"] += 0.125
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    metrics = []
    for generation in (0, 1):
        errors = [row["error"] for row in rows if row["generation"] == generation]
        metrics.append(
            (generation, (generation + 1) * 2, min(errors), np.median(errors))
        )
    runner._write_metrics(trial, metrics)
    with pytest.raises(ValueError, match="repeat mismatch"):
        compare_suite(copied_suite, test_output_dir / "analysis")


def test_tampered_parent_identity_is_not_ignored(copied_suite, test_output_dir):
    path = copied_suite / "first" / "selection" / "seed_0" / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["parent_id"] = "g0:i999"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent"):
        compare_suite(copied_suite, test_output_dir / "analysis")


def test_existing_suite_or_analysis_is_never_overwritten(saved_suite, test_output_dir):
    with pytest.raises(FileExistsError):
        execute_suite(saved_suite)
    out = test_output_dir / "analysis"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        compare_suite(saved_suite, out)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1,), (True,)])
def test_invalid_seed_plan_is_rejected_before_creating_outputs(test_output_dir, seeds):
    out = test_output_dir / "invalid"
    with pytest.raises(ValueError, match="seed"):
        execute_suite(out, seeds=seeds)
    assert not out.exists()
