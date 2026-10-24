import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from experiments.cppn_diagnostic import runner
from experiments.cppn_diagnostic.runner import (
    DiagnosticConfig,
    evaluate_genome,
    run_diagnostic,
)
from simulation.cppn import CPPNGenome, MutationConfig, NodeGene


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_candidates(path):
    return [
        json.loads(line)
        for line in (path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def small_config(**overrides):
    return replace(
        DiagnosticConfig(),
        population_size=4,
        generations=2,
        selected_parents=2,
        **overrides,
    )


def rng_from_words(words):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def test_approved_defaults_are_explicit_and_bounded():
    config = DiagnosticConfig()
    assert config.population_size == 16
    assert config.generations == 25
    assert config.selected_parents == 4
    assert config.timeout_seconds == 60.0
    assert config.mutation == MutationConfig(0.2, 0.2, 0.05, 0.1, 0.1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_size", 0),
        ("population_size", True),
        ("population_size", 2.5),
        ("generations", -1),
        ("generations", False),
        ("generations", "2"),
        ("selected_parents", 0),
        ("selected_parents", 17),
        ("seed", -1),
        ("seed", True),
        ("seed", 0.5),
        ("timeout_seconds", 0.0),
        ("timeout_seconds", -1.0),
        ("timeout_seconds", np.inf),
        ("timeout_seconds", np.nan),
        ("timeout_seconds", True),
        ("condition", "unknown"),
        ("condition", np.array(["selection"])),
        ("mutation", None),
    ],
)
def test_invalid_config_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        replace(DiagnosticConfig(), **{field: value})


@pytest.mark.parametrize(("bias", "expected"), [(0.0, 0.25), (1.0, 1.25)])
def test_product_mse_uses_all_25_fixed_input_pairs(bias, expected):
    genome = CPPNGenome(2, 1, (NodeGene(2, "identity", bias),), (), (2,))
    assert evaluate_genome(genome) == expected


def test_evaluator_rejects_wrong_io_and_nonfinite_squared_error():
    wrong_io = CPPNGenome(1, 1, (NodeGene(1, "identity"),), (), (1,))
    with pytest.raises(ValueError, match="2 inputs and 1 output"):
        evaluate_genome(wrong_io)
    overflowing = CPPNGenome(2, 1, (NodeGene(2, "identity", 1e308),), (), (2,))
    with pytest.raises(FloatingPointError):
        evaluate_genome(overflowing)


@pytest.mark.parametrize("condition", ["selection", "random_selection", "no_mutation"])
def test_small_run_preserves_lineage_seeds_genotypes_and_exact_budget(
    test_output_dir, condition
):
    config = small_config(condition=condition)
    out = test_output_dir / condition
    assert run_diagnostic(config, out) == out
    candidates = read_candidates(out)
    assert len(candidates) == 12
    assert len({row["candidate_id"] for row in candidates}) == 12
    status = read_json(out / "status.json")
    assert status["status"] == "completed"
    assert status["completed_evaluations"] == 12
    assert status["completed_input_evaluations"] == 300
    assert status["last_completed_generation"] == 2
    assert status["failure"] is None

    settings = read_json(out / "config.json")
    assert settings["planned_evaluations"] == 12
    assert settings["planned_input_evaluations"] == 300
    assert len(settings["inputs"]) == 25
    assert settings["targets"] == [x * y for x, y in settings["inputs"]]
    effective = MutationConfig(**settings["effective_mutation"])
    by_id = {row["candidate_id"]: row for row in candidates}
    for row in candidates:
        genome = CPPNGenome.from_dict(row["genotype"])
        assert genome.input_size == 2
        assert genome.output_ids == (6,)
        assert [node.id for node in genome.nodes] == list(range(2, 7))
        assert len(genome.connections) == 12
        assert evaluate_genome(genome) == row["error"]
        if row["generation"] == 0:
            assert row["parent_id"] is None
            assert row["mutations"] == []
            assert row["rng_seeds"]["initialization"] == [
                config.seed,
                0,
                row["slot"],
                0,
            ]
            assert row["rng_seeds"]["parent_selection"] is None
            assert row["rng_seeds"]["mutation"] is None
            assert [node.activation for node in genome.nodes] == ["tanh"] * 4 + [
                "identity"
            ]
            assert all(-1.0 <= node.bias < 1.0 for node in genome.nodes)
            assert all(-1.0 <= edge.weight < 1.0 for edge in genome.connections)
            initial_rng = rng_from_words(row["rng_seeds"]["initialization"])
            np.testing.assert_array_equal(
                [node.bias for node in genome.nodes],
                initial_rng.uniform(-1.0, 1.0, size=5),
            )
            np.testing.assert_array_equal(
                [edge.weight for edge in genome.connections],
                initial_rng.uniform(-1.0, 1.0, size=12),
            )
            continue

        previous = [
            candidate
            for candidate in candidates
            if candidate["generation"] == row["generation"] - 1
        ]
        if condition != "random_selection":
            previous = sorted(previous, key=lambda item: (item["error"], item["slot"]))[
                : config.selected_parents
            ]
        selection_words = [config.seed, row["generation"], row["slot"], 1]
        mutation_words = [config.seed, row["generation"], row["slot"], 2]
        assert row["rng_seeds"]["initialization"] is None
        assert row["rng_seeds"]["parent_selection"] == selection_words
        assert row["rng_seeds"]["mutation"] == mutation_words
        selection_rng = rng_from_words(selection_words)
        expected_parent = previous[int(selection_rng.integers(len(previous)))]
        assert row["parent_id"] == expected_parent["candidate_id"]
        parent = CPPNGenome.from_dict(by_id[row["parent_id"]]["genotype"])
        expected_child, changes = parent.mutate(
            effective, rng_from_words(mutation_words)
        )
        assert expected_child == genome
        assert len(changes) == len(row["mutations"])
        for change, saved in zip(changes, row["mutations"]):
            assert saved["parameter"] == change.parameter
            assert saved["before"] == change.before
            assert saved["after"] == change.after
            assert saved["gene_id"] == (
                list(change.gene_id)
                if isinstance(change.gene_id, tuple)
                else change.gene_id
            )
        if condition == "no_mutation":
            assert row["genotype"] == expected_parent["genotype"]
            assert row["mutations"] == []

    with np.load(out / "metrics.npz", allow_pickle=False) as metrics:
        np.testing.assert_array_equal(metrics["generation"], [0, 1, 2])
        np.testing.assert_array_equal(metrics["evaluations"], [4, 8, 12])
        for generation in range(3):
            errors = [
                row["error"] for row in candidates if row["generation"] == generation
            ]
            assert metrics["min_error"][generation] == min(errors)
            assert metrics["median_error"][generation] == np.median(errors)
    with (out / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [int(row["generation"]) for row in rows] == [0, 1, 2]


def test_conditions_start_from_identical_independent_initial_genotypes(test_output_dir):
    initial = []
    for condition in ("selection", "random_selection", "no_mutation"):
        out = test_output_dir / condition
        run_diagnostic(replace(small_config(condition=condition), generations=0), out)
        initial.append([row["genotype"] for row in read_candidates(out)])
    assert initial[0] == initial[1] == initial[2]


def test_equal_errors_use_numeric_slot_order_for_parent_pool(
    test_output_dir, monkeypatch
):
    monkeypatch.setattr(runner, "evaluate_genome", lambda genome: 0.25)
    out = run_diagnostic(small_config(), test_output_dir / "ties")
    candidates = read_candidates(out)
    for row in candidates:
        if row["generation"] > 0:
            index = int(
                rng_from_words(row["rng_seeds"]["parent_selection"]).integers(2)
            )
            assert row["parent_id"] == f"g{row['generation'] - 1}:i{index}"


def test_repeated_runs_reproduce_semantic_artifacts(test_output_dir):
    config = small_config(seed=3)
    first = run_diagnostic(config, test_output_dir / "first")
    second = run_diagnostic(config, test_output_dir / "second")
    assert read_candidates(first) == read_candidates(second)
    with np.load(first / "metrics.npz") as a, np.load(second / "metrics.npz") as b:
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    assert read_json(first / "config.json") == read_json(second / "config.json")


def test_provenance_records_runtime_revision_and_actual_source_contents(
    test_output_dir,
):
    out = run_diagnostic(
        replace(small_config(), generations=0), test_output_dir / "run"
    )
    provenance = read_json(out / "provenance.json")
    assert len(provenance["git_revision"]) == 40
    assert provenance["python_version"]
    assert provenance["numpy_version"] == np.__version__
    assert provenance["platform"]
    assert "git_status" in provenance
    assert "git_diff" in provenance
    assert any(source["path"].endswith("runner.py") for source in provenance["sources"])
    for source in provenance["sources"]:
        assert hashlib.sha256(source["content"].encode("utf-8")).hexdigest() == (
            source["sha256"]
        )


def test_existing_output_directory_is_never_overwritten(test_output_dir):
    out = test_output_dir / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_diagnostic(small_config(), out)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(out.iterdir()) == [sentinel]


def test_timeout_is_operational_failure_not_a_fitness(test_output_dir, monkeypatch):
    clock = [0.0]
    original = runner.evaluate_genome

    def slow_evaluation(genome):
        error = original(genome)
        clock[0] = 2.0
        return error

    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "evaluate_genome", slow_evaluation)
    monkeypatch.setattr(runner, "_provenance", lambda deadline: {"test_clock": True})
    out = test_output_dir / "timeout"
    with pytest.raises(TimeoutError):
        run_diagnostic(small_config(timeout_seconds=1.0), out)
    status = read_json(out / "status.json")
    assert status["status"] == "timeout"
    assert status["failure"]["kind"] == "timeout"
    assert status["completed_evaluations"] == 1
    assert status["last_completed_generation"] is None
    assert len(read_candidates(out)) == 1


def test_numerical_failure_preserves_completed_candidates_and_context(
    test_output_dir, monkeypatch
):
    original = runner.evaluate_genome
    count = [0]

    def failing_evaluation(genome):
        count[0] += 1
        if count[0] == 2:
            raise FloatingPointError("deliberate divergence")
        return original(genome)

    monkeypatch.setattr(runner, "evaluate_genome", failing_evaluation)
    out = test_output_dir / "failed"
    with pytest.raises(FloatingPointError, match="deliberate divergence"):
        run_diagnostic(small_config(), out)
    status = read_json(out / "status.json")
    assert status["status"] == "failed"
    assert status["failure"]["kind"] == "numerical_failure"
    assert "deliberate divergence" in status["failure"]["message"]
    assert status["active_candidate"]["generation"] == 0
    assert status["active_candidate"]["slot"] == 1
    assert status["completed_evaluations"] == 1
    assert len(read_candidates(out)) == 1


def test_genotype_failure_is_not_replaced_by_a_valid_candidate(
    test_output_dir, monkeypatch
):
    def invalid_initialization(rng):
        raise ValueError("invalid genotype fixture")

    monkeypatch.setattr(runner, "_initial_genome", invalid_initialization)
    out = test_output_dir / "invalid"
    with pytest.raises(ValueError, match="invalid genotype fixture"):
        run_diagnostic(small_config(), out)
    status = read_json(out / "status.json")
    assert status["status"] == "failed"
    assert status["failure"]["kind"] == "invalid_input"
    assert status["completed_evaluations"] == 0
    assert read_candidates(out) == []


def test_provenance_failure_is_recorded_without_starting_evaluations(
    test_output_dir, monkeypatch
):
    def fail_provenance(deadline):
        raise subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(runner, "_provenance", fail_provenance)
    out = test_output_dir / "provenance_failure"
    with pytest.raises(subprocess.CalledProcessError):
        run_diagnostic(small_config(), out)
    status = read_json(out / "status.json")
    assert status["status"] == "failed"
    assert status["failure"]["kind"] == "provenance_error"
    assert status["completed_evaluations"] == 0


def test_metrics_write_failure_is_recorded_and_propagated(test_output_dir, monkeypatch):
    def fail_metrics(*args, **kwargs):
        raise OSError("deliberate artifact failure")

    monkeypatch.setattr(runner, "write_metrics_csv_npz", fail_metrics)
    out = test_output_dir / "io_failure"
    with pytest.raises(OSError, match="deliberate artifact failure"):
        run_diagnostic(small_config(), out)
    status = read_json(out / "status.json")
    assert status["status"] == "failed"
    assert status["failure"]["kind"] == "artifact_error"
    assert status["completed_evaluations"] == 4
    assert status["last_completed_generation"] is None


def test_cli_runs_only_the_requested_small_seed_and_condition(test_output_dir):
    out = test_output_dir / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.cppn_diagnostic.runner",
            "--out-dir",
            str(out),
            "--seed",
            "2",
            "--condition",
            "random_selection",
            "--population-size",
            "2",
            "--generations",
            "0",
            "--selected-parents",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(read_candidates(out)) == 2
    settings = read_json(out / "config.json")
    assert settings["config"]["seed"] == 2
    assert settings["config"]["condition"] == "random_selection"
