"""Execute or compare the fixed P1-05 diagnostic plan, without extra searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from experiments.cppn_diagnostic import runner
from simulation.cppn import CPPNGenome, MutationConfig

_SEEDS = (0, 1, 2, 3, 4)
_CONDITIONS = ("selection", "random_selection", "no_mutation")
_REPLICATES = ("first", "repeat")
_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    ArithmeticError,
    OSError,
    subprocess.SubprocessError,
    ImportError,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")


def _source_record() -> dict[str, str]:
    content = Path(__file__).read_text(encoding="utf-8")
    return {
        "source_content": content,
        "source_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(runner.DiagnosticConfig(seed=value).seed for value in values)
    _require(
        1 <= len(result) <= 5 and len(set(result)) == len(result),
        "seed plan must contain 1 to 5 distinct nonnegative integer seeds",
    )
    return result


def _config(data) -> runner.DiagnosticConfig:
    values = dict(data)
    mutation = MutationConfig(**values.pop("mutation"))
    return runner.DiagnosticConfig(**values, mutation=mutation)


def execute_suite(
    run_root: str | Path,
    *,
    seeds: Sequence[int] = _SEEDS,
    template: runner.DiagnosticConfig | None = None,
) -> Path:
    """Run two independent sets sequentially; stop on the first failed trial."""
    seeds = _seeds(seeds)
    template = runner.DiagnosticConfig() if template is None else template
    if not isinstance(template, runner.DiagnosticConfig):
        raise ValueError("template must be a DiagnosticConfig")
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "template": asdict(template),
        "seeds": seeds,
        "conditions": _CONDITIONS,
        "replicates": _REPLICATES,
        "approved_budget": seeds == _SEEDS and template == runner.DiagnosticConfig(),
        "completed_trials": 0,
        "active_trial": None,
        "failure": None,
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        **_source_record(),
    }
    _write_json(root / "suite.json", manifest)
    completed = 0
    try:
        manifest["provenance"] = runner._provenance(
            time.perf_counter() + template.timeout_seconds
        )
        manifest["matplotlib_version"] = version("matplotlib")
        _write_json(root / "suite.json", manifest)
        for replicate in _REPLICATES:
            for condition in _CONDITIONS:
                for seed in seeds:
                    out = root / replicate / condition / f"seed_{seed}"
                    manifest["active_trial"] = str(out.relative_to(root))
                    _write_json(root / "suite.json", manifest)
                    runner.run_diagnostic(
                        replace(template, seed=seed, condition=condition), out
                    )
                    completed += 1
                    manifest["completed_trials"] = completed
        manifest["active_trial"] = None
        manifest["status"] = "trials_completed"
        _write_json(root / "suite.json", manifest)
    except _ERRORS as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"exception": type(exc).__name__, "message": str(exc)}
        _write_json(root / "suite.json", manifest)
        raise
    return root


@dataclass(frozen=True)
class _Saved:
    candidate_id: str
    slot: int
    genome: CPPNGenome
    error: float
    raw: dict[str, object]


@dataclass(frozen=True)
class _Trial:
    rows: list[_Saved]
    metrics: dict[str, np.ndarray]
    summary: dict[str, str | int | float]


def _provenance_identity(data) -> dict[str, object]:
    for source in data["sources"]:
        digest = hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
        _require(digest == source["sha256"], "source snapshot hash mismatch")
    return {
        name: data[name]
        for name in (
            "git_revision",
            "python_version",
            "numpy_version",
            "platform",
            "sources",
        )
    }


def _load_trial(
    path: Path, config: runner.DiagnosticConfig, provenance: dict[str, object]
) -> _Trial:
    status = _read_json(path / "status.json")
    planned = config.population_size * (config.generations + 1)
    _require(
        status["status"] == "completed"
        and status["failure"] is None
        and status["active_candidate"] is None
        and status["completed_evaluations"] == planned
        and status["completed_input_evaluations"] == planned * 25
        and status["last_completed_generation"] == config.generations,
        f"{path}: incomplete trial or evaluation budget mismatch",
    )
    settings = _read_json(path / "config.json")
    _require(
        settings["schema_version"] == 1
        and settings["definition_version"] == 1
        and settings["experiment"] == "cppn_product_diagnostic"
        and settings["config"] == asdict(config)
        and settings["planned_evaluations"] == planned
        and settings["planned_input_evaluations"] == planned * 25
        and settings["inputs"] == [list(point) for point in runner._INPUTS]
        and settings["targets"] == list(runner._TARGETS)
        and settings["rng"]
        == {
            "algorithm": "PCG64",
            "seed_words": ["root_seed", "generation", "slot", "purpose"],
            "purposes": {"initialization": 0, "parent_selection": 1, "mutation": 2},
        },
        f"{path}: diagnostic configuration mismatch",
    )
    effective = config.mutation
    if config.condition == "no_mutation":
        effective = replace(
            effective,
            weight_probability=0.0,
            bias_probability=0.0,
            activation_probability=0.0,
        )
    _require(
        settings["effective_mutation"] == asdict(effective),
        f"{path}: effective mutation mismatch",
    )
    _require(
        _provenance_identity(_read_json(path / "provenance.json")) == provenance,
        f"{path}: execution provenance mismatch",
    )
    raw_rows = [
        json.loads(line)
        for line in (path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _require(len(raw_rows) == planned, f"{path}: candidate count mismatch")
    saved: list[_Saved] = []
    previous: list[_Saved] = []
    current: list[_Saved] = []
    pool: list[_Saved] = []
    metrics = []
    seen_errors: dict[str, float] = {}
    changed_genotypes = changed_errors = mutation_records = 0
    counts = {"weight": 0, "bias": 0, "activation": 0}
    for index, row in enumerate(raw_rows):
        generation, slot = divmod(index, config.population_size)
        candidate_id = f"g{generation}:i{slot}"
        where = f"{path}: candidate {candidate_id}"
        _require(
            row["candidate_id"] == candidate_id
            and row["generation"] == generation
            and row["slot"] == slot,
            f"{where}: identity/order mismatch",
        )
        value = row["error"]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{where}: invalid error type",
        )
        error = float(value)
        _require(
            math.isfinite(error) and error >= 0.0, f"{where}: non-finite/negative error"
        )
        genome = CPPNGenome.from_dict(row["genotype"])
        _require(genome.to_dict() == row["genotype"], f"{where}: restore mismatch")
        if slot == 0:
            current = []
            pool = (
                previous
                if config.condition == "random_selection"
                else sorted(previous, key=lambda item: (item.error, item.slot))[
                    : config.selected_parents
                ]
            )
        words = [config.seed, generation, slot]
        if generation == 0:
            _require(
                row["parent_id"] is None and row["mutations"] == [],
                f"{where}: initial parent/mutation mismatch",
            )
            expected_seeds = {
                "initialization": [*words, 0],
                "parent_selection": None,
                "mutation": None,
            }
            expected = runner._initial_genome(runner._rng([*words, 0]))
        else:
            expected_seeds = {
                "initialization": None,
                "parent_selection": [*words, 1],
                "mutation": [*words, 2],
            }
            parent = pool[int(runner._rng([*words, 1]).integers(len(pool)))]
            _require(
                row["parent_id"] == parent.candidate_id, f"{where}: parent mismatch"
            )
            expected, changes = parent.genome.mutate(
                effective, runner._rng([*words, 2])
            )
            change_data = json.loads(json.dumps([asdict(change) for change in changes]))
            _require(
                change_data == row["mutations"], f"{where}: mutation record mismatch"
            )
            changed_genotypes += int(genome != parent.genome)
            changed_errors += int(error != parent.error)
            mutation_records += len(changes)
            for change in changes:
                counts[change.parameter] += 1
        _require(
            row["rng_seeds"] == expected_seeds, f"{where}: seed derivation mismatch"
        )
        _require(
            genome == expected, f"{where}: inheritance/initialization replay mismatch"
        )
        signature = json.dumps(genome.to_dict(), sort_keys=True)
        if signature in seen_errors:
            _require(
                seen_errors[signature] == error,
                f"{where}: identical genotype error drift",
            )
        seen_errors[signature] = error
        item = _Saved(candidate_id, slot, genome, error, row)
        saved.append(item)
        current.append(item)
        if slot == config.population_size - 1:
            errors = [candidate.error for candidate in current]
            with np.errstate(over="raise", invalid="raise"):
                median = float(np.median(errors))
            _require(math.isfinite(median), f"{where}: non-finite median")
            metrics.append((generation, index + 1, min(errors), median))
            previous = current

    expected_metrics = {
        name: np.asarray([row[index] for row in metrics])
        for index, name in enumerate(runner._METRIC_NAMES)
    }
    with np.load(path / "metrics.npz", allow_pickle=False) as archive:
        _require(
            set(archive.files) == set(expected_metrics), f"{path}: metric keys mismatch"
        )
        arrays = {name: archive[name].copy() for name in archive.files}
    with (path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            reader.fieldnames == list(runner._METRIC_NAMES),
            f"{path}: CSV columns mismatch",
        )
        csv_rows = list(reader)
    _require(len(csv_rows) == len(metrics), f"{path}: CSV generation count mismatch")
    for name, expected in expected_metrics.items():
        _require(
            np.array_equal(arrays[name], expected),
            f"{path}: NPZ metric {name} mismatch",
        )
        csv_values = np.array([float(row[name]) for row in csv_rows])
        _require(
            np.array_equal(csv_values, expected), f"{path}: CSV metric {name} mismatch"
        )
    return _Trial(
        saved,
        arrays,
        {
            "seed": config.seed,
            "condition": config.condition,
            "initial_min_error": float(arrays["min_error"][0]),
            "initial_median_error": float(arrays["median_error"][0]),
            "final_min_error": float(arrays["min_error"][-1]),
            "final_median_error": float(arrays["median_error"][-1]),
            "median_error_change": float(
                arrays["median_error"][-1] - arrays["median_error"][0]
            ),
            "offspring": config.population_size * config.generations,
            "genotype_changed_offspring": changed_genotypes,
            "error_changed_offspring": changed_errors,
            "mutation_records": mutation_records,
            "weight_changes": counts["weight"],
            "bias_changes": counts["bias"],
            "activation_changes": counts["activation"],
            "unique_genotypes": len(seen_errors),
        },
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    path: Path, seeds: tuple[int, ...], trials: dict[tuple[str, int], _Trial]
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, axis in enumerate(axes.flat):
        if index < 5:
            if index >= len(seeds):
                axis.set_visible(False)
                continue
            seed = seeds[index]
            for condition, color in zip(_CONDITIONS, colors):
                values = trials[(condition, seed)].metrics
                axis.plot(
                    values["generation"],
                    values["median_error"],
                    label=condition,
                    color=color,
                )
            axis.set(
                title=f"Seed {seed}: population median",
                xlabel="Generation",
                ylabel="MSE",
            )
            axis.legend(fontsize=8)
        else:
            for seed in seeds:
                errors = [
                    trials[(condition, seed)].metrics["median_error"][-1]
                    for condition in _CONDITIONS
                ]
                axis.plot(range(3), errors, marker="o", label=f"seed {seed}")
            axis.set_xticks(range(3), ["selection", "random", "no mutation"])
            axis.set(title="Final medians, paired by seed", ylabel="MSE")
            axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.suptitle("CPPN product diagnostic: primary runs (independent axes)")
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=120)
    finally:
        plt.close(figure)


def compare_suite(run_root: str | Path, out_dir: str | Path) -> dict[str, object]:
    """Validate full saved evidence and plot it without new fitness evaluations."""
    root, out = Path(run_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    status: dict[str, object] = {"status": "running", "failure": None}
    _write_json(out / "status.json", status)
    try:
        manifest = _read_json(root / "suite.json")
        seeds = _seeds(manifest["seeds"])
        template = _config(manifest["template"])
        trial_count = len(seeds) * 6
        _require(
            manifest["schema_version"] == 1
            and manifest["status"] == "trials_completed"
            and manifest["completed_trials"] == trial_count
            and manifest["failure"] is None
            and manifest["conditions"] == list(_CONDITIONS)
            and manifest["replicates"] == list(_REPLICATES),
            "suite incomplete or plan mismatch",
        )
        provenance = _provenance_identity(manifest["provenance"])
        _require(
            provenance["python_version"] == sys.version
            and provenance["numpy_version"] == np.__version__
            and provenance["platform"] == platform.platform(),
            "replay requires the recorded Python/NumPy/platform",
        )
        current = _provenance_identity(
            runner._provenance(time.perf_counter() + template.timeout_seconds)
        )
        _require(
            all(
                current[name] == provenance[name]
                for name in current
                if name != "git_revision"
            ),
            "replay requires the recorded core sources and runtime",
        )
        _write_json(
            out / "analysis_provenance.json",
            {
                **_source_record(),
                "command": [sys.executable, *sys.argv],
                "working_directory": str(Path.cwd()),
                "provenance": current,
                "matplotlib_version": version("matplotlib"),
            },
        )
        primary: dict[tuple[str, int], _Trial] = {}
        restored = replayed = 0
        per_trial = []
        paired = []
        for seed in seeds:
            initial = None
            for condition in _CONDITIONS:
                config = replace(template, seed=seed, condition=condition)
                first = _load_trial(
                    root / "first" / condition / f"seed_{seed}", config, provenance
                )
                repeat = _load_trial(
                    root / "repeat" / condition / f"seed_{seed}", config, provenance
                )
                for a, b in zip(first.rows, repeat.rows):
                    _require(
                        a.raw == b.raw,
                        f"repeat mismatch: {condition}, seed {seed}, candidate {a.candidate_id}",
                    )
                initial_genotypes = [
                    row.genome.to_dict() for row in first.rows[: config.population_size]
                ]
                if initial is None:
                    initial = initial_genotypes
                _require(
                    initial == initial_genotypes,
                    f"initial population mismatch: seed {seed}",
                )
                primary[(condition, seed)] = first
                per_trial.append(first.summary)
                restored += len(first.rows) + len(repeat.rows)
                replayed += 2 * config.population_size * config.generations
            errors = {
                condition: float(primary[(condition, seed)].metrics["median_error"][-1])
                for condition in _CONDITIONS
            }
            paired.append(
                {
                    "seed": seed,
                    **errors,
                    "selection_minus_random": errors["selection"]
                    - errors["random_selection"],
                    "selection_minus_no_mutation": errors["selection"]
                    - errors["no_mutation"],
                }
            )
        planned = template.population_size * (template.generations + 1)
        approved = seeds == _SEEDS and template == runner.DiagnosticConfig()
        _require(
            manifest["approved_budget"] == approved, "approved budget label mismatch"
        )
        summary: dict[str, object] = {
            "status": "completed",
            "approved_budget": approved,
            "trials": trial_count,
            "repeat_pairs": len(seeds) * 3,
            "individual_evaluations": trial_count * planned,
            "input_evaluations": trial_count * planned * 25,
            "restored_genotypes": restored,
            "replayed_offspring": replayed,
            "repeat_matches": True,
            "initial_populations_match": True,
            "selection_better_than_random": sum(
                row["selection_minus_random"] < 0 for row in paired
            ),
            "selection_improved_from_initial": sum(
                float(primary[("selection", seed)].summary["median_error_change"]) < 0
                for seed in seeds
            ),
            "per_trial": per_trial,
            "paired": paired,
        }
        _write_csv(out / "summary.csv", per_trial)
        _write_csv(out / "paired.csv", paired)
        _plot(out / "error_curves.png", seeds, primary)
        _write_json(out / "summary.json", summary)
        status["status"] = "completed"
        _write_json(out / "status.json", status)
    except _ERRORS as exc:
        status["status"] = "failed"
        status["failure"] = {"exception": type(exc).__name__, "message": str(exc)}
        _write_json(out / "status.json", status)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the approved 30 trials before comparison.",
    )
    args = parser.parse_args(argv)
    if args.execute:
        execute_suite(args.run_root)
    out = args.out_dir if args.out_dir is not None else args.run_root / "analysis"
    summary = compare_suite(args.run_root, out)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("per_trial", "paired")
            }
        )
    )


if __name__ == "__main__":
    main()
