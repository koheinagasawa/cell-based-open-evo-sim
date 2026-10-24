"""Run one seed and condition of the approved product approximation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.common.metrics import write_metrics_csv_npz
from simulation.cppn import ConnectionGene, CPPNGenome, MutationConfig, NodeGene

Condition = Literal["selection", "random_selection", "no_mutation"]
_CONDITIONS = ("selection", "random_selection", "no_mutation")
_INPUTS = tuple(product((-1.0, -0.5, 0.0, 0.5, 1.0), repeat=2))
_TARGETS = tuple(x * y for x, y in _INPUTS)
_METRIC_NAMES = ("generation", "evaluations", "min_error", "median_error")


@dataclass(frozen=True)
class DiagnosticConfig:
    seed: int = 0
    condition: Condition = "selection"
    population_size: int = 16
    generations: int = 25
    selected_parents: int = 4
    timeout_seconds: float = 60.0
    mutation: MutationConfig = field(
        default_factory=lambda: MutationConfig(0.2, 0.2, 0.05, 0.1, 0.1)
    )

    def __post_init__(self) -> None:
        for name in ("seed", "population_size", "generations", "selected_parents"):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "generations") else 1
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        if self.selected_parents > self.population_size:
            raise ValueError("selected_parents must not exceed population_size")
        if not isinstance(self.condition, str) or self.condition not in _CONDITIONS:
            raise ValueError(f"condition must be one of {_CONDITIONS}")
        if not isinstance(self.mutation, MutationConfig):
            raise ValueError("mutation must be a MutationConfig")
        if not isinstance(self.timeout_seconds, Real) or isinstance(
            self.timeout_seconds, bool
        ):
            raise ValueError("timeout_seconds must be a positive finite real number")
        try:
            timeout = float(self.timeout_seconds)
        except OverflowError as exc:
            raise ValueError("timeout_seconds must be finite") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    slot: int
    genome: CPPNGenome
    error: float


def _rng(words: Sequence[int]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def _initial_genome(rng: np.random.Generator) -> CPPNGenome:
    nodes = [
        NodeGene(
            node_id,
            "tanh" if node_id < 6 else "identity",
            float(rng.uniform(-1.0, 1.0)),
        )
        for node_id in range(2, 7)
    ]
    endpoints = [(source, target) for source in range(2) for target in range(2, 6)] + [
        (source, 6) for source in range(2, 6)
    ]
    connections = [
        ConnectionGene(source, target, float(rng.uniform(-1.0, 1.0)))
        for source, target in endpoints
    ]
    return CPPNGenome(2, 1, nodes, connections, (6,))


def evaluate_genome(genome: CPPNGenome) -> float:
    """Evaluate all 25 fixed points without State, interpretation, or randomness."""
    if genome.input_size != 2 or genome.output_size != 1:
        raise ValueError("product diagnostic requires 2 inputs and 1 output")
    predictions = np.array([genome.activate(inputs)[0] for inputs in _INPUTS])
    with np.errstate(over="raise", invalid="raise"):
        error = float(np.mean(np.square(predictions - np.asarray(_TARGETS))))
    if not math.isfinite(error):
        raise FloatingPointError("non-finite product diagnostic error")
    return error


def _remaining(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0.0:
        raise TimeoutError("diagnostic exceeded its operational timeout")
    return remaining


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")


def _provenance(deadline: float) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path("experiments") / "cppn_diagnostic" / "runner.py",
        Path("simulation") / "cppn.py",
        Path("experiments") / "common" / "metrics.py",
    )

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "--no-pager", *args],
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=_remaining(deadline),
        ).stdout

    sources = []
    for relative in source_paths:
        content = (root / relative).read_text(encoding="utf-8")
        sources.append(
            {
                "path": str(relative),
                "content": content,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "git_revision": git("rev-parse", "HEAD").strip(),
        "git_status": git("status", "--short"),
        "git_diff": git("diff", "HEAD", "--", *(str(path) for path in source_paths)),
        "sources": sources,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "argv": sys.argv,
    }


def _write_metrics(out_dir: Path, rows: list[tuple[int, int, float, float]]) -> None:
    arrays = {
        name: np.asarray(
            [row[index] for row in rows], dtype=np.int64 if index < 2 else np.float64
        )
        for index, name in enumerate(_METRIC_NAMES)
    }
    write_metrics_csv_npz(out_dir, arrays, header=_METRIC_NAMES)


def _failure_kind(error: Exception) -> str:
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(error, ArithmeticError):
        return "numerical_failure"
    if isinstance(error, OSError):
        return "artifact_error"
    if isinstance(error, subprocess.SubprocessError):
        return "provenance_error"
    return "invalid_input"


def run_diagnostic(config: DiagnosticConfig, out_dir: str | Path) -> Path:
    """Run one bounded trial; preserve partial evidence and re-raise failures.

    Existing output directories are never reused. Timeout checks are cooperative,
    between bounded candidate evaluations and artifact operations, not fitness.
    This is external replacement of genotypes, not Agent reproduction.
    """
    if not isinstance(config, DiagnosticConfig):
        raise ValueError("config must be a DiagnosticConfig")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + config.timeout_seconds
    status: dict[str, object] = {
        "status": "running",
        "completed_evaluations": 0,
        "completed_input_evaluations": 0,
        "last_completed_generation": None,
        "active_candidate": None,
        "failure": None,
        "elapsed_seconds": 0.0,
    }
    _write_json(out / "status.json", status)
    completed = 0
    metrics: list[tuple[int, int, float, float]] = []
    effective_mutation = config.mutation
    if config.condition == "no_mutation":
        effective_mutation = replace(
            config.mutation,
            weight_probability=0.0,
            bias_probability=0.0,
            activation_probability=0.0,
        )
    try:
        planned = config.population_size * (config.generations + 1)
        _write_json(
            out / "config.json",
            {
                "schema_version": 1,
                "experiment": "cppn_product_diagnostic",
                "definition_version": 1,
                "config": asdict(config),
                "effective_mutation": asdict(effective_mutation),
                "inputs": _INPUTS,
                "targets": _TARGETS,
                "rng": {
                    "algorithm": "PCG64",
                    "seed_words": ["root_seed", "generation", "slot", "purpose"],
                    "purposes": {
                        "initialization": 0,
                        "parent_selection": 1,
                        "mutation": 2,
                    },
                },
                "planned_evaluations": planned,
                "planned_input_evaluations": planned * len(_INPUTS),
            },
        )
        _write_json(out / "provenance.json", _provenance(deadline))
        previous: list[_Candidate] = []
        with (out / "candidates.jsonl").open("x", encoding="utf-8") as stream:
            for generation in range(config.generations + 1):
                if config.condition == "random_selection":
                    pool = previous
                else:
                    pool = sorted(previous, key=lambda item: (item.error, item.slot))[
                        : config.selected_parents
                    ]
                current: list[_Candidate] = []
                for slot in range(config.population_size):
                    context: dict[str, object] = {
                        "candidate_id": f"g{generation}:i{slot}",
                        "generation": generation,
                        "slot": slot,
                        "parent_id": None,
                    }
                    status["active_candidate"] = context
                    _remaining(deadline)
                    words = [config.seed, generation, slot]
                    seeds: dict[str, list[int] | None] = {
                        "initialization": None,
                        "parent_selection": None,
                        "mutation": None,
                    }
                    if generation == 0:
                        seeds["initialization"] = [*words, 0]
                        genome = _initial_genome(_rng([*words, 0]))
                        changes = ()
                    else:
                        seeds["parent_selection"] = [*words, 1]
                        seeds["mutation"] = [*words, 2]
                        parent = pool[int(_rng([*words, 1]).integers(len(pool)))]
                        context["parent_id"] = parent.candidate_id
                        genome, changes = parent.genome.mutate(
                            effective_mutation, _rng([*words, 2])
                        )
                    error = evaluate_genome(genome)
                    record = {
                        **context,
                        "rng_seeds": seeds,
                        "genotype": genome.to_dict(),
                        "mutations": [asdict(change) for change in changes],
                        "error": error,
                    }
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    current.append(
                        _Candidate(f"g{generation}:i{slot}", slot, genome, error)
                    )
                    completed += 1
                    status["completed_evaluations"] = completed
                    status["completed_input_evaluations"] = completed * len(_INPUTS)
                    _remaining(deadline)
                with np.errstate(over="raise", invalid="raise"):
                    median = float(
                        np.median([candidate.error for candidate in current])
                    )
                if not math.isfinite(median):
                    raise FloatingPointError("non-finite generation median")
                metrics.append(
                    (
                        generation,
                        completed,
                        min(candidate.error for candidate in current),
                        median,
                    )
                )
                _write_metrics(out, metrics)
                status["last_completed_generation"] = generation
                status["active_candidate"] = None
                _remaining(deadline)
                previous = current
        status["status"] = "completed"
        status["elapsed_seconds"] = time.perf_counter() - started
        _write_json(out / "status.json", status)
    except (
        ValueError,
        TypeError,
        ArithmeticError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        kind = _failure_kind(exc)
        status["status"] = "timeout" if kind == "timeout" else "failed"
        status["failure"] = {
            "kind": kind,
            "exception": type(exc).__name__,
            "message": str(exc),
        }
        status["elapsed_seconds"] = time.perf_counter() - started
        _write_json(out / "status.json", status)
        raise
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DiagnosticConfig()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=_CONDITIONS, required=True)
    parser.add_argument("--population-size", type=int, default=defaults.population_size)
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument(
        "--selected-parents", type=int, default=defaults.selected_parents
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=defaults.timeout_seconds
    )
    args = parser.parse_args(argv)
    config = DiagnosticConfig(
        seed=args.seed,
        condition=args.condition,
        population_size=args.population_size,
        generations=args.generations,
        selected_parents=args.selected_parents,
        timeout_seconds=args.timeout_seconds,
    )
    print(run_diagnostic(config, args.out_dir))


if __name__ == "__main__":
    main()
