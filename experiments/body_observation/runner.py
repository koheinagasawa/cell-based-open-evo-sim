"""Observe five random CPPNs, drive-off controls, replay and half-step runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np

from experiments.common.metrics import write_metrics_csv_npz
from simulation.cppn import ConnectionGene, CPPNGenome, NodeGene
from simulation.minimal_organism import MinimalOrganismWorld
from simulation.physics.inertial import PhysicsDiagnostics, PhysicsFailure

SEEDS = tuple(range(5))
RUNS = (("normal", 0.01), ("replay", 0.01), ("half_step", 0.005))
_METRICS = (
    "time",
    "cells",
    "growth_budget",
    "max_speed",
    "max_overlap",
    "max_bond_length",
    "centroid_x",
    "centroid_y",
)


@dataclass(frozen=True)
class ObservationConfig:
    control_steps: int = 1000
    timeout_seconds: float = 300.0

    def __post_init__(self):
        if (
            not isinstance(self.control_steps, Integral)
            or isinstance(self.control_steps, bool)
            or not 1 <= self.control_steps <= 1000
        ):
            raise ValueError("control_steps must be an integer from 1 to 1000")
        if (
            not isinstance(self.timeout_seconds, Real)
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    seed: int
    drive_off: bool
    physics_dt: float
    status: Literal["completed", "failed"]
    reason: str | None
    completed_control_intervals: int
    births: int
    final_cells: int
    max_cells: int
    growth_budget: float | None
    physics_diagnostics: PhysicsDiagnostics | None
    out_dir: Path
    elapsed_seconds: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["out_dir"] = str(self.out_dir)
        return result


def initial_genome(seed: int) -> CPPNGenome:
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
    activations = ["tanh"] * 4 + ["sin"] * 2 + ["gaussian"] * 2 + ["identity"] * 8
    nodes = tuple(
        NodeGene(18 + i, activation, float(rng.uniform(-0.5, 0.5)))
        for i, activation in enumerate(activations)
    )
    connections = tuple(
        ConnectionGene(source, target, float(rng.uniform(-1, 1) / np.sqrt(n)))
        for targets, sources, n in (
            (range(18, 26), range(18), 18),
            (range(26, 34), range(18, 26), 8),
        )
        for target in targets
        for source in sources
    )
    return CPPNGenome(18, 8, nodes, connections, tuple(range(26, 34)))


def json_record(value) -> str:
    def safe(item):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return {
                "nonfinite": (
                    "NaN"
                    if math.isnan(item)
                    else "Infinity" if item > 0 else "-Infinity"
                )
            }
        if isinstance(item, dict):
            return {key: safe(content) for key, content in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(content) for content in item]
        return item

    return json.dumps(safe(value), ensure_ascii=False, allow_nan=False)


def _write_json(path: Path, value) -> None:
    path.write_text(json_record(value) + "\n", encoding="utf-8")


def _provenance(
    out: Path,
    config: ObservationConfig,
    *,
    startup: bool = False,
    candidate: CandidateResult | None = None,
):
    root = Path(__file__).resolve().parents[2]
    relative_paths = [
        Path(__file__).resolve().relative_to(root),
        *(
            Path("simulation") / name
            for name in (
                "minimal_organism.py",
                "world.py",
                "cell.py",
                "agent.py",
                "cppn.py",
                "interpreter.py",
                "policies.py",
            )
        ),
        Path("simulation") / "physics" / "inertial.py",
        Path("simulation") / "physics" / "repulsion.py",
        Path("experiments") / "common" / "metrics.py",
        Path("doc") / "MinimalOrganismContract_ja.md",
    ]
    sources = []
    for relative in relative_paths:
        content = (root / relative).read_text(encoding="utf-8")
        sources.append(
            {
                "path": str(relative),
                "content": content,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )

    def git(*args):
        return subprocess.run(
            ["git", "--no-pager", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        ).stdout

    _write_json(
        out / "metadata.json",
        {
            "schema_version": 1,
            "mode": (
                "candidate" if candidate else "startup" if startup else "observation"
            ),
            "candidate_id": candidate.candidate_id if candidate else None,
            "config": asdict(config),
            "seeds": (
                [candidate.seed]
                if candidate
                else tuple(range(100)) if startup else SEEDS
            ),
            "runs": (
                [("candidate", candidate.physics_dt)]
                if candidate
                else [("until_first_birth", 0.01)] if startup else RUNS
            ),
            "conditions": (
                ["drive_off" if candidate.drive_off else "active"]
                if candidate
                else ["active"] if startup else ["active", "drive_off"]
            ),
            "purpose": (
                "single Agent redevelopment; no evaluation or selection"
                if candidate
                else (
                    "unselected birth-startup diagnosis"
                    if startup
                    else "unselected initial body observations; not evolution"
                )
            ),
            "total_timeout_seconds": config.timeout_seconds if startup else None,
            "revision": git("rev-parse", "HEAD").strip(),
            "workspace_status": git("status", "--porcelain"),
            "workspace_diff": git("diff", "HEAD", "--"),
            "source_snapshot": sources,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
            },
            "command": sys.argv,
            "initialization": (
                {"method": "restore supplied genotype; no random initialization"}
                if candidate
                else {
                    "rng": "PCG64(SeedSequence(seed))",
                    "bias_order": "node IDs 18..33",
                    "connection_order": "target-major: 0..17 -> 18..25; 18..25 -> 26..33",
                    "draw_order": "all biases, then all connection weights",
                }
            ),
            "nonfinite_encoding": "tagged objects in failure evidence, not fallback values",
        },
    )


def _run_trial(
    genotype: dict,
    seed: int,
    drive_off: bool,
    dt: float,
    out: Path,
    config: ObservationConfig,
    *,
    stop_on_first_birth: bool = False,
    overall_deadline: float | None = None,
    candidate_id: str | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config.timeout_seconds
    if overall_deadline is not None:
        deadline = min(deadline, overall_deadline)
    world = None
    status = {
        "status": "running",
        "seed": seed,
        "drive_off": drive_off,
        "physics_dt": dt,
        "completed_control_intervals": 0,
        "reason": None,
        "candidate_id": candidate_id,
        "ended_by": None,
        "initial_gate": None,
        "min_gate": None,
        "max_gate": None,
        "gate_evaluations": 0,
        "gate_requests": 0,
    }
    _write_json(out / "status.json", status)
    metrics = {name: [] for name in _METRICS}
    event_cursor = 0

    with (
        (out / "frames.jsonl").open("w", encoding="utf-8") as frames,
        (out / "events.jsonl").open("w", encoding="utf-8") as events,
    ):

        def save_events():
            nonlocal event_cursor
            if world is not None:
                for event in world.events[event_cursor:]:
                    events.write(json_record(event) + "\n")
                event_cursor = len(world.events)
                events.flush()

        def sample():
            frame = world.snapshot()
            frames.write(json_record(frame) + "\n")
            frames.flush()
            for cell in frame["cells"]:
                if cell["raw_output"] is not None:
                    gate = cell["raw_output"][4]
                    if status["initial_gate"] is None:
                        status["initial_gate"] = gate
                    status["min_gate"] = (
                        gate
                        if status["min_gate"] is None
                        else min(status["min_gate"], gate)
                    )
                    status["max_gate"] = (
                        gate
                        if status["max_gate"] is None
                        else max(status["max_gate"], gate)
                    )
                    status["gate_evaluations"] += 1
                    status["gate_requests"] += int(gate > 0.5)
            centroid = np.mean([c["position"] for c in frame["cells"]], axis=0)
            values = {
                "time": frame["simulation_time"],
                "cells": len(frame["cells"]),
                "growth_budget": sum(c["energy"] for c in frame["cells"]),
                **frame["physics_diagnostics"],
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
            for name in _METRICS:
                metrics[name].append(values[name])

        try:
            world = MinimalOrganismWorld(
                CPPNGenome.from_dict(genotype),
                seed=seed,
                physics_dt=dt,
                drive_off=drive_off,
            )
            _write_json(
                out / "config.json",
                {
                    **asdict(config),
                    "seed": seed,
                    "physics_dt": dt,
                    "drive_off": drive_off,
                    "physics": asdict(world.physics_solver.config),
                    "interpreter": world.agent.interpreter.to_metadata(),
                    "genotype": genotype,
                    "candidate_id": candidate_id,
                    "stop_on_first_birth": stop_on_first_birth,
                    "remaining_budget_seconds": max(0.0, deadline - started),
                },
            )
            sample()
            for _ in range(config.control_steps):
                world.step(deadline=deadline)
                sample()
                save_events()
                if stop_on_first_birth and any(
                    e.get("status") == "born" for e in world.events
                ):
                    status["ended_by"] = "first_birth"
                    break
            status["status"] = "completed"
            if status["ended_by"] is None:
                status["ended_by"] = "observation_limit"
        except (FloatingPointError, ValueError, TypeError, TimeoutError) as exc:
            status.update(
                status="failed",
                reason=(
                    exc.reason
                    if isinstance(exc, PhysicsFailure)
                    else (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else (
                            "numerical_failure"
                            if isinstance(exc, FloatingPointError)
                            else "invalid_input"
                        )
                    )
                ),
                message=str(exc),
            )
            _write_json(
                out / "failure.json",
                {
                    "reason": status["reason"],
                    "message": str(exc),
                    "last_completed_state": (
                        world.last_completed_snapshot if world else None
                    ),
                    "failure": world.failure if world else None,
                    "current_state": world.snapshot() if world else None,
                },
            )
        finally:
            save_events()
            status["completed_control_intervals"] = world.time if world else 0
            status["elapsed_seconds"] = time.monotonic() - started
            status["final_cells"] = len(world.cells) if world else 0
            status["births"] = (
                sum(e.get("status") == "born" for e in world.events) if world else 0
            )
            status["birth_rejections"] = (
                dict(
                    Counter(
                        e["reason"]
                        for e in world.events
                        if e.get("status") == "rejected"
                    )
                )
                if world
                else {}
            )
            status["max_cells"] = max(
                max(metrics["cells"], default=0), status["final_cells"]
            )
            status["growth_budget"] = (
                float(sum(c.energy for c in world.cells)) if world else None
            )
            status["physics_diagnostics"] = (
                {
                    name: float(max(metrics[name], default=0))
                    for name in ("max_speed", "max_overlap", "max_bond_length")
                }
                if metrics["time"]
                else None
            )
            _write_json(out / "status.json", status)
            write_metrics_csv_npz(
                out,
                {key: np.array(value) for key, value in metrics.items()},
                header=_METRICS,
            )
    return status


def run_candidate(
    candidate_id: str,
    genotype: CPPNGenome,
    *,
    seed: int,
    out_dir: str | Path,
    config: ObservationConfig | None = None,
    drive_off: bool = False,
    physics_dt: float = 0.01,
) -> CandidateResult:
    """Redevelop one independent Agent and return descriptive, unscored results.

    Execution failures are explicit failed results with saved evidence. Invalid
    requests and artifact/provenance I/O errors raise instead of returning a score.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a nonempty string")
    if not isinstance(genotype, CPPNGenome) or (
        genotype.input_size,
        genotype.output_size,
    ) != (18, 8):
        raise ValueError("genotype must be a CPPN with 18 inputs and 8 outputs")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(drive_off, bool):
        raise ValueError("drive_off must be a boolean")
    if (
        not isinstance(physics_dt, Real)
        or isinstance(physics_dt, bool)
        or physics_dt not in (0.01, 0.005)
    ):
        raise ValueError("physics_dt must be 0.01 or 0.005")
    config = ObservationConfig() if config is None else config
    if not isinstance(config, ObservationConfig):
        raise ValueError("config must be an ObservationConfig")
    out = Path(out_dir)
    record = _run_trial(
        genotype.to_dict(),
        int(seed),
        drive_off,
        float(physics_dt),
        out,
        config,
        candidate_id=candidate_id,
    )
    if record["status"] not in ("completed", "failed"):
        raise ValueError("candidate execution returned an invalid status")
    result = CandidateResult(
        candidate_id=candidate_id,
        seed=int(seed),
        drive_off=drive_off,
        physics_dt=float(physics_dt),
        status="completed" if record["status"] == "completed" else "failed",
        reason=record["reason"],
        completed_control_intervals=record["completed_control_intervals"],
        births=record["births"],
        final_cells=record["final_cells"],
        max_cells=record["max_cells"],
        growth_budget=record["growth_budget"],
        physics_diagnostics=(
            PhysicsDiagnostics(**record["physics_diagnostics"])
            if record["physics_diagnostics"] is not None
            else None
        ),
        out_dir=out,
        elapsed_seconds=record["elapsed_seconds"],
    )
    try:
        _provenance(out, config, candidate=result)
        _write_json(out / "result.json", result.to_dict())
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        _write_json(
            out / "status.json",
            {
                **record,
                "status": "failed",
                "reason": "artifact_or_provenance_error",
                "message": str(exc),
                "execution_status": record["status"],
                "execution_reason": record["reason"],
            },
        )
        raise
    return result


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _compare(out: Path, seed: int, condition: str) -> dict:
    directory = out / f"seed_{seed}" / condition
    normal = _read_records(directory / "normal" / "frames.jsonl")
    replay = _read_records(directory / "replay" / "frames.jsonl")
    half = _read_records(directory / "half_step" / "frames.jsonl")
    normal_events = _read_records(directory / "normal" / "events.jsonl")
    replay_events = _read_records(directory / "replay" / "events.jsonl")
    half_events = _read_records(directory / "half_step" / "events.jsonl")
    statuses = {
        run: json.loads((directory / run / "status.json").read_text(encoding="utf-8"))
        for run, _ in RUNS
    }
    status_match = (
        statuses["normal"]["status"] == statuses["replay"]["status"]
        and statuses["normal"]["reason"] == statuses["replay"]["reason"]
        and statuses["normal"]["completed_control_intervals"]
        == statuses["replay"]["completed_control_intervals"]
    )
    replay_match = status_match and normal == replay and normal_events == replay_events
    if status_match and statuses["normal"]["status"] == "failed":
        failure_normal = json.loads(
            (directory / "normal" / "failure.json").read_text(encoding="utf-8")
        )
        failure_replay = json.loads(
            (directory / "replay" / "failure.json").read_text(encoding="utf-8")
        )
        replay_match = replay_match and failure_normal == failure_replay
    if (
        statuses["normal"]["reason"] == "timeout"
        or statuses["replay"]["reason"] == "timeout"
    ):
        replay_match = False
    comparable = 0
    topology_differences = 0
    max_position = 0.0
    max_velocity = 0.0
    for first, second in zip(normal, half):
        first_ids = [c["id"] for c in first["cells"]]
        second_ids = [c["id"] for c in second["cells"]]
        if first_ids != second_ids or first["bonds"] != second["bonds"]:
            topology_differences += 1
            continue
        comparable += 1
        for a, b in zip(first["cells"], second["cells"]):
            max_position = max(
                max_position,
                float(np.linalg.norm(np.array(a["position"]) - b["position"])),
            )
            max_velocity = max(
                max_velocity,
                float(np.linalg.norm(np.array(a["velocity"]) - b["velocity"])),
            )
    result = {
        "seed": seed,
        "condition": condition,
        "replay_matches": replay_match,
        "normal_status": statuses["normal"]["status"],
        "replay_status": statuses["replay"]["status"],
        "half_step_status": statuses["half_step"]["status"],
        "half_step_event_matches": normal_events == half_events,
        "half_step_topology_different_frames": topology_differences,
        "half_step_comparable_frames": comparable,
        "normal_frames": len(normal),
        "half_step_frames": len(half),
        "max_position_difference_on_matching_topology": (
            max_position if comparable else None
        ),
        "max_velocity_difference_on_matching_topology": (
            max_velocity if comparable else None
        ),
        "interpretation": "closed-loop observations; no physics-test tolerance imposed",
    }
    _write_json(directory / "comparison.json", result)
    return result


def _plot(out):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 12), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 3)
    for row, seed in enumerate(SEEDS):
        for condition, style in (("active", "-"), ("drive_off", "--")):
            with np.load(
                out / f"seed_{seed}" / condition / "normal" / "metrics.npz",
                allow_pickle=False,
            ) as data:
                for column, key in enumerate(("cells", "max_speed", "max_bond_length")):
                    axes[row, column].plot(
                        data["time"], data[key], style, label=condition
                    )
                    axes[row, column].set_title(f"seed {seed}: {key}")
                    axes[row, column].set_xlabel("simulation time")
                    axes[row, column].grid(alpha=0.25)
                    axes[row, column].legend(fontsize=7)
    figure.savefig(out / "observations.png", dpi=120)


def run_observations(
    out_dir: Path | str, *, config: ObservationConfig | None = None
) -> Path:
    config = ObservationConfig() if config is None else config
    if not isinstance(config, ObservationConfig):
        raise ValueError("config must be an ObservationConfig")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rollouts = []
    _write_json(out / "status.json", {"status": "running", "finished_rollouts": 0})
    try:
        _provenance(out, config)
        for seed in SEEDS:
            seed_dir = out / f"seed_{seed}"
            seed_dir.mkdir()
            genotype = initial_genome(seed).to_dict()
            _write_json(seed_dir / "genotype.json", genotype)
            for condition in ("active", "drive_off"):
                for run, dt in RUNS:
                    result = run_candidate(
                        f"seed-{seed}-{condition}-{run}",
                        CPPNGenome.from_dict(genotype),
                        seed=seed,
                        drive_off=condition == "drive_off",
                        physics_dt=dt,
                        out_dir=seed_dir / condition / run,
                        config=config,
                    )
                    record = result.to_dict()
                    record.update(condition=condition, run=run)
                    rollouts.append(record)
                    _write_json(
                        out / "status.json",
                        {
                            "status": "running",
                            "finished_rollouts": len(rollouts),
                            "failed_rollouts": sum(
                                r["status"] == "failed" for r in rollouts
                            ),
                        },
                    )
        comparisons = [
            _compare(out, seed, condition)
            for seed in SEEDS
            for condition in ("active", "drive_off")
        ]
        replay_matches = all(r["replay_matches"] for r in comparisons)
        failures = sum(r["status"] == "failed" for r in rollouts)
        _write_json(
            out / "summary.json",
            {
                "rollouts": rollouts,
                "comparisons": comparisons,
                "failed_rollouts": failures,
                "replay_matches": replay_matches,
                "claim": "initial observations, not selection response or evolved motion",
            },
        )
        _plot(out)
        _write_json(
            out / "status.json",
            {
                "status": (
                    "completed"
                    if not failures and replay_matches
                    else "completed_with_failures"
                ),
                "finished_rollouts": len(rollouts),
                "failed_rollouts": failures,
                "replay_matches": replay_matches,
            },
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        _write_json(
            out / "status.json",
            {
                "status": "failed",
                "reason": type(exc).__name__,
                "message": str(exc),
                "finished_rollouts": len(rollouts),
            },
        )
        raise
    return out


def _plot_startup(out: Path, individuals: list[dict]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    counts, gates = figure.subplots(1, 2)
    labels = ("first_birth", "observation_limit", "failed", "not_run")
    values = [
        sum(
            (
                r.get("ended_by") == label
                if label in labels[:2]
                else r["status"] == label
            )
            for r in individuals
        )
        for label in labels
    ]
    counts.bar(labels, values)
    counts.set_title("All 100 predeclared genotypes")
    counts.set_ylabel("individuals")
    counts.tick_params(axis="x", labelrotation=20)
    for r in individuals:
        if r.get("max_gate") is not None:
            gates.scatter(
                r["seed"],
                r["max_gate"],
                color="green" if r.get("ended_by") == "first_birth" else "gray",
                s=15,
            )
    gates.axhline(0.5, color="black", linestyle="--", label="request threshold")
    gates.set_title("Max gate before first birth or observation end")
    gates.set_xlabel("seed")
    gates.set_ylabel("Raw output 4")
    gates.legend()
    figure.savefig(out / "startup.png", dpi=120)


def run_startup_diagnostic(
    out_dir: str | Path,
    *,
    config: ObservationConfig | None = None,
) -> Path:
    config = ObservationConfig() if config is None else config
    if not isinstance(config, ObservationConfig):
        raise ValueError("config must be an ObservationConfig")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config.timeout_seconds
    individuals = []
    _write_json(out / "status.json", {"status": "running", "finished_individuals": 0})
    try:
        _provenance(out, config, startup=True)
        # Preserve every predeclared genotype, including those not run on timeout.
        for seed in range(100):
            seed_dir = out / f"seed_{seed}"
            seed_dir.mkdir()
            _write_json(seed_dir / "genotype.json", initial_genome(seed).to_dict())
        for seed in range(100):
            seed_dir = out / f"seed_{seed}"
            if time.monotonic() > deadline:
                record = {
                    "seed": seed,
                    "status": "not_run",
                    "reason": "global_timeout",
                    "ended_by": None,
                    "max_gate": None,
                    "initial_gate": None,
                }
                _write_json(seed_dir / "status.json", record)
            else:
                genotype = json.loads(
                    (seed_dir / "genotype.json").read_text(encoding="utf-8")
                )
                record = _run_trial(
                    genotype,
                    seed,
                    False,
                    0.01,
                    seed_dir / "startup",
                    config,
                    stop_on_first_birth=True,
                    overall_deadline=deadline,
                )
            individuals.append(record)
            _write_json(
                out / "status.json",
                {
                    "status": "running",
                    "finished_individuals": len(individuals),
                },
            )
        completed = sum(r["status"] == "completed" for r in individuals)
        startup_count = sum(r.get("ended_by") == "first_birth" for r in individuals)
        no_startup = sum(r.get("ended_by") == "observation_limit" for r in individuals)
        failed = sum(r["status"] == "failed" for r in individuals)
        not_run = sum(r["status"] == "not_run" for r in individuals)
        summary = {
            "individuals": individuals,
            "completed_count": completed,
            "startup_count": startup_count,
            "no_startup_count": no_startup,
            "failed_count": failed,
            "not_run_count": not_run,
            "startup_fraction_of_completed": (
                startup_count / completed if completed else None
            ),
            "budget_elapsed_seconds": time.monotonic() - started,
            "claim": "startup under this initialization and window, not continued growth or evolution",
        }
        _write_json(out / "summary.json", summary)
        arrays = {
            "seed": np.arange(100),
            "completed": np.array(
                [r["status"] == "completed" for r in individuals], dtype=int
            ),
            "started": np.array(
                [r.get("ended_by") == "first_birth" for r in individuals], dtype=int
            ),
            "failed": np.array(
                [r["status"] == "failed" for r in individuals], dtype=int
            ),
            "not_run": np.array(
                [r["status"] == "not_run" for r in individuals], dtype=int
            ),
            "max_gate": np.array(
                [
                    r["max_gate"] if r["max_gate"] is not None else np.nan
                    for r in individuals
                ]
            ),
        }
        write_metrics_csv_npz(
            out,
            arrays,
            csv_name="startup.csv",
            npz_name="startup.npz",
            header=arrays.keys(),
        )
        _plot_startup(out, individuals)
        _write_json(
            out / "status.json",
            {
                "status": (
                    "incomplete"
                    if not_run
                    else "completed_with_failures" if failed else "completed"
                ),
                "completed_count": completed,
                "startup_count": startup_count,
                "no_startup_count": no_startup,
                "failed_count": failed,
                "not_run_count": not_run,
            },
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        _write_json(
            out / "status.json",
            {
                "status": "failed",
                "reason": type(exc).__name__,
                "message": str(exc),
                "finished_individuals": len(individuals),
            },
        )
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument(
        "--startup",
        action="store_true",
        help="Diagnose startup for seeds 0..99 with one total timeout",
    )
    args = parser.parse_args()
    out = (
        run_startup_diagnostic(args.out) if args.startup else run_observations(args.out)
    )
    print(out)
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise SystemExit("Observations contain failures; see summary.json")


if __name__ == "__main__":
    main()
