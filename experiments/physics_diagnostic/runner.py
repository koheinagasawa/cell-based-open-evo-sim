"""Run the four approved inertial diagnostics and the active-spring control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np

from experiments.common.metrics import write_metrics_csv_npz
from simulation.cell import Cell
from simulation.physics.inertial import (
    InertialPhysicsConfig,
    InertialPhysicsSolver,
    PhysicalBond,
    PhysicsDiagnostics,
    PhysicsFailure,
)

CASES = (
    "free_decay",
    "passive_spring",
    "contact",
    "active_spring",
    "active_spring_off",
)
CONTROL_INTERVAL = 0.1
CONTROL_STEPS = 100
POSITION_TOLERANCE = 0.01
VELOCITY_TOLERANCE = 0.05
CENTER_TOLERANCE = 1e-10
_METRICS = ("time", "max_speed", "max_overlap", "max_bond_length")


def drive_for_interval(case: str, interval: int) -> float:
    if case not in CASES:
        raise ValueError(f"unknown diagnostic: {case}")
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or not 0 <= interval < CONTROL_STEPS
    ):
        raise ValueError("interval must be an integer from 0 to 99")
    if case != "active_spring" or interval >= 60:
        return 0.0
    return 1.0 if interval < 30 else -1.0


def _initial_case(case: str):
    if case == "free_decay":
        positions = [(0.0, 0.0)]
        endpoints = []
    elif case == "passive_spring":
        positions = [(-0.6, 0.0), (0.6, 0.0)]
        endpoints = [("a", "b")]
    elif case == "contact":
        positions = [(-0.475, 0.0), (0.475, 0.0)]
        endpoints = []
    elif case in ("active_spring", "active_spring_off"):
        positions = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        endpoints = [("a", "b"), ("b", "c")]
    else:
        raise ValueError(f"unknown diagnostic: {case}")
    cells = [
        Cell(
            position,
            genome=None,
            id=chr(ord("a") + i),
            velocity=(1.0, 0.0) if case == "free_decay" else (0.0, 0.0),
        )
        for i, position in enumerate(positions)
    ]
    return cells, tuple(PhysicalBond(*pair) for pair in endpoints)


def _json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _snapshot(cells):
    return (
        np.array([c.position for c in cells], dtype=float),
        np.array([c.velocity for c in cells], dtype=float),
    )


def _metadata(out: Path, timeout_seconds: float) -> None:
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(
            ["git", "--no-pager", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout

    source_paths = [
        Path(__file__).resolve(),
        root / "simulation" / "cell.py",
        root / "simulation" / "physics" / "inertial.py",
        root / "simulation" / "physics" / "repulsion.py",
        root / "experiments" / "common" / "metrics.py",
        root / "doc" / "MinimalOrganismContract_ja.md",
        root / "requirements-dev.txt",
    ]
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(root)
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "workspace.patch").write_text(git("diff", "HEAD", "--"), encoding="utf-8")
    initial = {}
    for name in CASES:
        cells, bonds = _initial_case(name)
        position, velocity = _snapshot(cells)
        initial[name] = {
            "ids": [c.id for c in cells],
            "position": position.tolist(),
            "velocity": velocity.tolist(),
            "bonds": [asdict(b) for b in bonds],
        }
    _json(
        out / "metadata.json",
        {
            "schema_version": 1,
            "purpose": "fixed numerical diagnostics; no CPPN, development or evolution",
            "command": sys.argv,
            "revision": git("rev-parse", "HEAD").strip(),
            "workspace_status": git("status", "--porcelain"),
            "source_hashes": hashes,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
            },
            "physics": {**asdict(InertialPhysicsConfig()), "mass": 1.0, "radius": 0.5},
            "physics_steps": [0.01, 0.005],
            "control_interval": CONTROL_INTERVAL,
            "control_steps": CONTROL_STEPS,
            "timeout_seconds": timeout_seconds,
            "seeds": [],
            "cases": initial,
            "drive_schedule": [
                {"first_interval": 0, "end_interval": 30, "command": 1},
                {"first_interval": 30, "end_interval": 60, "command": -1},
                {"first_interval": 60, "end_interval": 100, "command": 0},
            ],
            "tolerances": {
                "position": POSITION_TOLERANCE,
                "velocity": VELOCITY_TOLERANCE,
                "center": CENTER_TOLERANCE,
            },
            "frames": "initial and control endpoints; command is the last held command",
        },
    )


def _run_case(case: str, dt: float, out: Path, timeout_seconds: float) -> None:
    started = time.monotonic()
    cells, bonds = _initial_case(case)
    solver = InertialPhysicsSolver(
        bonds, config=replace(InertialPhysicsConfig(), dt=dt)
    )
    substeps = round(CONTROL_INTERVAL / dt)
    out.mkdir(parents=True, exist_ok=False)
    _json(
        out / "config.json",
        {
            "case": case,
            "physics": asdict(solver.config),
            "initial_position": [c.position.tolist() for c in cells],
            "initial_velocity": [c.velocity.tolist() for c in cells],
            "ids": [c.id for c in cells],
            "bonds": [asdict(b) for b in bonds],
            "control_interval": CONTROL_INTERVAL,
            "control_steps": CONTROL_STEPS,
            "timeout_seconds": timeout_seconds,
        },
    )
    positions, velocities, times, commands = [], [], [], []
    metrics = {name: [] for name in _METRICS}
    completed_steps = 0
    last_safe_position = np.empty((0, 2))
    last_safe_velocity = np.empty((0, 2))

    def check_deadline():
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"{case} dt={dt}: exceeded {timeout_seconds} seconds")

    def sample(t, command, diagnostic):
        position, velocity = _snapshot(cells)
        positions.append(position)
        velocities.append(velocity)
        times.append(t)
        commands.append(np.full(len(cells), command))
        metrics["time"].append(t)
        for name in _METRICS[1:]:
            metrics[name].append(getattr(diagnostic, name))

    def save_frames():
        np.savez_compressed(
            out / "frames.npz",
            time=np.array(times),
            ids=np.array([c.id for c in cells]),
            position=np.array(positions).reshape(-1, len(cells), 2),
            velocity=np.array(velocities).reshape(-1, len(cells), 2),
            command=np.array(commands).reshape(-1, len(cells)),
            bonds=np.array([(b.first, b.second) for b in bonds], dtype=str).reshape(
                -1, 2
            ),
        )
        write_metrics_csv_npz(
            out,
            {name: np.array(values) for name, values in metrics.items()},
            header=_METRICS,
        )

    with (out / "events.jsonl").open("w", encoding="utf-8") as events:

        def event(value):
            events.write(json.dumps(value, allow_nan=False) + "\n")
            events.flush()

        event({"event": "start", "case": case, "dt": dt})
        try:
            sample(0.0, drive_for_interval(case, 0), solver.inspect(cells))
            last_safe_position, last_safe_velocity = _snapshot(cells)
            check_deadline()
            for interval in range(CONTROL_STEPS):
                command = drive_for_interval(case, interval)
                held = {c.id: command for c in cells}
                maximum = PhysicsDiagnostics()
                for _ in range(substeps):
                    check_deadline()
                    result = solver.step(cells, held)
                    completed_steps += 1
                    last_safe_position, last_safe_velocity = _snapshot(cells)
                    maximum = PhysicsDiagnostics(
                        *(
                            max(getattr(maximum, name), getattr(result, name))
                            for name in _METRICS[1:]
                        )
                    )
                sample((interval + 1) * CONTROL_INTERVAL, command, maximum)
            check_deadline()
        except (PhysicsFailure, TimeoutError) as exc:
            failed_position, failed_velocity = _snapshot(cells)
            if isinstance(exc, PhysicsFailure):
                failed_position, failed_velocity = exc.positions, exc.velocities
            np.savez_compressed(
                out / "failure.npz",
                ids=np.array([c.id for c in cells]),
                last_safe_position=last_safe_position,
                last_safe_velocity=last_safe_velocity,
                last_completed_position=(
                    positions[-1] if positions else np.empty((0, 2))
                ),
                last_completed_velocity=(
                    velocities[-1] if velocities else np.empty((0, 2))
                ),
                failed_position=failed_position,
                failed_velocity=failed_velocity,
            )
            event(
                {
                    "event": "failure",
                    "reason": (
                        exc.reason if isinstance(exc, PhysicsFailure) else "timeout"
                    ),
                    "stage": (
                        exc.stage if isinstance(exc, PhysicsFailure) else "execution"
                    ),
                    "cell_ids": (
                        list(exc.cell_ids) if isinstance(exc, PhysicsFailure) else []
                    ),
                    "completed_physics_steps": completed_steps,
                    "last_safe_time": completed_steps * dt,
                    "has_safe_state": bool(last_safe_position.size),
                    "failed_time": (
                        (completed_steps + 1) * dt
                        if isinstance(exc, PhysicsFailure) and exc.stage == "after"
                        else completed_steps * dt
                    ),
                    "completed_control_intervals": max(0, len(times) - 1),
                }
            )
            raise
        else:
            event(
                {"event": "completed", "physics_steps": completed_steps, "time": 10.0}
            )
        finally:
            save_frames()


def _compare_case(name: str, out: Path) -> dict:
    with np.load(out / "dt_0.01" / "frames.npz", allow_pickle=False) as data:
        coarse = {key: data[key] for key in ("time", "position", "velocity")}
    with np.load(out / "dt_0.005" / "frames.npz", allow_pickle=False) as data:
        fine = {key: data[key] for key in ("time", "position", "velocity")}
    expected_time = np.arange(CONTROL_STEPS + 1) * CONTROL_INTERVAL
    for arrays in (coarse, fine):
        if not np.array_equal(arrays["time"], expected_time):
            raise ValueError(f"{name}: missing or mismatched comparison times")
    difference = {
        key + "_difference": np.linalg.norm(coarse[key] - fine[key], axis=2)
        for key in ("position", "velocity")
    }
    np.savez_compressed(out / "comparison.npz", time=expected_time, **difference)
    write_metrics_csv_npz(
        out,
        {
            "time": expected_time,
            **{key: values.max(axis=1) for key, values in difference.items()},
        },
        csv_name="comparison.csv",
        npz_name="comparison_metrics.npz",
        header=("time", "position_difference", "velocity_difference"),
    )
    result = {
        "name": name,
        "max_position_difference": float(difference["position_difference"].max()),
        "max_velocity_difference": float(difference["velocity_difference"].max()),
    }
    passed = (
        result["max_position_difference"] <= POSITION_TOLERANCE
        and result["max_velocity_difference"] <= VELOCITY_TOLERANCE
    )
    if name == "free_decay":
        analytic_velocity = np.zeros_like(coarse["velocity"])
        analytic_position = np.zeros_like(coarse["position"])
        analytic_velocity[:, 0, 0] = np.exp(-expected_time)
        analytic_position[:, 0, 0] = 1 - np.exp(-expected_time)
        analytic = {}
        for label, arrays in (("dt_0.01", coarse), ("dt_0.005", fine)):
            errors = {
                "position": np.linalg.norm(
                    arrays["position"] - analytic_position, axis=2
                ),
                "velocity": np.linalg.norm(
                    arrays["velocity"] - analytic_velocity, axis=2
                ),
            }
            analytic[label] = {key: float(value.max()) for key, value in errors.items()}
            passed = passed and (
                analytic[label]["position"] <= POSITION_TOLERANCE
                and analytic[label]["velocity"] <= VELOCITY_TOLERANCE
            )
        np.savez_compressed(
            out / "analytic.npz",
            time=expected_time,
            position=analytic_position,
            velocity=analytic_velocity,
        )
        result["analytic_errors"] = analytic
    if name in ("passive_spring", "contact"):
        drift = max(
            float(
                np.linalg.norm(
                    arrays["position"].mean(axis=1)
                    - arrays["position"][0].mean(axis=0),
                    axis=1,
                ).max()
            )
            for arrays in (coarse, fine)
        )
        result["max_center_drift"] = drift
        passed = passed and drift <= CENTER_TOLERANCE
    result["passed"] = bool(passed)
    return result


def _plot(out: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 14), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(CASES), 3)
    for row, name in enumerate(CASES):
        for label, style in (("dt_0.01", "-"), ("dt_0.005", "--")):
            with np.load(out / name / label / "frames.npz", allow_pickle=False) as data:
                t, p, v = data["time"], data["position"], data["velocity"]
                if p.shape[1] == 1:
                    axes[row, 0].plot(t, p[:, 0, 0], style, label=label)
                else:
                    for i in range(p.shape[1] - 1):
                        axes[row, 0].plot(
                            t,
                            np.linalg.norm(p[:, i + 1] - p[:, i], axis=1),
                            style,
                            label=f"{label} pair {i}-{i+1}",
                        )
                axes[row, 1].plot(
                    t, np.linalg.norm(v, axis=2).max(axis=1), style, label=label
                )
        with np.load(out / name / "comparison.npz", allow_pickle=False) as data:
            for key in ("position_difference", "velocity_difference"):
                axes[row, 2].plot(data["time"], data[key].max(axis=1), label=key)
        axes[row, 0].set_title(
            name + (" / x" if name == "free_decay" else " / distance")
        )
        axes[row, 1].set_title("max sampled speed")
        axes[row, 2].set_title("half-step difference")
        for axis in axes[row]:
            axis.set_xlabel("simulation time")
            axis.legend(fontsize=6)
            axis.grid(alpha=0.25)
    figure.savefig(out / "diagnostics.png", dpi=120)


def run_diagnostics(out_dir: str | Path, *, timeout_seconds: float = 300.0) -> Path:
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    completed = 0
    current = None
    _json(out / "status.json", {"status": "running", "completed_rollouts": 0})
    try:
        _metadata(out, timeout_seconds)
        for case in CASES:
            for dt in (0.01, 0.005):
                current = {"case": case, "dt": dt}
                _json(
                    out / "status.json",
                    {
                        "status": "running",
                        "completed_rollouts": completed,
                        **current,
                    },
                )
                _run_case(case, dt, out / case / f"dt_{dt}", timeout_seconds)
                completed += 1
        results = [_compare_case(case, out / case) for case in CASES]
        passed = all(result["passed"] for result in results)
        _json(out / "summary.json", {"passed": passed, "cases": results})
        _plot(out)
        if not passed:
            raise FloatingPointError("acceptance criteria failed; see summary.json")
    except (
        PhysicsFailure,
        TimeoutError,
        FloatingPointError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        reason = (
            exc.reason
            if isinstance(exc, PhysicsFailure)
            else "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
        )
        _json(
            out / "status.json",
            {
                "status": "failed",
                "reason": reason,
                "message": str(exc),
                "completed_rollouts": completed,
                "current": current,
            },
        )
        raise
    _json(out / "status.json", {"status": "completed", "completed_rollouts": completed})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    print(run_diagnostics(args.out))


if __name__ == "__main__":
    main()
