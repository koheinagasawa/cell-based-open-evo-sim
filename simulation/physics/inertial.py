"""Opt-in 2D inertial physics; independent of legacy message-edge springs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import NoReturn, Protocol

import numpy as np

from simulation.physics.repulsion import compute_repulsion_forces


class Particle(Protocol):
    id: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    mass: float


@dataclass(frozen=True)
class PhysicalBond:
    first: str
    second: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.first, str)
            or not isinstance(self.second, str)
            or not self.first
            or not self.second
            or self.first == self.second
        ):
            raise ValueError("physical bond requires two distinct nonempty cell ids")
        first, second = sorted((self.first, self.second))
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)


@dataclass(frozen=True)
class InertialPhysicsConfig:
    dt: float = 0.01
    drag: float = 1.0
    spring_stiffness: float = 10.0
    repulsion_stiffness: float = 100.0
    max_strain: float = 0.2
    max_cells: int = 64
    max_speed: float = 10.0
    max_distance: float = 100.0
    max_overlap: float = 0.2
    max_bond_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "dt",
            "drag",
            "spring_stiffness",
            "repulsion_stiffness",
            "max_strain",
            "max_speed",
            "max_distance",
            "max_overlap",
            "max_bond_ratio",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            if name in ("max_strain", "max_overlap") and value >= 1:
                raise ValueError(f"{name} must be less than 1")
            if name == "max_bond_ratio" and value <= 1:
                raise ValueError("max_bond_ratio must be greater than 1")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.max_cells, Integral)
            or isinstance(self.max_cells, bool)
            or self.max_cells < 1
        ):
            raise ValueError("max_cells must be a positive integer")
        object.__setattr__(self, "max_cells", int(self.max_cells))


class PhysicsFailure(FloatingPointError):
    """A rejected physical state, including uncommitted candidate states."""

    def __init__(
        self,
        reason: str,
        stage: str,
        cell_ids: tuple[str, ...],
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        super().__init__(f"{reason} ({stage}): {', '.join(cell_ids)}")
        self.reason = reason
        self.stage = stage
        self.cell_ids = cell_ids
        self.positions = positions.copy()
        self.velocities = velocities.copy()


@dataclass(frozen=True)
class PhysicsDiagnostics:
    max_speed: float = 0.0
    max_overlap: float = 0.0
    max_bond_length: float = 0.0


class InertialPhysicsSolver:
    """Update all velocities, then positions, from one common force snapshot.

    Commands are already interpreted, finite values in [-1, 1]. Missing commands
    are neutral. Raw-output clipping belongs to the Interpreter, not this solver.
    """

    def __init__(
        self,
        bonds: Sequence[PhysicalBond] = (),
        *,
        config: InertialPhysicsConfig | None = None,
    ) -> None:
        self.config = config if config is not None else InertialPhysicsConfig()
        if not isinstance(self.config, InertialPhysicsConfig):
            raise ValueError("config must be an InertialPhysicsConfig")
        if any(not isinstance(b, PhysicalBond) for b in bonds):
            raise ValueError("bonds must contain PhysicalBond values")
        self.bonds = tuple(sorted(bonds, key=lambda b: (b.first, b.second)))
        if len(set(self.bonds)) != len(self.bonds):
            raise ValueError("duplicate physical bond")

    def _read(
        self, particles: Sequence[Particle]
    ) -> tuple[list[Particle], tuple[str, ...], np.ndarray, np.ndarray, dict[str, int]]:
        if any(not isinstance(p.id, str) or not p.id for p in particles):
            raise ValueError("particle id must be a nonempty string")
        ordered = sorted(particles, key=lambda p: p.id)
        ids = tuple(p.id for p in ordered)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate particle id")
        for particle in ordered:
            for name in ("radius", "mass"):
                value = getattr(particle, name)
                if (
                    not isinstance(value, Real)
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"particle {particle.id}: invalid {name}")
            if np.shape(particle.position) != (2,) or np.shape(particle.velocity) != (
                2,
            ):
                raise ValueError(
                    f"particle {particle.id}: expected 2D position and velocity"
                )
        positions = np.array([p.position for p in ordered], dtype=float).reshape(-1, 2)
        velocities = np.array([p.velocity for p in ordered], dtype=float).reshape(-1, 2)
        index = {cell_id: i for i, cell_id in enumerate(ids)}
        degree = dict.fromkeys(ids, 0)
        for bond in self.bonds:
            if bond.first not in index or bond.second not in index:
                raise ValueError(f"dangling physical bond: {bond}")
            degree[bond.first] += 1
            degree[bond.second] += 1
        if any(value > 4 for value in degree.values()):
            raise ValueError("physical degree exceeds four")
        return ordered, ids, positions, velocities, index

    def _inspect(
        self,
        ordered: Sequence[Particle],
        ids: tuple[str, ...],
        positions: np.ndarray,
        velocities: np.ndarray,
        index: Mapping[str, int],
        stage: str,
    ) -> PhysicsDiagnostics:
        config = self.config

        def fail(reason: str, affected: Iterable[str]) -> NoReturn:
            raise PhysicsFailure(reason, stage, tuple(affected), positions, velocities)

        if len(ids) > config.max_cells:
            fail("cell_limit", ids)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            fail("nonfinite_state", ids)
        speeds = np.hypot(velocities[:, 0], velocities[:, 1])
        distances = np.hypot(positions[:, 0], positions[:, 1])
        if np.any(speeds > config.max_speed):
            fail(
                "speed_limit",
                [ids[i] for i in np.flatnonzero(speeds > config.max_speed)],
            )
        if np.any(distances > config.max_distance):
            fail(
                "distance_limit",
                [ids[i] for i in np.flatnonzero(distances > config.max_distance)],
            )
        max_overlap = 0.0
        for i, first in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                length = float(np.linalg.norm(positions[j] - positions[i]))
                radii = first.radius + ordered[j].radius
                # Compare distances to avoid cancelling at the inclusive boundary.
                if length < radii * (1 - config.max_overlap):
                    fail("overlap_limit", (ids[i], ids[j]))
                max_overlap = max(max_overlap, (radii - length) / radii)
        max_length = 0.0
        for bond in self.bonds:
            i, j = index[bond.first], index[bond.second]
            length = float(np.linalg.norm(positions[j] - positions[i]))
            if length > (ordered[i].radius + ordered[j].radius) * config.max_bond_ratio:
                fail("bond_limit", (bond.first, bond.second))
            max_length = max(max_length, length)
        return PhysicsDiagnostics(
            max_speed=float(np.max(speeds)) if len(ids) else 0.0,
            max_overlap=max_overlap,
            max_bond_length=max_length,
        )

    def inspect(self, particles: Sequence[Particle]) -> PhysicsDiagnostics:
        ordered, ids, positions, velocities, index = self._read(particles)
        return self._inspect(ordered, ids, positions, velocities, index, "before")

    def step(
        self,
        particles: Sequence[Particle],
        commands: Mapping[str, float] | None = None,
    ) -> PhysicsDiagnostics:
        ordered, ids, positions, velocities, index = self._read(particles)
        before = self._inspect(ordered, ids, positions, velocities, index, "before")
        commands = {} if commands is None else commands
        for cell_id, value in commands.items():
            if cell_id not in index:
                raise ValueError(f"command references unknown cell: {cell_id}")
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or not -1 <= value <= 1
            ):
                raise ValueError(f"invalid interpreted command for {cell_id}")
        if not ordered:
            return before
        config = self.config
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                forces = compute_repulsion_forces(
                    ordered, stiffness=config.repulsion_stiffness
                )
                forces -= config.drag * velocities
                for bond in self.bonds:
                    i, j = index[bond.first], index[bond.second]
                    relative = positions[j] - positions[i]
                    length = float(np.linalg.norm(relative))
                    command = 0.5 * (
                        commands.get(bond.first, 0) + commands.get(bond.second, 0)
                    )
                    target = (ordered[i].radius + ordered[j].radius) * (
                        1 + config.max_strain * command
                    )
                    force = (
                        config.spring_stiffness * (length - target) * relative / length
                    )
                    forces[i] += force
                    forces[j] -= force
                masses = np.array([p.mass for p in ordered], dtype=float)
                new_velocity = velocities + forces / masses[:, None] * config.dt
                new_position = positions + new_velocity * config.dt
        except FloatingPointError as exc:
            raise PhysicsFailure(
                "nonfinite_calculation", "integration", ids, positions, velocities
            ) from exc
        after = self._inspect(ordered, ids, new_position, new_velocity, index, "after")
        for i, particle in enumerate(ordered):
            particle.velocity = new_velocity[i].copy()
            particle.position = new_position[i].copy()
        return PhysicsDiagnostics(
            max(before.max_speed, after.max_speed),
            max(before.max_overlap, after.max_overlap),
            max(before.max_bond_length, after.max_bond_length),
        )
