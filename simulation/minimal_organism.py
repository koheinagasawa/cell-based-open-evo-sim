"""Shared CPPN development and physical actuation for the minimal organism."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from simulation.agent import Agent
from simulation.cell import Cell
from simulation.cppn import CPPNGenome
from simulation.interpreter import ProfiledInterpreter, SlotBasedInterpreter
from simulation.physics.inertial import (
    InertialPhysicsConfig,
    InertialPhysicsSolver,
    PhysicalBond,
    PhysicsFailure,
)
from simulation.policies import ConstantMaintenance
from simulation.world import World


class MinimalOrganismInterpreter(SlotBasedInterpreter):
    def __init__(self) -> None:
        super().__init__({"state": slice(0, 4), "bud": slice(4, 7), "contraction": 7})

    def interpret(self, output, profile=0) -> dict:
        if profile != 0:
            raise ValueError("the minimal organism requires profile 0")
        if isinstance(output, dict):
            raise ValueError("expected an 8-element Raw Vector, not keyed outputs")
        vector = np.asarray(output, dtype=float)
        if vector.shape != (8,) or not np.isfinite(vector).all():
            raise ValueError("expected a finite 8-element Raw Vector")
        slots = super().interpret(vector, profile)
        slots["contraction"] = float(np.clip(slots["contraction"], -1, 1))
        return slots


class _BodyCell(Cell):
    def __init__(
        self,
        world: MinimalOrganismWorld,
        birth_order: int,
        position,
        development_position,
        energy: float,
        velocity,
    ) -> None:
        self._world = world
        self._birth_order = birth_order
        self._development_position = tuple(float(x) for x in development_position)
        super().__init__(
            position,
            world.agent.genome,
            id=f"cell-{birth_order:06d}",
            agent_id=world.agent.id,
            interpreter=world.agent.interpreter,
            profile=0,
            state_size=4,
            max_neighbors=0,
            include_num_neighbors=False,
            energy_init=energy,
            energy_max=63,
            radius=0.5,
            mass=1,
            velocity=velocity,
        )

    @property
    def birth_order(self) -> int:
        return self._birth_order

    @property
    def development_position(self) -> tuple[float, ...]:
        return self._development_position

    @property
    def profile(self) -> int:
        return self._fixed_profile

    @profile.setter
    def profile(self, value) -> None:
        if hasattr(self, "_fixed_profile"):
            raise AttributeError("Profile is fixed at Cell birth")
        if value != 0:
            raise ValueError("the minimal organism requires profile 0")
        self._fixed_profile = 0

    def sense(self, neighbors):
        return self._world.inputs_for(self.id)

    def act(self, inputs):
        raw = self.genome.activate(inputs, rng=self.rng)
        slots = self.interpreter.interpret(raw, profile=self.profile)
        self.raw_output = raw.copy()
        self.output_slots = slots
        self.next_state = slots["state"].copy()


@dataclass(frozen=True)
class _BirthReservation:
    parent_id: str
    offset: tuple[float, float]
    energy_before: float
    reserved_cost: float = 1.0


class _ReservedBudding:
    def apply(self, world, parent, value, spawn_fn) -> None:
        # Birth is finalized after physics, not through the legacy spawn buffer.
        world.reserve_birth(parent, value)


class MinimalOrganismWorld(World):
    """One Agent, fixed Profile, direct CPPN evaluation and interval-end births.

    Existing World policies and Cell sensing remain unchanged. This specialized
    path uses the same World control/physics scheduling, without fields or messages.
    """

    def __init__(
        self,
        genome: CPPNGenome,
        *,
        seed: int = 0,
        physics_dt: float = 0.01,
        drive_off: bool = False,
    ):
        if not isinstance(genome, CPPNGenome) or (
            genome.input_size,
            genome.output_size,
        ) != (18, 8):
            raise ValueError(
                "the minimal organism requires a CPPN with 18 inputs and 8 outputs"
            )
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
        interpreter = ProfiledInterpreter(
            {0: MinimalOrganismInterpreter()}, default_profile=0
        )
        self.agent = Agent(genome, interpreter, id="agent-000000")
        self.events: list[dict] = []
        self.pending_births: list[_BirthReservation] = []
        self.physics_steps = 0
        self.failed = False
        self.failure: dict | None = None
        self._next_birth_order = 1
        self._phase = "initialization"
        self.drive_off = drive_off
        self._applied_commands: dict[str, float] = {}
        self._commands_ready = False
        self._deadline: float | None = None
        self._interval_diagnostics = {
            "max_speed": 0.0,
            "max_overlap": 0.0,
            "max_bond_length": 0.0,
        }
        self._candidate_ids: tuple[str, ...] | None = None
        first = self._new_cell(0, (0, 0), (0, 0), 63, (0, 0))
        self.agent.cells.append(first)
        super().__init__(
            [first],
            agents=[self.agent],
            seed=int(seed),
            energy_policy=ConstantMaintenance(0),
            reproduction_policy=_ReservedBudding(),
            use_neighbors=False,
            use_fields=False,
            physics_solver=InertialPhysicsSolver(
                config=replace(InertialPhysicsConfig(), dt=float(physics_dt))
            ),
            physics_substeps=10 if physics_dt == 0.01 else 20,
        )
        self.last_completed_snapshot = self.snapshot()

    def _new_cell(self, birth_order, position, development_position, energy, velocity):
        return _BodyCell(
            self, birth_order, position, development_position, energy, velocity
        )

    def inputs_for(self, cell_id: str) -> np.ndarray:
        cells = {c.id: c for c in self.cells}
        cell = cells[cell_id]
        neighbors = []
        for bond in self.physics_solver.bonds:
            if bond.first == cell_id:
                other = cells[bond.second]
            elif bond.second == cell_id:
                other = cells[bond.first]
            else:
                continue
            neighbors.append((max(cell.birth_order, other.birth_order), other))
        neighbors.sort(key=lambda pair: pair[0])
        if len(neighbors) > 4:
            raise ValueError("physical input slots exceed four")
        relative = np.zeros((4, 2))
        mask = np.zeros(4)
        for slot, (_, other) in enumerate(neighbors):
            relative[slot] = other.position - cell.position
            mask[slot] = 1
        return np.concatenate(
            (
                cell.development_position,
                cell.state,
                relative.ravel(),
                mask,
            )
        )

    def _birth_event(self, status: str, parent: Cell, *, at_end=False, **details):
        self.events.append(
            {
                "event": "birth",
                "status": status,
                "parent_id": parent.id,
                "control_interval": self.time,
                "time": (self.time + int(at_end)) * 0.1,
                **details,
            }
        )

    def reserve_birth(self, parent: _BodyCell, value) -> None:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("bud requires finite gate and two direction elements")
        if vector[0] <= 0.5:
            return
        degree = sum(
            parent.id in (b.first, b.second) for b in self.physics_solver.bonds
        )
        reason = None
        if degree >= 4:
            reason = "degree_limit"
        elif parent.energy < 1:
            reason = "energy_limit"
        elif len(self.cells) + len(self.pending_births) >= 64:
            reason = "cell_limit"
        scale = float(np.max(np.abs(vector[1:])))
        if reason is None and scale == 0:
            reason = "zero_direction"
        if reason is not None:
            self._birth_event("rejected", parent, reason=reason, spent_cost=0)
            return
        direction = vector[1:] / scale
        direction /= np.hypot(*direction)
        reservation = _BirthReservation(
            parent.id, (float(direction[0]), float(direction[1])), float(parent.energy)
        )
        self.pending_births.append(reservation)
        self._birth_event(
            "reserved",
            parent,
            offset=list(reservation.offset),
            reserved_cost=1,
            energy_before=reservation.energy_before,
        )

    def _finish_birth(self, reservation: _BirthReservation) -> None:
        parent = next(c for c in self.cells if c.id == reservation.parent_id)
        offset = np.asarray(reservation.offset)
        position = parent.position + offset
        for other in self.cells:
            if other is parent:
                continue
            radii = parent.radius + other.radius
            overlap = (radii - float(np.linalg.norm(position - other.position))) / radii
            if overlap > 0.05 + 1e-9:
                self._birth_event(
                    "rejected",
                    parent,
                    at_end=True,
                    reason="birth_overlap",
                    other_id=other.id,
                    overlap=overlap,
                    offset=list(reservation.offset),
                    reserved_cost=1,
                    spent_cost=0,
                )
                return
        energy = (float(parent.energy) - 1) / 2
        development_position = np.asarray(parent.development_position) + offset
        child = self._new_cell(
            self._next_birth_order,
            position,
            development_position,
            energy,
            parent.velocity,
        )
        solver = InertialPhysicsSolver(
            (*self.physics_solver.bonds, PhysicalBond(parent.id, child.id)),
            config=self.physics_solver.config,
        )
        candidates = [*self.cells, child]
        self._candidate_ids = tuple(sorted(c.id for c in candidates))
        solver.inspect(candidates)
        self._candidate_ids = None
        parent.energy = energy
        self.agent.cells.append(child)
        self.add_cell(child)
        self.physics_solver = solver
        self._next_birth_order += 1
        self._birth_event(
            "born",
            parent,
            at_end=True,
            child_id=child.id,
            offset=list(reservation.offset),
            position=child.position.tolist(),
            velocity=child.velocity.tolist(),
            development_position=list(child.development_position),
            energy_before=reservation.energy_before,
            spent_cost=1,
            parent_energy=energy,
            child_energy=energy,
        )

    def _validate_body(self) -> None:
        if self.pending_births:
            raise ValueError("pending births remained from an earlier interval")
        if len(self.cells) != len(self.agent.cells) or {id(c) for c in self.cells} != {
            id(c) for c in self.agent.cells
        }:
            raise ValueError("Agent and World cell membership differ")
        orders = [c.birth_order for c in self.cells]
        if len(set(orders)) != len(orders):
            raise ValueError("duplicate birth order")
        for cell in self.cells:
            if (
                cell.genome is not self.agent.genome
                or cell.interpreter is not self.agent.interpreter
                or cell.agent_id != self.agent.id
                or cell.profile != 0
                or cell.state.shape != (4,)
                or not np.isfinite(cell.state).all()
                or not np.isfinite(cell.energy)
                or not 0 <= cell.energy <= 63
                or cell.mass != 1
                or cell.radius != 0.5
            ):
                raise ValueError(f"invalid minimal-organism cell: {cell.id}")
        self.physics_solver.inspect(self.cells)

    def _step_physics(self):
        self._phase = "physics"
        self._check_deadline()
        if not self._commands_ready:
            self._applied_commands = {
                c.id: 0.0 if self.drive_off else float(c.output_slots["contraction"])
                for c in self.cells
            }
            self._commands_ready = True
            self._interval_diagnostics = dict.fromkeys(self._interval_diagnostics, 0.0)
        diagnostic = self.physics_solver.step(self.cells, self._applied_commands)
        for name in self._interval_diagnostics:
            self._interval_diagnostics[name] = max(
                self._interval_diagnostics[name], getattr(diagnostic, name)
            )
        self.physics_steps += 1

    def _check_deadline(self):
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise TimeoutError("minimal-organism rollout deadline exceeded")

    def step(self, *, deadline: float | None = None):
        if deadline is not None and (
            not isinstance(deadline, Real)
            or isinstance(deadline, bool)
            or not np.isfinite(deadline)
        ):
            raise ValueError("deadline must be a finite monotonic-clock time")
        if self.failed:
            raise RuntimeError("cannot continue a failed minimal-organism rollout")
        if self.time >= 1000:
            raise RuntimeError(
                "the 1000-control-interval observation limit was reached"
            )
        start = self.time
        self._deadline = deadline
        self._phase = "control"
        self._commands_ready = False
        try:
            self._check_deadline()
            self._validate_body()
            self.cells.sort(key=lambda c: c.birth_order)
            super().step()
            # A control interval is complete only after births are finalized.
            self.time = start
            self._phase = "birth"
            while self.pending_births:
                self._check_deadline()
                self._finish_birth(self.pending_births[0])
                self.pending_births.pop(0)
            self._check_deadline()
            self.time = start + 1
            self.last_completed_snapshot = self.snapshot()
        except (ValueError, FloatingPointError, TypeError, TimeoutError) as exc:
            self.time = start
            self.failed = True
            failure_time = self.physics_steps * self.physics_solver.config.dt
            if isinstance(exc, PhysicsFailure) and exc.stage == "after":
                failure_time += self.physics_solver.config.dt
            for reservation in self.pending_births:
                parent = next(c for c in self.cells if c.id == reservation.parent_id)
                self._birth_event(
                    "unexecuted",
                    parent,
                    reason="rollout_failure",
                    offset=list(reservation.offset),
                    reserved_cost=1,
                    spent_cost=0,
                    time=failure_time,
                )
            self.pending_births.clear()
            self.failure = {
                "reason": (
                    exc.reason
                    if isinstance(exc, PhysicsFailure)
                    else (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else type(exc).__name__
                    )
                ),
                "phase": self._phase,
                "state": self.snapshot(),
                "message": str(exc),
                "time": failure_time,
            }
            if isinstance(exc, PhysicsFailure):
                self.failure["physics_state_ids"] = list(
                    self._candidate_ids
                    if self._candidate_ids is not None
                    else sorted(c.id for c in self.cells)
                )
                self.failure["physics_position"] = exc.positions.tolist()
                self.failure["physics_velocity"] = exc.velocities.tolist()
                self.failure["physics_stage"] = exc.stage
            self.events.append(
                {
                    "event": "failure",
                    "time": failure_time,
                    "control_interval": start,
                    "reason": self.failure["reason"],
                    "phase": self._phase,
                }
            )
            raise

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "simulation_time": self.time * 0.1,
            "physical_time": self.physics_steps * self.physics_solver.config.dt,
            "physics_steps": self.physics_steps,
            "seed": self.seed,
            "drive_off": self.drive_off,
            "physics_diagnostics": dict(self._interval_diagnostics),
            "cells": [
                {
                    "id": c.id,
                    "agent_id": c.agent_id,
                    "birth_order": c.birth_order,
                    "position": c.position.tolist(),
                    "velocity": c.velocity.tolist(),
                    "development_position": list(c.development_position),
                    "state": c.state.tolist(),
                    "profile": c.profile,
                    "energy": float(c.energy),
                    "age": c.age,
                    "raw_output": (
                        None if c.raw_output is None else c.raw_output.tolist()
                    ),
                    "requested_contraction": float(
                        c.output_slots.get("contraction", 0)
                    ),
                    "applied_contraction": self._applied_commands.get(c.id, 0.0),
                }
                for c in sorted(self.cells, key=lambda c: c.birth_order)
            ],
            "bonds": [
                [b.first, b.second]
                for b in sorted(
                    self.physics_solver.bonds, key=lambda b: (b.first, b.second)
                )
            ],
        }
