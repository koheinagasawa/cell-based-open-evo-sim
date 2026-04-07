"""Performance gate: 100 cells x 200 steps with physics must complete within budget.

Roadmap Phase 1 requirement:
  "100 particles x 200 steps must complete within a few hundred ms to a few seconds. If the threshold is exceeded, first optimize neighbor and input construction."
"""
import time

import numpy as np
import pytest

from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.physics.solver import PhysicsSolver
from simulation.policies import ConstantMaintenance


class _StaticGenome:
    def __init__(self, size):
        self._size = size

    def activate(self, inputs):
        return np.zeros(self._size)


def _build_100_cell_world():
    """Create 100 cells in a 10x10 grid with nearest-neighbor spring bonds."""
    interp = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})
    genome = _StaticGenome(6)

    cells = []
    id_grid = {}
    spacing = 1.0
    idx = 0
    for row in range(10):
        for col in range(10):
            pid = f"c{idx}"
            pos = [col * spacing, row * spacing]
            c = Cell(
                pos, genome, id=pid,
                interpreter=interp, state_size=4,
                max_neighbors=0, radius=0.4,
            )
            cells.append(c)
            id_grid[(row, col)] = pid
            idx += 1

    # Connect each cell to its immediate grid neighbors (up/down/left/right)
    for row in range(10):
        for col in range(10):
            pid = id_grid[(row, col)]
            cell = next(c for c in cells if c.id == pid)
            conns = {}
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    conns[id_grid[(nr, nc)]] = 1.0
            cell.set_connections(conns)

    return cells


class TestPerformanceGate:
    """Phase 1 performance gate: 100 cells × 200 steps."""

    def test_100_cells_200_steps_within_budget(self, world_factory):
        """Must complete within 10 seconds (generous budget for CI)."""
        cells = _build_100_cell_world()
        solver = PhysicsSolver(dt=0.05, repulsion_stiffness=2.0, spring_stiffness=2.0)
        w = world_factory(
            cells,
            physics_solver=solver,
            energy_policy=ConstantMaintenance(0.0),
        )

        t0 = time.perf_counter()
        for _ in range(200):
            w.step()
        elapsed = time.perf_counter() - t0

        print(f"\n  100 cells × 200 steps: {elapsed:.3f}s")
        # Gate: must be under 10s (roadmap requires "a few seconds")
        assert elapsed < 10.0, f"Performance gate FAILED: {elapsed:.2f}s > 10s"

    def test_100_cells_200_steps_physics_breakdown(self, world_factory):
        """Profile where time is spent: physics vs rest."""
        cells = _build_100_cell_world()
        solver = PhysicsSolver(dt=0.05, repulsion_stiffness=2.0, spring_stiffness=2.0)
        w = world_factory(
            cells,
            physics_solver=solver,
            energy_policy=ConstantMaintenance(0.0),
        )

        for _ in range(200):
            w.step()

        stats = w.perf_stats
        physics_ms = stats.get("time_phase3_5_physics", 0.0) * 1000
        decide_ms = stats.get("time_phase1_decide", 0.0) * 1000
        commit_ms = stats.get("time_phase2_commit_state", 0.0) * 1000
        actions_ms = stats.get("time_phase3_apply_actions", 0.0) * 1000
        maint_ms = stats.get("time_phase4_maintenance", 0.0) * 1000
        msg_ms = stats.get("time_phase5_connected_messaging", 0.0) * 1000

        print(f"\n  Last-step timing (ms):")
        print(f"    Phase 1 decide:    {decide_ms:.2f}")
        print(f"    Phase 2 commit:    {commit_ms:.2f}")
        print(f"    Phase 3 actions:   {actions_ms:.2f}")
        print(f"    Phase 3.5 physics: {physics_ms:.2f}")
        print(f"    Phase 4 maint:     {maint_ms:.2f}")
        print(f"    Phase 5 messaging: {msg_ms:.2f}")
