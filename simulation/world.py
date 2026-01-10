import hashlib
import struct
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np

try:
    # Optional import; World can run without fields.
    from simulation.fields import FieldRouter
except Exception:  # pragma: no cover
    FieldRouter = Any  # type: ignore


class EnergyPolicy(Protocol):
    """Interface: returns per-step basal energy drain for a given cell."""

    def per_step(self, cell) -> float: ...


class BudPolicy(Protocol):
    """Interface: performs budding according to policy and spawns a newborn."""

    def apply(
        self, world, parent, value, spawn_fn: Callable[[object], None]
    ) -> None: ...


class LifecyclePolicy(Protocol):
    """Interface: lifecycle gating/removal policy."""

    def can_act(self, cell) -> bool: ...
    def should_remove(self, cell) -> bool: ...


class _NoDeath:
    """Fallback lifecycle policy used when none is provided."""

    def can_act(self, cell) -> bool:
        return True

    def should_remove(self, cell) -> bool:
        return False


def _align_vec(vec, target_dim: int) -> np.ndarray:
    """Align a 1-D vector to target_dim by right-padding zeros or truncation."""
    a = np.asarray(vec, dtype=float).ravel()
    n = a.shape[0]
    if n == target_dim:
        return a
    if n < target_dim:
        return np.pad(a, (0, target_dim - n), mode="constant")
    return a[:target_dim]


BirthCallback = Callable[["World", Dict[str, Any]], None]


class World:
    supported_actions = ["move", "bud"]

    def __init__(
        self,
        cells,
        *,
        seed: int | None = None,
        actions: Dict[str, Callable] | None = None,
        message_router=None,
        energy_policy: EnergyPolicy,
        reproduction_policy: BudPolicy,
        lifecycle_policy: LifecyclePolicy | None = None,
        use_neighbors: bool = True,
        field_router: FieldRouter | None = None,
        use_fields: bool = False,
        birth_callback: Optional[BirthCallback] = None,
    ):
        """
        :param seed: master seed for reproducible experiments. If None, use 0.

        World depends on abstract policies injected via the constructor.
        Do not import concrete policy classes here; keep this layer thin.
        """
        self.cells = cells
        self.time = 0
        self.seed = int(seed) if seed is not None else 0

        # Attach RNG to cells if your implementation supports it
        if hasattr(self, "_attach_rng_to_cells"):
            # Attach per-cell RNGs deterministically (order-invariant).
            self._attach_rng_to_cells(self.cells)

        # Order-stable spawn buffer; newborns are attached after maintenance
        self._spawn_buffer = []

        # Optional: allow action handler injection; otherwise use methods on this class
        self.actions = actions or {}

        self.message_router = message_router

        # Feature gates
        self.use_neighbors = bool(use_neighbors)
        self.field_router = field_router
        self.use_fields = bool(use_fields)

        # If False, World will not perform neighbor search at all.
        # Per-cell max_neighbors==0 also guarantees empty neighbors.
        self.use_neighbors = bool(use_neighbors)

        # Policies (thin, swappable)
        self.energy_policy = energy_policy
        self.reproduction_policy = reproduction_policy
        self._birth_callback: Optional[BirthCallback] = birth_callback

        # Use provided lifecycle policy or a no-op fallback to keep behavior unchanged.
        self.lifecycle_policy = lifecycle_policy or _NoDeath()

        # --- Performance Metrics ---
        self.perf_stats: Dict[str, float] = defaultdict(float)

    # --- RNG wiring ----------------------------------------------------------
    @staticmethod
    def _stable_spawn_key_from_cell(cell, quant: float = 1e6):
        """
        Build a deterministic spawn_key from immutable attributes (initial position, state size).
        Do NOT use Python's hash() due to per-process randomization.
        """
        pos = np.asarray(cell.position, dtype=float)
        coords = [int(round(float(x) * quant)) for x in pos.tolist()]
        state_sz = int(getattr(cell, "state_size", 0))
        payload = ("pos:" + ",".join(map(str, coords)) + f"|S:{state_sz}").encode(
            "utf-8"
        )
        digest = hashlib.sha256(payload).digest()
        # Take first 5x32-bit words as spawn_key
        k = struct.unpack("!5I", digest[:20])
        return k

    def _attach_rng_to_cells(self, cells):
        for cell in cells:
            spawn_key = self._stable_spawn_key_from_cell(cell)
            ss = np.random.SeedSequence(self.seed, spawn_key=spawn_key)
            cell.rng = np.random.default_rng(ss)

    def add_cell(self, cell):
        self._attach_rng_to_cells([cell])
        self.cells.append(cell)

    def get_neighbors(self, target_cell, radius=10.0):
        """
        Return neighbors sorted deterministically:
        1) by squared distance (ascending)
        2) by position tuple (lexicographic) to break ties
        3) by cell.id as a final tiebreaker (should be rare)
        """

        # Fast path: globally disabled.
        if not self.use_neighbors:
            return []

        # Fast path: the cell itself never uses neighbor inputs.
        if getattr(target_cell, "max_neighbors", 0) <= 0:
            return []

        # metric: neighbor search count
        self.perf_stats["neighbor_search_count"] += 1.0

        entries = []
        r2 = float(radius * radius)

        for cell in self.cells:
            if cell is target_cell:
                continue
            vec = cell.position - target_cell.position
            dist2 = float(vec @ vec)
            if dist2 <= r2:
                # Use a tuple of native Python types for a total ordering
                pos_key = tuple(map(float, cell.position.tolist()))
                entries.append((dist2, pos_key, getattr(cell, "id", ""), cell))

        # Deterministic order, independent of self.cells iteration order
        entries.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in entries]

    def step(self):
        # --- Two-phase update schedule (design contract) ---------------------
        # 1) (Optionally) Build neighbor snapshots for ALL cells (read-only, previous-frame state)
        # 2) Sense+Act for ALL cells (produce outputs; DO NOT mutate cell.state here)
        # 3) Commit: apply cell.next_state -> cell.state for ALL cells (synchronous state update)
        # 4) Reproduction, maintenance, deaths, time++ (project-specific policies)
        # 5) Connected messaging + Field routing (two-phase; affect NEXT frame)

        # Reset per-step performance stats
        self.perf_stats.clear()

        # -------- Phase 1: decide (no mutation to world state, only for cells allowed to act by lifecycle) --------
        t_phase1_start = time.perf_counter()
        intents = []
        for cell in self.cells:
            if self.lifecycle_policy.can_act(cell):
                # Skip neighbor search when globally disabled or max_neighbors==0.
                if not self.use_neighbors or getattr(cell, "max_neighbors", 0) <= 0:
                    neighbors = []
                else:
                    # Measure neighbor search time
                    t0 = time.perf_counter()
                    neighbors = self.get_neighbors(cell)
                    self.perf_stats["time_neighbor_search"] += time.perf_counter() - t0

                # Populate field inputs for this frame (read-only snapshot)
                if self.use_fields and self.field_router is not None:
                    try:
                        self.field_router.sample_into_cell(cell)
                    except Exception:
                        # Be robust: if router misconfigured, zero out field inputs.
                        cell.field_inputs = {}
                else:
                    # Ensure no stale field inputs from prior frames
                    if hasattr(cell, "field_inputs"):
                        cell.field_inputs = {}

                cell.step(
                    neighbors
                )  # computes raw_output/output_slots; may update internal state

                # Snapshot interpreted outputs to decouple from later mutations
                snap = {}
                for k, v in (cell.output_slots or {}).items():
                    # Convert lists/tuples to ndarray; copy ndarray to avoid aliasing
                    if isinstance(v, np.ndarray):
                        snap[k] = v.copy()
                    elif isinstance(v, (list, tuple)):
                        snap[k] = np.array(v, dtype=float)
                    else:
                        snap[k] = v
                intents.append((cell, snap))
            else:
                # Skip acting; treat as producing no outputs this frame.
                intents.append((cell, {}))

        self.perf_stats["time_phase1_decide"] = time.perf_counter() - t_phase1_start

        # --------  Phase 2: commit state update after all cells have acted  --------
        t_phase2_start = time.perf_counter()

        for cell in self.cells:
            ns = getattr(cell, "next_state", None)
            if ns is not None:
                cell.update_state(ns)
                cell.next_state = None

        self.perf_stats["time_phase2_commit_state"] = (
            time.perf_counter() - t_phase2_start
        )

        # --------  Phase 3: apply actions (order-stable, using a spawn buffer) --------
        t_phase3_start = time.perf_counter()

        self._spawn_buffer = []
        for cell, slots in intents:
            for action_key in self.supported_actions:
                value = slots.get(action_key)
                if value is None:
                    continue
                handler = self.actions.get(action_key) or getattr(
                    self, f"apply_{action_key}", self.noop
                )
                handler(cell, value)

        self.perf_stats["time_phase3_apply_actions"] = (
            time.perf_counter() - t_phase3_start
        )

        t_phase4_start = time.perf_counter()

        #  --------  Phase 4.1: per-step maintenance on existing cells only --------
        for cell in self.cells:
            drain = float(self.energy_policy.per_step(cell))
            if drain > 0.0:
                cell.energy = max(0.0, min(cell.energy_max, float(cell.energy) - drain))

        # --------  Phase 4.2: remove cells according to lifecycle (after maintenance) --------
        if self.cells:
            survivors = []
            for cell in self.cells:
                if not self.lifecycle_policy.should_remove(cell):
                    survivors.append(cell)
            self.cells = survivors

        # --------  Phase 4.3: attach newborns (they do NOT pay maintenance this frame) --------
        if self._spawn_buffer:
            for info in self._spawn_buffer:
                child = info["child"]
                self.add_cell(child)
                if self._birth_callback is not None:
                    self._birth_callback(self, info)
            self._spawn_buffer.clear()

        self.perf_stats["time_phase4_maintenance"] = (
            time.perf_counter() - t_phase4_start
        )

        # --------  Phase 5: Connected messaging & Field routing (NEXT frame) --------
        t_phase5_start = time.perf_counter()

        # Fields: decay old sources, then collect current-frame deposits
        if self.use_fields and self.field_router is not None:
            self.field_router.apply_decay()
            self.field_router.collect_from_cells(self.cells)

        # Connected messaging: route 'emit:*' to 'recv:*' staged inboxes
        if self.message_router is not None:
            self.message_router.route_and_stage(self.cells)
            # Commit staged inboxes for next frame
            self.message_router.swap_inboxes(self.cells)

        self.perf_stats["time_phase5_connected_messaging"] = (
            time.perf_counter() - t_phase5_start
        )

        self.time += 1

    def apply_move(self, cell, delta):
        # Make 'delta' match the dimensionality of the cell position.
        d = _align_vec(delta, int(cell.position.shape[0]))
        cell.position += d

    def apply_bud(self, parent, value):
        """Delegate budding to the injected reproduction policy."""

        def _spawn(child, parent, metadata=None):
            self._spawn_buffer.append(
                {
                    "child": child,
                    "parent": parent,
                    "metadata": metadata or {},
                }
            )

        self.reproduction_policy.apply(self, parent, value, _spawn)

    def noop(self, cell, value):
        # No-op action handler
        pass
