import hashlib
import struct
from typing import Any, Callable, Dict, Protocol

import numpy as np


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


class World:
    supported_actions = ["move", "bud"]

    def __init__(
        self,
        cells,
        *,
        seed: int | None = None,
        actions: Dict[str, Callable] | None = None,
        energy_policy: EnergyPolicy,
        reproduction_policy: BudPolicy,
        lifecycle_policy: LifecyclePolicy | None = None,
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

        # Policies (thin, swappable)
        self.energy_policy = energy_policy
        self.reproduction_policy = reproduction_policy
        # Use provided lifecycle policy or a no-op fallback to keep behavior unchanged.
        self.lifecycle_policy = lifecycle_policy or _NoDeath()

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
        """
        Two-phase update:
        Phase 1: Every cell senses and decides (no world mutation).
        Phase 2: Apply all actions at once (order-invariant).
        """
        # -------- Phase 1: decide (no mutation to world state, only for cells allowed to act by lifecycle) --------
        intents = []
        for cell in self.cells:
            if self.lifecycle_policy.can_act(cell):
                neighbors = self.get_neighbors(cell)
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

        # --------  Phase 2: apply actions (order-stable, using a spawn buffer) --------
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

        #  --------  Phase 3: per-step maintenance on existing cells only --------
        for cell in self.cells:
            drain = float(self.energy_policy.per_step(cell))
            if drain > 0.0:
                cell.energy = max(0.0, min(cell.energy_max, float(cell.energy) - drain))

        # --------  Phase 3.5: remove cells according to lifecycle (after maintenance) --------
        if self.cells:
            survivors = []
            for cell in self.cells:
                if not self.lifecycle_policy.should_remove(cell):
                    survivors.append(cell)
            self.cells = survivors

        # --------  Phase 4: attach newborns (they do NOT pay maintenance this frame) --------
        if self._spawn_buffer:
            for newborn in self._spawn_buffer:
                self.add_cell(newborn)
            self._spawn_buffer.clear()

        self.time += 1

    def apply_move(self, cell, delta):
        # Make 'delta' match the dimensionality of the cell position.
        d = _align_vec(delta, int(cell.position.shape[0]))
        cell.position += d

    def apply_bud(self, cell, value):
        """Delegate budding to the injected reproduction policy."""
        self.reproduction_policy.apply(self, cell, value, self._spawn_buffer.append)

    def noop(self, cell, value):
        # No-op action handler
        pass
