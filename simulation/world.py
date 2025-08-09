import numpy as np


class World:
    supported_actions = ["move"]

    def __init__(self, cells):
        """
        Initialize the world with a list of Cell objects.
        """
        self.cells = cells
        self.time = 0

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
        # -------- Phase 1: decide (no mutation to world state) --------
        intents = []
        for cell in self.cells:
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

        # -------- Phase 2: apply (order-invariant) --------
        for cell, slots in intents:
            for action_key in self.supported_actions:
                value = slots.get(action_key)
                if value is not None:
                    handler = getattr(self, f"apply_{action_key}", self.noop)
                    handler(cell, value)

        self.time += 1

    def apply_move(self, cell, delta):
        # Apply a displacement vector to the cell's position
        cell.position += np.array(delta)

    def noop(self, cell, value):
        # No-op action handler
        pass
