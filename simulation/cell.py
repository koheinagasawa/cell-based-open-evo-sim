import uuid
from typing import Optional

import numpy as np


class Cell:
    def __init__(
        self,
        position,
        genome,
        id=None,
        agent_id: Optional[str] = None,
        state_size=4,
        interpreter=None,
        profile=None,
        time_encoding_fn=None,
        max_neighbors=4,
        neighbor_aggregation: str | None = None,  # {None,'mean','max'}
        include_neighbor_mask=True,
        include_num_neighbors=True,
        # Legacy spatial-neighbor features (to be phased out later):
        # - max_neighbors:         upper bound for per-neighbor slots (0 disables)
        # - include_neighbor_mask: append K-length mask for padded neighbors
        # - include_num_neighbors: append neighbor count as a scalar tail
        # - neighbor_aggregation:  {'mean','max'} summary BEFORE mask/count tails
        # These are ignored once the input is built purely from connected messaging
        # (i.e., recv:* aggregates). New experiments should prefer connected messaging.
        energy_init: float = 1.0,
        energy_max: float = 1.0,
        **kwargs,
    ):
        self.id = id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.position = np.array(position, dtype=float)
        self.genome = genome  # Neural network shared by multiple cells
        self.interpreter = (
            interpreter  # Interpreter used to map the genome's output to local slots
        )
        self.profile = profile
        self.state = np.zeros(
            state_size, dtype=float
        )  # Abstract internal states of the cell
        self.next_state = None  # defer state commit for two-phase update
        self.state_size = state_size
        self.time_encoding_fn = time_encoding_fn  # Optional time input encoder

        # --- Static directed connections owned by this cell -----------------
        # Store outgoing edges as {dst_id: weight}. Keep IDs (not object refs)
        # to avoid stale references; resolve via an id->Cell registry when needed.
        self.conn_out: dict[str, float] = {}
        # --- Connected messaging ----------------------------------------
        # Two-phase receive buffers (current frame inbox, and next-frame staging)
        self.inbox: dict[str, np.ndarray] = {}  # key: 'recv:<name>' -> 1-D float array
        self._next_inbox: dict[str, np.ndarray] = {}
        # Expected layout for recv slots; ensures fixed input size.
        # Example: {'recv:a': 2, 'recv:b': 4}
        self.recv_layout: dict[str, int] = dict(kwargs.get("recv_layout", {}))
        # --- Environmental fields (declarative, optional) -------------------
        # Input declaration for environmental fields. Keys are *explicit*:
        #   'field:<name>:val'  -> dim=1
        #   'field:<name>:grad' -> dim=D (position dimensionality)
        # Values are lengths (int). Absent keys imply zero-length / not appended.
        self.field_layout: dict[str, int] = dict(kwargs.get("field_layout", {}))
        # Per-frame buffer populated by World/FieldRouter before sense().
        self.field_inputs: dict[str, np.ndarray] = {}

        self.max_neighbors = int(max_neighbors)
        self.include_neighbor_mask = bool(include_neighbor_mask)
        self.include_num_neighbors = bool(include_num_neighbors)
        self.neighbor_aggregation = neighbor_aggregation  # Aggregation mode

        self.energy_max = float(energy_max)
        self.energy = float(energy_init)

        self.age = 0
        self.raw_output = None  # Last raw output vector from genome
        self.output_slots = {}  # Interpreted output dictionary (e.g. "move", "state")

        self.rng = None  # will be injected by World

    @property
    def pos_dim(self) -> int:
        """Dimensionality of the position vector."""
        return int(self.position.shape[0])

    def sense(self, neighbors):
        """
        Build a fixed-length input vector in the following order (2D example):
          [ self_pos(2), self_state(S),
            n0_relpos(2), n0_state(S),
            n1_relpos(2), n1_state(S),
            ... up to max_neighbors, zero-padded,
            time_features(optional),
            neighbor_mask(max_neighbors, 1.0 for present else 0.0)  <-- appended at tail
            num_neighbors(1)                                        <-- appended at tail
            + connected messaging tail:  sorted(recv_layout) vectors (fixed)
            + field inputs tail:         sorted(field_layout) vectors (fixed)
          ]

        Notes:
        - Neighbor *order* must be deterministic and is provided by World.get_neighbors().
        - We use zeros for padding of absent neighbors, but downstream models should rely on
          'neighbor_mask' instead of assuming zeros mean "no data".
        """
        pos_dim = self.position.shape[0]
        S = self.state_size
        K = self.max_neighbors

        # Base: self position + self state
        input_vec = [*self.position.tolist(), *self.state.tolist()]

        # Fast bypass: when K == 0, skip all spatial-neighbor features entirely
        if K == 0:
            # Optional: time features (kept before mask/count)
            if self.time_encoding_fn is not None:
                tfeat = np.asarray(
                    self.time_encoding_fn(self.age), dtype=float
                ).tolist()
                input_vec += tfeat

            # Mask is zero-length when K==0; nothing to append.
            if self.include_num_neighbors:
                input_vec.append(0.0)

            # Connected messaging tail (deterministic, fixed)
            if self.recv_layout:
                for key in sorted(self.recv_layout.keys()):
                    dim = int(self.recv_layout[key])
                    v = np.asarray(
                        self.inbox.get(key, np.zeros(dim, dtype=float)), dtype=float
                    ).ravel()
                    out = np.zeros(dim, dtype=float)
                    n = min(dim, v.size)
                    if n > 0:
                        out[:n] = v[:n]
                    input_vec += out.tolist()

            # Field inputs tail (deterministic, fixed; zeros when not provided)
            if self.field_layout:
                for key in sorted(self.field_layout.keys()):
                    dim = int(self.field_layout[key])
                    v = np.asarray(
                        self.field_inputs.get(key, np.zeros(dim, dtype=float)),
                        dtype=float,
                    ).ravel()
                    out = np.zeros(dim, dtype=float)
                    n = min(dim, v.size)
                    if n > 0:
                        out[:n] = v[:n]
                    input_vec += out.tolist()

            return np.asarray(input_vec, dtype=float)

        # Clip neighbors to K and build blocks
        sorted_neighbors = neighbors[:K]

        # Track mask as float (1.0 present, 0.0 absent)
        mask = [0.0] * K

        # Clip neighbors to K and build blocks
        sorted_neighbors = neighbors[:K]

        # Track mask as float (1.0 present, 0.0 absent)
        mask = [0.0] * K

        # Fill existing neighbors
        rels = []  # collect relative positions for aggregation
        nstates = []  # collect neighbor states for aggregation
        for i, n in enumerate(sorted_neighbors):
            rel = (n.position - self.position).astype(float)
            # Safety: match dimensionality
            if rel.shape[0] != pos_dim:
                raise ValueError("Neighbor dimensionality mismatch.")
            input_vec += rel.tolist()

            # Neighbor state (pad/trim to S if external code ever diverges)
            n_state = np.asarray(
                getattr(n, "state", np.zeros(S, dtype=float)), dtype=float
            )
            if n_state.shape[0] != S:
                if n_state.shape[0] < S:
                    pad = np.zeros(S - n_state.shape[0], dtype=float)
                    n_state = np.concatenate([n_state, pad], axis=0)
                else:
                    n_state = n_state[:S]
            input_vec += n_state.tolist()
            rels.append(rel)
            nstates.append(n.state.astype(float))

            mask[i] = 1.0

        # Pad remaining neighbor slots with zeros
        missing = K - len(sorted_neighbors)
        if missing > 0:
            input_vec += [0.0] * missing * (pos_dim + S)

        # Optional: aggregated neighbor summary (BEFORE mask/count tails)
        if self.neighbor_aggregation is not None:
            # NOTE: Only present neighbors are aggregated.
            if len(rels) == 0:
                input_vec += [0.0] * (pos_dim + S)
            else:
                R = np.stack(rels, axis=0)  # [N, pos_dim]
                NS = np.stack(nstates, axis=0)  # [N, S]
                mode = str(self.neighbor_aggregation).lower()
                if mode == "mean":
                    # NOTE: Pure arithmetic mean; no distance weighting.
                    input_vec += R.mean(axis=0).tolist()
                    input_vec += NS.mean(axis=0).tolist()
                elif mode == "max":
                    # NOTE: Element-wise max; not rotation invariant.
                    input_vec += R.max(axis=0).tolist()
                    input_vec += NS.max(axis=0).tolist()
                else:
                    # Unknown mode -> append zeros to keep shape deterministic.
                    input_vec += [0.0] * (pos_dim + S)

        # Optional: time features (kept in the same place as before)
        if self.time_encoding_fn is not None:
            # time features should not depend on world step order; use 'age' or provided fn
            tfeat = np.asarray(self.time_encoding_fn(self.age), dtype=float).tolist()
            input_vec += tfeat

        # Append mask and neighbor count at the very tail (to avoid shifting older indices)
        if self.include_neighbor_mask:
            input_vec += mask
        if self.include_num_neighbors:
            input_vec.append(float(len(sorted_neighbors)))

        # --- Connected messaging tail (deterministic, fixed) -----------------
        # For each declared recv key (sorted by name), append a fixed-length vector.
        # If inbox is missing the key, append zeros of the declared dimension.
        if self.recv_layout:
            for key in sorted(self.recv_layout.keys()):
                dim = int(self.recv_layout[key])
                v = np.asarray(
                    self.inbox.get(key, np.zeros(dim, dtype=float)), dtype=float
                ).ravel()
                out = np.zeros(dim, dtype=float)
                # copy up to dim (truncate or zero-pad)
                n = min(dim, v.size)
                if n > 0:
                    out[:n] = v[:n]
                input_vec += out.tolist()

        # --- Field inputs tail (deterministic, fixed) ------------------------
        # For each declared field key (sorted by name), append a fixed-length vector.
        # World/FieldRouter must have populated cell.field_inputs before sense().
        if self.field_layout:
            for key in sorted(self.field_layout.keys()):
                dim = int(self.field_layout[key])
                v = np.asarray(
                    self.field_inputs.get(key, np.zeros(dim, dtype=float)),
                    dtype=float,
                ).ravel()
                out = np.zeros(dim, dtype=float)
                n = min(dim, v.size)
                if n > 0:
                    out[:n] = v[:n]
                input_vec += out.tolist()

        return np.asarray(input_vec, dtype=float)

    def act(self, inputs):
        """
        Generate outputs using the genome network and provided inputs.
        The outputs are interpreted by the interpreter (context-aware via profile) and used to update Cell's internal states.
        Pass cell-local RNG if supported.
        """
        try:
            self.raw_output = self.genome.activate(inputs, rng=self.rng)
        except TypeError:
            if hasattr(self.genome, "set_rng"):
                self.genome.set_rng(self.rng)
            elif hasattr(self.genome, "rng"):
                self.genome.rng = self.rng
            self.raw_output = self.genome.activate(inputs)

        if self.interpreter:
            # Pass self.profile as context
            slots = self.interpreter.interpret(self.raw_output, profile=self.profile)
            self.output_slots = slots
            if "state" in slots:
                # Two-phase contract: never write to self.state here.
                # Store the desired next state, which World.step() will commit
                # synchronously for all cells in the commit phase.
                self.next_state = np.array(slots["state"], dtype=float)

    def update_state(self, new_state):
        """Default behavior: overwrite internal state."""
        self.state = np.asarray(new_state, dtype=float)

    def step(self, neighbors):
        inputs = self.sense(neighbors)
        self.act(inputs)
        self.age += 1

    # ----------------------- Connections API ----------------------------
    def set_connections(self, edges) -> None:
        """
        Set outgoing connections from this cell.
        Accepts any of:
          - iterable of dst_id strings:        ["B", "C"]
          - iterable of (dst_id, weight) pairs: [("B", 0.8), ("C", 0.2)]
          - dict mapping:                       {"B": 0.8, "C": 0.2}
        Last occurrence wins on duplicate keys.
        """
        bucket: dict[str, float] = {}
        if hasattr(edges, "items"):
            for k, w in edges.items():
                bucket[str(k)] = float(w)
        else:
            for item in edges:
                if isinstance(item, tuple) and len(item) == 2:
                    k, w = item
                    bucket[str(k)] = float(w)
                else:
                    bucket[str(item)] = 1.0
        self.conn_out = bucket

    def clear_connections(self) -> None:
        """Remove all outgoing connections."""
        self.conn_out.clear()

    def connected_ids(self) -> list[str]:
        """Return sorted destination IDs by (-weight, id) for determinism."""
        return [k for k, _ in self._sorted_conn_items()]

    def connected_pairs(
        self, id_to_cell: dict[str, "Cell"]
    ) -> list[tuple["Cell", float]]:
        """
        Resolve connections against a registry (id -> Cell).
        Raises KeyError if any id is unknown.
        """
        out = []
        for dst_id, w in self._sorted_conn_items():
            try:
                out.append((id_to_cell[dst_id], w))
            except KeyError as e:
                raise KeyError(f"Unknown dst id in Cell.connections: {dst_id}") from e
        return out

    def _sorted_conn_items(self) -> list[tuple[str, float]]:
        """Internal: return [(dst_id, weight), ...] sorted by (-weight, id)."""
        return sorted(self.conn_out.items(), key=lambda kv: (-kv[1], kv[0]))
