import uuid

import numpy as np


class Cell:
    def __init__(
        self,
        position,
        genome,
        id=None,
        state_size=4,
        interpreter=None,
        time_encoding_fn=None,
        max_neighbors=4,
        neighbor_aggregation: str | None = None,  # {None,'mean','max'}
        include_neighbor_mask=True,
        include_num_neighbors=True,
        energy_init: float = 1.0,
        energy_max: float = 1.0,
        **kwargs
    ):
        self.id = id or str(uuid.uuid4())
        self.position = np.array(position, dtype=float)
        self.genome = genome  # Neural network shared by multiple cells
        self.interpreter = (
            interpreter  # Interpreter used to map the genome's output to local slots
        )
        self.state = np.zeros(
            state_size, dtype=float
        )  # Abstract internal states of the cell
        self.state_size = state_size
        self.time_encoding_fn = time_encoding_fn  # Optional time input encoder

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

        return np.asarray(input_vec, dtype=float)

    def act(self, inputs):
        """
        Generate outputs using the genome network and provided inputs.
        The outputs are interpreted by the interpreter and used to update Cell's internal states.
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
            slots = self.interpreter.interpret(self.raw_output)
            self.output_slots = slots
            if "state" in slots:
                self.state = np.array(slots["state"])

    def update_state(self, new_state):
        """Default behavior: overwrite internal state."""
        self.state = np.asarray(new_state, dtype=float)

    def step(self, neighbors):
        inputs = self.sense(neighbors)
        self.act(inputs)
        self.age += 1
