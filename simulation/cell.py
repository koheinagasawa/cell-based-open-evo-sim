import numpy as np
import uuid

class Cell:
    def __init__(self, position, genome, id=None, state_size=4, time_encoding_fn=None):
        self.id = id or str(uuid.uuid4())
        self.position = np.array(position, dtype=float)
        self.genome = genome # Neural network shared by multiple cells
        self.state = np.zeros(state_size) # Abstract internal states of the cell
        self.output_action = np.zeros(0) # will be overwritten by act()
        self.age = 0
        self.time_encoding_fn = time_encoding_fn  # Optional time input encoder

    def sense(self, neighbors, max_neighbors=4):
        """
        Returns a fixed-length input vector containing:
        - self position
        - self state
        - states of up to N neighbors (zero-padded if fewer)
        """
        input_vec = list(self.position) + list(self.state)

        # Sort neighbors by distance (optional but can stabilize learning)
        sorted_neighbors = sorted(
            neighbors, key=lambda n: np.linalg.norm(n.position - self.position)
        )[:max_neighbors]

        for neighbor in sorted_neighbors:
            rel_pos = neighbor.position - self.position  # relative position
            input_vec += list(rel_pos)
            input_vec += list(neighbor.state)

        # Pad with zeros if not enough neighbors
        position_dim = self.position.shape[0]
        pad_count = max_neighbors - len(sorted_neighbors)
        input_vec += [0.0] * (pad_count * (position_dim + len(self.state)))

        # Add time encoding, if applicable
        if self.time_encoding_fn:
            input_vec += self.time_encoding_fn(self.age)

        return np.array(input_vec)

    def act(self, inputs):
        """
        Generate outputs using the genome network and provided inputs.
        The outputs contain updated cell's internal state and also one-time actions.
        """
        outputs = self.genome.activate(inputs)
        new_state = np.array(outputs[:len(self.state)])
        self.update_state(new_state)
        self.output_action = outputs[len(self.state):]

    def update_state(self, new_state):
        """Default behavior: overwrite internal state."""
        self.state = new_state

    def step(self, neighbors):
        inputs = self.sense(neighbors)
        self.act(inputs)
        self.age += 1
