import numpy as np
import uuid

class Cell:
    def __init__(self, position, genome, id=None, state_size=4):
        self.id = id or str(uuid.uuid4())
        self.position = np.array(position, dtype=float)
        self.genome = genome
        self.state = np.zeros(state_size)
        self.age = 0

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
        pad_count = max_neighbors - len(sorted_neighbors)
        input_vec += [0.0] * (pad_count * (3 + len(self.state)))  # 3 = rel_pos

        return np.array(input_vec)


    def act(self, inputs):
        outputs = self.genome.activate(inputs)
        self.state = np.array(outputs[:len(self.state)])

    def step(self, neighbors):
        inputs = self.sense(neighbors)
        self.act(inputs)
        self.age += 1
