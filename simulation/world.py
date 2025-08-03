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
        Returns a list of neighboring cells within a given radius from target_cell.
        """
        neighbors = []
        for cell in self.cells:
            if cell is target_cell:
                continue
            dist = np.linalg.norm(cell.position - target_cell.position)
            if dist <= radius:
                neighbors.append(cell)
        return neighbors

    def step(self):
        """
        Advances the simulation by one step for all cells.
        """
        for cell in self.cells:
            neighbors = self.get_neighbors(cell)
            cell.step(neighbors)
            self.apply_cell_actions(cell)
        self.time += 1

    def apply_cell_actions(self, cell):
        for action_key in self.supported_actions:
            value = cell.output_slots.get(action_key)
            if value is not None:
                handler = getattr(self, f"apply_{action_key}", self.noop)
                handler(cell, value)

    def apply_move(self, cell, delta):
        cell.position += np.array(delta)

    def noop(self, cell, value):
        pass
