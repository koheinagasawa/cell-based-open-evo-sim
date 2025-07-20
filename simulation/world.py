import numpy as np

class World:
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
        self.time += 1
