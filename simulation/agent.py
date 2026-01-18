import uuid
from typing import Any, List, Optional

from simulation.cell import Cell
from simulation.interpreter import Interpreter


class Agent:
    """
    Evolutionary unit that aggregates multiple cells sharing the same Genome and Interpreter.
    """

    def __init__(
        self,
        genome: Any,
        interpreter: Interpreter,
        id: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.genome = genome
        self.interpreter = interpreter
        self.cells: List[Cell] = []

    def spawn_cell(self, position, **kwargs) -> Cell:
        """
        Create a new Cell belonging to this Agent.
        The cell inherits the agent's genome and interpreter.

        Args:
            position: Initial position of the cell.
            **kwargs: Additional arguments passed to Cell constructor
                      (e.g., state_size, max_neighbors).

        Returns:
            The created Cell instance.
        """
        # Ensure genome and interpreter are passed from the Agent
        cell = Cell(
            position=position,
            genome=self.genome,
            interpreter=self.interpreter,
            **kwargs,
        )

        # Tag the cell with this agent's ID for backward reference
        # (This attribute is dynamic and not strictly defined in Cell yet, but useful)
        cell.agent_id = self.id

        self.cells.append(cell)
        return cell

    def __repr__(self):
        return f"<Agent id={self.id[:6]} cells={len(self.cells)}>"
