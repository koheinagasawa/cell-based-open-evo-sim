import numpy as np
import pytest

import tests.utils.test_utils as tu
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World

# ---------------------------
# Shared fixtures
# ---------------------------


@pytest.fixture(scope="session")
def interpreter4():
    """Default interpreter: 4-d state + 2-d move (slices 0..3, 4..5)."""
    return SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(4, 6),
        }
    )


@pytest.fixture(scope="session")
def interpreter4_skip4():
    """Alternative interpreter that uses indices 5..6 for 'move'."""
    return SlotBasedInterpreter(
        {
            "state": slice(0, 4),
            "move": slice(5, 7),
        }
    )


@pytest.fixture(scope="session")
def positions_line():
    """Three positions on a 2D line used by many tests."""
    return [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]


@pytest.fixture
def world_line_factory(interpreter4, positions_line):
    """Factory that builds a World with three cells on a line.

    Parameters:
        seed (int): master seed for the world RNG
        order (str): 'normal' or 'reversed' cell ordering
        genome_builder (callable|object): callable taking (idx)->genome; or a single genome instance
        state_size (int): state vector size for each cell

    Returns:
        World
    """

    def make(seed=0, order="normal", genome_builder=None, state_size=4):
        # Build genomes for each cell
        if callable(genome_builder):
            genomes = [genome_builder(i) for i in range(3)]
        elif genome_builder is not None:
            genomes = [genome_builder] * 3
        else:
            # Minimal dummy genome that outputs zeros for 4 state + 2 move
            class _ZeroG:
                def activate(self, inputs):
                    return [0.0] * (state_size + 2)

            genomes = [_ZeroG(), _ZeroG(), _ZeroG()]

        cells = [
            Cell(
                position=positions_line[i].tolist(),
                genome=genomes[i],
                state_size=state_size,
                interpreter=interpreter4,
            )
            for i in range(3)
        ]

        if order == "reversed":
            cells = list(reversed(cells))

        return World(cells, seed=seed)

    return make


@pytest.fixture
def interpreter_factory():
    """Create a SlotBasedInterpreter for arbitrary (state_size, action_size)."""

    def make(state_size=4, action_size=2, move_start=None):
        # Default: move starts right after state
        start = state_size if move_start is None else int(move_start)
        return SlotBasedInterpreter(
            {
                "state": slice(0, state_size),
                "move": slice(start, start + action_size),
            }
        )

    return make


@pytest.fixture
def run_env_factory():
    """Factory wrapping tests.utils.test_utils.prepare_run().
    Returns a function that takes a config_dict and returns (run_config, recorder).
    """

    def make(config_dict):
        return tu.prepare_run(config_dict)

    return make
