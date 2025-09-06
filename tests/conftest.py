import os
import pathlib
import re
import time
import uuid
from typing import Any, Callable, Dict, Optional, Sequence

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
def session_run_dir():
    """Create a single output directory per pytest session and export it via env var.
    Example: outputs/pytest_YYYYmmdd_HHMMSS_<8hex>/
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    root = pathlib.Path("outputs")
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / f"pytest_{stamp}_{short}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Make it available to any code via environment variable (fallback path)
    os.environ["ALIFE_OUTPUT_SESSION_DIR"] = str(session_dir)
    return str(session_dir)


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
def world_factory() -> Callable[..., World]:
    """Return a factory function that builds a World with optional overrides.

    Usage:
        w = world_factory(
            [cell],
            seed=123,
            energy_policy=DummyEnergyPolicy(0.0),
            reproduction_policy=DummyBudPolicy(),
        )
    """

    def _factory(
        cells: Sequence,
        *,
        seed: int = 0,
        actions: Optional[Dict[str, Callable]] = None,
        message_router=None,
        energy_policy: Any = None,
        reproduction_policy: Any = None,
        lifecycle_policy: Any = None,
        use_neighbors: bool = True,
    ) -> World:
        return World(
            cells,
            seed=seed,
            actions=actions or {},
            message_router=message_router,
            energy_policy=energy_policy or tu.DummyEnergyPolicy(),
            reproduction_policy=reproduction_policy or tu.DummyBudPolicy(),
            lifecycle_policy=lifecycle_policy,  # may be None (World falls back to _NoDeath)
            use_neighbors=use_neighbors,
        )

    return _factory


@pytest.fixture
def world_line_factory(world_factory, interpreter4, positions_line):
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

        return world_factory(cells, seed=seed)

    return make


@pytest.fixture
def world_random_factory(world_factory: Callable[..., World]):
    """Build a world with N cells placed uniformly at random in a box."""

    def _make(
        *,
        n: int = 80,
        box=(-5.0, 5.0, -5.0, 5.0),  # (xmin, xmax, ymin, ymax)
        seed: int = 123,
        state_size: int = 4,
        genome_builder: Callable[[int], object],
        actions: Optional[Dict[str, Callable]] = None,
        **world_kwargs,
    ) -> World:
        rng = np.random.default_rng(seed)
        xs = rng.uniform(box[0], box[1], size=n)
        ys = rng.uniform(box[2], box[3], size=n)

        S = state_size
        interp = SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + 2)})

        cells: Sequence[Cell] = [
            Cell(
                position=[float(xs[i]), float(ys[i])],
                genome=genome_builder(i),
                state_size=S,
                interpreter=interp,
                energy_init=1.0,
                energy_max=1.0,
            )
            for i in range(n)
        ]
        return world_factory(
            cells,
            seed=seed,
            actions=actions,
            **world_kwargs,  # pass policies here
        )

    return _make


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
def run_env_factory(session_run_dir, request):
    """Factory wrapping prepare_run(). Names each experiment folder after the test function.

    Example: tests/test_genome_interactions.py::test_multiple_genomes_interaction
             -> exp folder 'test_multiple_genomes_interaction'
             Parametrized cases append the param id.
             Multiple runs within the same test get _02, _03... suffixes.
    """
    counter = {"n": 0}

    def _sanitize(name: str) -> str:
        # Include param id if present (pytest parametrization)
        callspec = getattr(request.node, "callspec", None)
        if callspec and getattr(callspec, "id", None):
            name = f"{name}__{callspec.id}"
        # Filesystem-safe: keep letters, numbers, '_', '-', '.'
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("._-")
        return name[:80]  # keep short-ish (Windows-friendly)

    def make(config_dict, exp_name: str | None = None):
        base = exp_name or _sanitize(request.node.name)
        counter["n"] += 1
        # If the same test calls prepare_run() multiple times, suffix with an index
        name = base if counter["n"] == 1 else f"{base}_{counter['n']:02d}"
        try:
            return tu.prepare_run(
                config_dict, session_dir=session_run_dir, exp_name=name
            )
        except TypeError:
            # Old signature fallback (shouldn't happen after this patch)
            os.environ["ALIFE_OUTPUT_SESSION_DIR"] = session_run_dir
            return tu.prepare_run(config_dict)

    return make
