from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from simulation.cell import Cell


@dataclass(frozen=True)
class ConstantMaintenance:
    """Global per-step basal metabolism.
    Set maintenance=0.0 to start with no passive drain.
    """

    maintenance: float = 0.0

    def per_step(self, cell) -> float:
        # Could be made cell-dependent later if needed.
        return float(self.maintenance)


@dataclass(frozen=True)
class BaseBudding:
    """
    Base class for budding policies.
    Handles the common physics and economics of budding:
    - Gate threshold check
    - Energy check & cost deduction
    - Offset calculation
    - Basic kwargs preparation
    """

    threshold: float = 0.6
    cost: float = 0.5
    init_energy: float = 0.4
    offset_sigma: float = 0.2
    init_child: Optional[Callable[["Cell", "Cell", "World"], None]] = None

    def apply(self, world, parent, value, spawn_fn):
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        D = int(parent.position.shape[0])

        # --- 1. Parse Input Vector ---
        # Format: [gate, offset_0, ..., offset_D, ...extra_data...]
        if arr.size == 1:
            gate = float(arr[0])
            rng = getattr(parent, "rng", None)
            sigma = float(self.offset_sigma)
            offset = (
                rng.normal(0.0, sigma, size=D)
                if rng is not None
                else np.zeros(D, dtype=float)
            )
            extra_data = np.array([])
        elif arr.size >= 1 + D:
            gate = float(arr[0])
            offset = arr[1 : 1 + D]
            extra_data = arr[1 + D :]
        else:
            # Not enough data
            return

        # --- 2. Check Conditions ---
        if gate <= 0.5:
            return
        if float(parent.energy) < float(self.threshold):
            return

        # --- 3. Execute Budding (Physics & Economics) ---
        # Pay cost
        parent.energy = max(0.0, float(parent.energy) - float(self.cost))

        # Calculate new position
        new_pos = (parent.position + offset).tolist()

        # Prepare common kwargs for Cell creation
        cell_kwargs = dict(
            state_size=parent.state_size,
            energy_init=float(self.init_energy),
            energy_max=float(getattr(parent, "energy_max", 1.0)),
            recv_layout=getattr(parent, "recv_layout", {}),
            field_layout=getattr(parent, "field_layout", {}),
            max_neighbors=getattr(parent, "max_neighbors", 0),
            neighbor_aggregation=getattr(parent, "neighbor_aggregation", None),
            include_neighbor_mask=getattr(parent, "include_neighbor_mask", True),
            include_num_neighbors=getattr(parent, "include_num_neighbors", True),
            profile=parent.profile,  # Default: inherit
        )

        # --- 4. Delegate Specific Creation Logic ---
        child = self._create_child(world, parent, new_pos, cell_kwargs, extra_data)

        if child:
            spawn_fn(child, parent)

        # Call optional birth hook (after child exists, before step ends)
        if self.init_child is not None:
            self.init_child(child, parent, world)

    def _create_child(
        self, world, parent, position, cell_kwargs, extra_data
    ) -> Optional[Cell]:
        """
        Abstract method to create the specific type of child cell.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleBudding(BaseBudding):
    """
    Standard budding: Simply creates a new Cell copying parent's genome/interpreter.
    Does not handle Agent lookup.
    """

    def _create_child(self, world, parent, position, cell_kwargs, extra_data):
        return Cell(
            position=position,
            genome=parent.genome,
            interpreter=parent.interpreter,
            **cell_kwargs,
        )


@dataclass(frozen=True)
class AgentBudding(BaseBudding):
    """
    Agent-aware budding: Ensures the child belongs to the same Agent.
    """

    def _create_child(self, world, parent, position, cell_kwargs, extra_data):
        # 1. Try to find the Agent instance in the World
        agent_id = getattr(parent, "agent_id", None)
        agent = None
        if hasattr(world, "get_agent") and agent_id:
            agent = world.get_agent(agent_id)

        if agent:
            # Case A: Agent exists -> Delegate to Agent
            return agent.spawn_cell(position=position, **cell_kwargs)
        else:
            # Case B: Fallback -> Create manually and attach ID
            child = Cell(
                position=position,
                genome=parent.genome,
                interpreter=parent.interpreter,
                **cell_kwargs,
            )
            if agent_id:
                child.agent_id = agent_id
            return child


@dataclass(frozen=True)
class AgentProfileBudding(AgentBudding):
    """
    Agent + Profile budding:
    Determines the child's profile from extra genome output,
    then delegates to AgentBudding for creation.
    """

    inherit_profile: bool = True
    profile_map: Optional[Dict[int, str]] = None

    def _create_child(self, world, parent, position, cell_kwargs, extra_data):
        # Determine Profile
        new_profile = cell_kwargs.get("profile", "default")

        if not self.inherit_profile and self.profile_map:
            # extra_data corresponds to the part of genome output after offset
            if extra_data.size > 0:
                # The first element of extra_data is the profile control
                p_idx = int(round(extra_data[0]))
                new_profile = self.profile_map.get(p_idx, new_profile)

        # Update kwargs with the new profile
        cell_kwargs["profile"] = new_profile

        # Delegate to AgentBudding to handle the actual instantiation & Agent lookup
        return super()._create_child(world, parent, position, cell_kwargs, extra_data)


class ParentChildLinkWrapper:
    """
    BudPolicy wrapper that automatically links parent and child upon birth.

    It composes an existing BudPolicy (e.g., SimpleBudding) and intercepts
    the spawn callback to install message connections:
      parent --(weight)--> child
      and optionally child --(weight)--> parent (bidirectional=True)

    This does NOT alter any other behavior (energy accounting, thresholds, etc.).
    """

    def __init__(
        self, base_policy, *, weight: float = 1.0, bidirectional: bool = False
    ):
        self.base_policy = base_policy
        self.weight = float(weight)
        self.bidirectional = bool(bidirectional)

    def apply(self, world, parent, value, spawn_cb):
        """Delegate to base policy while injecting the link after child creation."""

        def _spawn(child, parent, metadata=None):
            # Create parent -> child link
            try:
                parent.set_connections([(child.id, self.weight)])
            except Exception as e:
                raise RuntimeError(f"Failed to set parent->child connection: {e}")

            # Optionally create child -> parent link
            if self.bidirectional:
                try:
                    child.set_connections([(parent.id, self.weight)])
                except Exception as e:
                    raise RuntimeError(f"Failed to set child->parent connection: {e}")

            # Forward to world spawn buffer
            meta = dict(metadata or {})
            meta.setdefault("link_weight", self.weight)
            if self.bidirectional:
                meta.setdefault("bidirectional", True)
            spawn_cb(child, parent, meta)

        # Call base policy with our intercepted spawn
        return self.base_policy.apply(world, parent, value, _spawn)


@dataclass(frozen=True)
class NoDeath:
    """Lifecycle: never blocks acting; never removes."""

    def can_act(self, cell) -> bool:
        return True

    def should_remove(self, cell) -> bool:
        return False


@dataclass(frozen=True)
class KillAtZero:
    """Lifecycle: blocks acting at energy<=0 and removes at energy<=0."""

    def can_act(self, cell) -> bool:
        return cell.energy > 0.0

    def should_remove(self, cell) -> bool:
        return cell.energy <= 0.0
