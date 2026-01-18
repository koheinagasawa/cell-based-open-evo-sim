from dataclasses import dataclass
from typing import Callable, Optional

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
class SimpleBudding:
    """Minimal budding policy with a scalar threshold and cost."""

    threshold: float = 0.6  # min energy required to attempt budding
    cost: float = 0.5  # energy paid by parent at bud time
    init_energy: float = 0.4  # newborn initial energy
    offset_sigma: float = 0.2  # jitter if no offset is provided
    init_child: Optional[Callable[["Cell", "Cell", "World"], None]] = None

    def apply(self, world, parent, value, spawn_fn):
        """Interpret 'value' and spawn a single offspring if conditions hold.

        Accepted shapes:
          - scalar: gate in (0..1); offset sampled ~ N(0, sigma)
          - D-vector: treated as offset; gate=1.0
          - [gate, *offset(D)]: explicit gate and offset
        """
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        D = int(parent.position.shape[0])

        # Parse gate and offset
        if arr.size == 1:
            gate = float(arr[0])
            rng = getattr(parent, "rng", None)
            sigma = float(self.offset_sigma)
            offset = (
                rng.normal(0.0, sigma, size=D)
                if rng is not None
                else np.zeros(D, dtype=float)
            )
        elif arr.size == D:
            gate, offset = 1.0, arr[:D]
        else:
            gate = float(arr[0])
            offset = arr[1 : 1 + D]

        # Check energy/threshold
        if gate <= 0.5:
            return
        if float(parent.energy) < float(self.threshold):
            return

        # Pay cost and spawn
        parent.energy = max(0.0, float(parent.energy) - float(self.cost))

        Baby = parent.__class__
        baby = Baby(
            position=(parent.position + offset).tolist(),
            genome=parent.genome,
            state_size=parent.state_size,
            interpreter=parent.interpreter,
            # Inherit energy cap; newborn gets init_energy
            energy_init=float(self.init_energy),
            energy_max=float(getattr(parent, "energy_max", 1.0)),
            # Inherit IO layout and neighbor settings
            recv_layout=getattr(parent, "recv_layout", {}),
            field_layout=getattr(parent, "field_layout", {}),
            max_neighbors=getattr(parent, "max_neighbors", 0),
            neighbor_aggregation=getattr(parent, "neighbor_aggregation", None),
            include_neighbor_mask=getattr(parent, "include_neighbor_mask", True),
            include_num_neighbors=getattr(parent, "include_num_neighbors", True),
        )
        # RNG will be attached by the world; newborn does not pay maintenance this frame.
        spawn_fn(baby, parent)

        # Call optional birth hook (after child exists, before step ends)
        if self.init_child is not None:
            self.init_child(baby, parent, world)


@dataclass(frozen=True)
class AgentBudding:
    """
    Budding policy that ensures the child cell belongs to the same Agent as the parent.
    It attempts to retrieve the parent's Agent instance from the World to spawn the child.
    """

    threshold: float = 0.6
    cost: float = 0.5
    init_energy: float = 0.4
    offset_sigma: float = 0.2

    def apply(self, world, parent, value, spawn_fn):
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        D = int(parent.position.shape[0])

        # Parse gate and offset (Standard logic)
        if arr.size == 1:
            gate = float(arr[0])
            rng = getattr(parent, "rng", None)
            sigma = float(self.offset_sigma)
            offset = (
                rng.normal(0.0, sigma, size=D)
                if rng is not None
                else np.zeros(D, dtype=float)
            )
        elif arr.size == D:
            gate, offset = 1.0, arr[:D]
        else:
            gate = float(arr[0])
            offset = arr[1 : 1 + D]

        # Check conditions
        if gate <= 0.5:
            return
        if float(parent.energy) < float(self.threshold):
            return

        # Pay cost
        parent.energy = max(0.0, float(parent.energy) - float(self.cost))

        # --- Create Child via Agent Lookup ---
        new_pos = (parent.position + offset).tolist()

        # Prepare basic attributes for the new cell
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
            profile=parent.profile,  # Simply copy profile
        )

        # 1. Try to find the Agent instance in the World
        agent_id = parent.agent_id
        agent = world.get_agent(agent_id)

        # This ensures the child is added to agent.cells and shares genome/interpreter
        child = agent.spawn_cell(position=new_pos, **cell_kwargs)

        # Register to World buffer
        spawn_fn(child, parent)


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
