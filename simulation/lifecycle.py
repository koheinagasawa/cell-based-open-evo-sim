# hooks.py
from __future__ import annotations

import numpy as np


def log_birth(recorder):
    """
    Return an init_child function that records a birth using the given recorder.
    Recorder must implement: record_birth(t, parent_id, child_id, pos)
    """

    def _init(child, parent, world):
        try:
            recorder.record_birth(world.time, parent.id, child.id, child.position)
        except Exception:
            # Logging must never break the simulation
            pass

    return _init


def chain_inits(*inits):
    """
    Compose multiple init_child functions into one.
    Each init: (child, parent, world) -> None
    """

    def _init(child, parent, world):
        for f in inits:
            if f:
                f(child, parent, world)

    return _init


def init_connections_copy(
    child,
    parent,
    world,
    weight_noise_std: float = 0.0,
    clip_min: float | None = 0.0,
) -> None:
    """
    Initialize the child's outgoing connections by copying parent's conn_out.
    Optionally add small Gaussian noise and clip to a minimum.

    This function is intentionally World-agnostic; it only touches parent/child.
    """
    # Copy weights with optional noise
    new_edges: dict[str, float] = {}
    for dst_id, w in getattr(parent, "conn_out", {}).items():
        ww = float(w)
        if weight_noise_std and weight_noise_std > 0.0:
            ww += float(np.random.normal(0.0, weight_noise_std))
        if clip_min is not None:
            ww = max(float(clip_min), ww)
        new_edges[str(dst_id)] = ww
    # Apply to child (kept as ids; resolution is done by router when needed)
    child.set_connections(new_edges)


# Optional registry for config-based selection
BIRTH_INITS = {
    "copy_connections": init_connections_copy,
}
