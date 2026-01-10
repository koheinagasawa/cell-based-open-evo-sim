from __future__ import annotations

from typing import Dict, Optional, Protocol


class MetricsHook(Protocol):
    """Experiment-specific metrics extension."""

    def begin(self, world) -> None:
        """Called once after world is constructed (optional)."""

    def on_step(self, world, step_index: int) -> Dict[str, float]:
        """
        Called each step. Return a dict of {metric_name: scalar}.
        All returned keys will be recorded each step.
        """

    def end(self) -> Optional[Dict[str, float]]:
        """Called once after the loop (optional; return any final scalars)."""


class PerformanceMetricsHook:
    """
    Collects performance statistics exposed by World.perf_stats.
    Expected keys in world.perf_stats:
      - 'neighbor_search_count': Total calls to get_neighbors
      - 'neighbor_search_time': Total time spent in get_neighbors (sec)
      - 'time_phase1_decide': Total time in Phase 1 (ms)
      - 'time_phase2_commit_state': Total time in Phase 2 (ms)
      - 'time_phase3_apply_actions': Total time in Phase 3 (ms)
      - 'time_phase4_maintenance': Total time in Phase 4 (ms)
      - 'time_phase5_connected_messaging': Total time in Phase 5 (ms)
    """

    def __init__(self, prefix: str = "perf_"):
        self.prefix = prefix

    def begin(self, world) -> None:
        pass

    def on_step(self, world, step_index: int) -> Dict[str, float]:
        stats = getattr(world, "perf_stats", {})
        if not stats:
            return {}

        # Convert seconds to ms for consistency where appropriate
        out = {}
        for k, v in stats.items():
            # If the key starts with 'time_' (sec), convert to ms for easier reading
            if k.startswith("time_"):
                out[f"{self.prefix}{k}_ms"] = v * 1000.0
            else:
                out[f"{self.prefix}{k}"] = float(v)
        return out

    def end(self) -> Optional[Dict[str, float]]:
        return None
