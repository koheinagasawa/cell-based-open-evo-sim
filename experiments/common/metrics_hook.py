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
