# utils/event_logger.py
import csv
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple


@dataclass
class BirthEvent:
    # NOTE: keep columns explicit for stable CSV schema
    time_step: int
    parent_id: Optional[str]  # None if unknown (e.g., spontaneous)
    child_id: str
    x: float
    y: float
    link_weight: Optional[float]  # None if not applicable


class EventLogger:
    """Collects birth events and writes them to events.csv."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self._birth_events: List[BirthEvent] = []

    def log_birth(
        self,
        time_step: int,
        parent_id: Optional[str],
        child_id: str,
        pos_xy: Tuple[float, float],
        link_weight: Optional[float],
    ) -> None:
        """Record a single birth event."""
        x, y = float(pos_xy[0]), float(pos_xy[1])
        self._birth_events.append(
            BirthEvent(time_step, parent_id, child_id, x, y, link_weight)
        )

    def write_csv(self, filename: str = "events.csv") -> str:
        """Write all birth events to CSV (append-safe overwrite)."""
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time_step",
                    "parent_id",
                    "child_id",
                    "x",
                    "y",
                    "link_weight",
                ],
            )
            writer.writeheader()
            for ev in self._birth_events:
                writer.writerow(asdict(ev))
        return path

    def clear(self) -> None:
        self._birth_events.clear()
