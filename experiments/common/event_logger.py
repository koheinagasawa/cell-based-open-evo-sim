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


@dataclass
class FieldEvent:
    time_step: int
    cell_id: str
    x: float
    y: float
    field_name: str
    sigma: float
    decay: float


class EventLogger:
    """Collects birth events and writes them to events.csv."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self._birth_events: List[BirthEvent] = []
        self._field_events: List[FieldEvent] = []

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

    def log_field_add(
        self,
        time_step: int,
        cell_id: str,
        pos_xy: Tuple[float, float],
        field_name: str,
        sigma: float,
        decay: float,
    ) -> None:
        """Record a field creation event."""
        x, y = float(pos_xy[0]), float(pos_xy[1])
        self._field_events.append(
            FieldEvent(time_step, cell_id, x, y, field_name, sigma, decay)
        )

    def write_csv(self, filename: str = "events.csv") -> str:
        """Write all birth events to CSV (append-safe overwrite)."""
        os.makedirs(self.out_dir, exist_ok=True)
        paths = {}

        # Birth events
        if self._birth_events:
            path_birth = os.path.join(self.out_dir, filename)
            with open(path_birth, "w", newline="", encoding="utf-8") as f:
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
            paths["events_birth_csv_path"] = path_birth

        # Field events
        if self._field_events:
            path_field = os.path.join(self.out_dir, "events_field.csv")
            with open(path_field, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "time_step",
                        "cell_id",
                        "x",
                        "y",
                        "field_name",
                        "sigma",
                        "decay",
                    ],
                )
                writer.writeheader()
                for ev in self._field_events:
                    writer.writerow(asdict(ev))
            paths["events_field_csv_path"] = path_field

        return paths

    def clear(self) -> None:
        self._birth_events.clear()
        self._field_events.clear()
