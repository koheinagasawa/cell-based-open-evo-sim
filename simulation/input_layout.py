# simulation/input_layout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class InputLayout:
    """
    Minimal, declaration-driven helper to *parse tail inputs* (recv/field) without magic indices.

    Design notes:
    - This helper assumes experiments running with max_neighbors == 0.
      In that regime, the input vector structure is:
        [ self_pos, self_state, (optional time), (optional num_neighbors=0),
          recv_tail(sorted by key), field_tail(sorted by key) ]
    - We avoid making assumptions about the head (pos/state/time/etc).
      Instead, we *derive tail length* from the cell's layouts and slice from the end.
    - For future evolvable IO, callers should treat keys as semantic labels rather than fixed offsets.
    """

    # Ordered keys and dims for deterministic tail composition
    recv_items: List[Tuple[str, int]]
    field_items: List[Tuple[str, int]]

    @staticmethod
    def from_cell(cell) -> "InputLayout":
        """Convenience only. Not required if you prefer pure dicts."""
        recv_items = sorted(
            [(k, int(v)) for k, v in (getattr(cell, "recv_layout", {}) or {}).items()],
            key=lambda kv: kv[0],
        )
        field_items = sorted(
            [(k, int(v)) for k, v in (getattr(cell, "field_layout", {}) or {}).items()],
            key=lambda kv: kv[0],
        )
        return InputLayout(recv_items=recv_items, field_items=field_items)

    @staticmethod
    def from_dicts(
        recv_layout: Dict[str, int] | None, field_layout: Dict[str, int] | None
    ) -> "InputLayout":
        """
        Pure functional constructor: no Cell required.
        Pass exactly the dicts you used to construct the Cell.
        """
        recv_items = sorted(
            [(k, int(v)) for k, v in ((recv_layout or {}).items())],
            key=lambda kv: kv[0],
        )
        field_items = sorted(
            [(k, int(v)) for k, v in ((field_layout or {}).items())],
            key=lambda kv: kv[0],
        )
        return InputLayout(recv_items=recv_items, field_items=field_items)

    # ---------- Tail sizing ----------
    def recv_tail_dim(self) -> int:
        return sum(dim for _, dim in self.recv_items)

    def field_tail_dim(self) -> int:
        return sum(dim for _, dim in self.field_items)

    def total_tail_dim(self) -> int:
        return self.recv_tail_dim() + self.field_tail_dim()

    # ---------- Tail slicing helpers ----------
    def _tail_slice_range(self, inputs: Iterable[float]) -> Tuple[int, int]:
        """Return [start, end) indices of the full tail inside inputs."""
        n = len(inputs)
        t = self.total_tail_dim()
        return (n - t, n)

    def _recv_slice_range(self, inputs: Iterable[float]) -> Tuple[int, int]:
        n = len(inputs)
        r = self.recv_tail_dim()
        f = self.field_tail_dim()
        return (n - (r + f), n - f)

    def _field_slice_range(self, inputs: Iterable[float]) -> Tuple[int, int]:
        n = len(inputs)
        f = self.field_tail_dim()
        return (n - f, n)

    # ---------- Public API ----------
    def split_tail(self, inputs: Iterable[float]) -> Dict[str, np.ndarray]:
        """
        Parse the tail portion into a {key: vector} dict.
        Keys include 'recv:*' and 'field:*:*' as declared on the cell.
        """
        arr = np.asarray(inputs, dtype=float).ravel()
        out: Dict[str, np.ndarray] = {}

        # recv tail
        if self.recv_items:
            s, e = self._recv_slice_range(arr)
            cursor = s
            for key, dim in self.recv_items:
                out[key] = arr[cursor : cursor + dim].copy()
                cursor += dim

        # field tail
        if self.field_items:
            s, e = self._field_slice_range(arr)
            cursor = s
            for key, dim in self.field_items:
                out[key] = arr[cursor : cursor + dim].copy()
                cursor += dim

        return out

    def get_vector(self, inputs: Iterable[float], key: str) -> np.ndarray:
        """Convenience: return a single vector by key; zeros if not declared."""
        if key not in dict(self.recv_items + self.field_items):
            return np.zeros(0, dtype=float)
        return self.split_tail(inputs).get(key, np.zeros(0, dtype=float))
