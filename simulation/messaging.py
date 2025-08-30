from __future__ import annotations

from typing import Dict, List

import numpy as np


class MessageRouter:
    """
    Routes keyed 'emit:*' output slots along Cell.conn_out to build next-frame
    'recv:*' inbox vectors. Reduction: sum (weighted).
    """

    def __init__(self):
        pass

    @staticmethod
    def _emit_keys(slots: dict) -> List[str]:
        return [k for k in slots.keys() if isinstance(k, str) and k.startswith("emit:")]

    def route_and_stage(self, cells: List["Cell"]) -> None:
        """
        Read each cell.output_slots['emit:*'], distribute to connected neighbors
        with edge weights, and stage into neighbor._next_inbox['recv:*'].
        Uses neighbor.recv_layout to determine target dims (truncate/pad).
        """
        idreg: Dict[str, "Cell"] = {c.id: c for c in cells}

        # Clear staging
        for c in cells:
            c._next_inbox = {}

        for src in cells:
            slots = getattr(src, "output_slots", None) or {}
            emit_keys = self._emit_keys(slots)
            if not emit_keys or not src.conn_out:
                continue

            # Resolve outgoing connections once
            pairs = (
                src.connected_pairs(idreg) if hasattr(src, "connected_pairs") else []
            )
            if not pairs:
                continue

            for ek in emit_keys:
                vec = np.asarray(slots[ek], dtype=float).ravel()
                rkey = "recv:" + ek.split(":", 1)[1]  # 'emit:x' -> 'recv:x'
                for dst, w in pairs:
                    # Determine destination dim from declared layout (if any)
                    dim = (
                        int(dst.recv_layout.get(rkey, vec.size))
                        if hasattr(dst, "recv_layout")
                        else vec.size
                    )
                    out = np.zeros(dim, dtype=float)
                    n = min(dim, vec.size)
                    if n > 0:
                        out[:n] = vec[:n]
                    acc = dst._next_inbox.get(rkey)
                    if acc is None:
                        dst._next_inbox[rkey] = float(w) * out
                    else:
                        dst._next_inbox[rkey] = acc + float(w) * out

    @staticmethod
    def swap_inboxes(cells: List["Cell"]) -> None:
        """Commit staged inboxes for next frame."""
        for c in cells:
            c.inbox = c._next_inbox
            c._next_inbox = {}
