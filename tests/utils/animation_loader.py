from typing import Dict, List, Optional, Tuple

import numpy as np

CellId = str
Pos = Tuple[float, float]


def build_frames_from_recorder(
    *,
    # Below arrays/lists should come from your recorder outputs (npz/csv).
    field_frames: List[np.ndarray],  # [T] of (H,W)
    ids_per_frame: List[List[CellId]],  # [T][N_t]
    pos_per_frame: List[np.ndarray],  # [T] of (N_t,2)
    edges_per_frame: Optional[List[list]] = None,  # [T] of [(idA,idB,weight)]
):
    """Return (field_frames, cell_frames, edge_frames) for animate_field_cells_connections."""
    T = len(field_frames)
    assert (
        len(ids_per_frame) == T and len(pos_per_frame) == T
    ), "Recorder sequences length mismatch"
    if edges_per_frame is None:
        edges_per_frame = [[] for _ in range(T)]

    cell_frames: List[Dict[CellId, Pos]] = []
    for t in range(T):
        ids = ids_per_frame[t]
        pos = np.asarray(pos_per_frame[t], dtype=float)
        id2pos = {
            str(cid): (float(pos[i, 0]), float(pos[i, 1])) for i, cid in enumerate(ids)
        }
        cell_frames.append(id2pos)

    return field_frames, cell_frames, edges_per_frame
