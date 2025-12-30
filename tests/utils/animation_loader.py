from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

CellId = str
Pos = Tuple[float, float]


@dataclass(frozen=True)
class PlotRangeSpec:
    """Controls plot view range for animations."""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    pad_frac: float = 0.05
    min_span: float = 1.0
    square: bool = True
    percentile_clip: Optional[Tuple[float, float]] = None  # e.g. (1.0, 99.0)


def _compute_bbox_from_pos_per_frame(
    pos_per_frame: List[np.ndarray],
    *,
    percentile_clip: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Compute bbox (xmin, xmax, ymin, ymax) across all frames."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for pos in pos_per_frame:
        p = np.asarray(pos, dtype=float)
        if p.size == 0:
            continue
        xs.append(p[:, 0])
        ys.append(p[:, 1])

    if not xs:
        # No cells at all; default bbox.
        return -0.5, 0.5, -0.5, 0.5

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    if percentile_clip is not None:
        lo, hi = percentile_clip
        xmin, xmax = np.percentile(x_all, [lo, hi])
        ymin, ymax = np.percentile(y_all, [lo, hi])
    else:
        xmin, xmax = float(x_all.min()), float(x_all.max())
        ymin, ymax = float(y_all.min()), float(y_all.max())

    return xmin, xmax, ymin, ymax


def _make_square_range(
    xmin: float, xmax: float, ymin: float, ymax: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Expand the shorter axis to make x/y spans equal, keeping the center."""
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    sx = xmax - xmin
    sy = ymax - ymin
    s = max(sx, sy)
    return (cx - 0.5 * s, cx + 0.5 * s), (cy - 0.5 * s, cy + 0.5 * s)


def resolve_plot_range_from_frames(
    *,
    pos_per_frame: List[np.ndarray],
    spec: Optional[PlotRangeSpec] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Decide (xlim, ylim) from pos_per_frame and optional spec overrides."""
    spec = spec or PlotRangeSpec()

    # If both ranges are explicitly provided, honor them as-is.
    if spec.xlim is not None and spec.ylim is not None:
        return spec.xlim, spec.ylim

    xmin, xmax, ymin, ymax = _compute_bbox_from_pos_per_frame(
        pos_per_frame,
        percentile_clip=spec.percentile_clip,
    )

    # Prevent degenerate ranges.
    sx = xmax - xmin
    sy = ymax - ymin
    if sx < 1e-12:
        xmin -= 0.5 * spec.min_span
        xmax += 0.5 * spec.min_span
        sx = spec.min_span
    if sy < 1e-12:
        ymin -= 0.5 * spec.min_span
        ymax += 0.5 * spec.min_span
        sy = spec.min_span

    # Add padding.
    padx = spec.pad_frac * sx
    pady = spec.pad_frac * sy
    xmin -= padx
    xmax += padx
    ymin -= pady
    ymax += pady

    if spec.square:
        xlim, ylim = _make_square_range(xmin, xmax, ymin, ymax)
    else:
        xlim, ylim = (xmin, xmax), (ymin, ymax)

    # If only one axis was explicitly provided, honor it.
    if spec.xlim is not None:
        xlim = spec.xlim
    if spec.ylim is not None:
        ylim = spec.ylim

    return xlim, ylim


def build_frames_from_recorder(
    *,
    field_frames: List[np.ndarray],          # [T] of (H,W)
    ids_per_frame: List[List[CellId]],       # [T][N_t]
    pos_per_frame: List[np.ndarray],         # [T] of (N_t,2)
    edges_per_frame: Optional[List[list]] = None,  # [T] of [(idA,idB,weight)]
    view: Optional[PlotRangeSpec] = None,
):
    """
    Return (field_frames, cell_frames, edge_frames, view_range) for animation.

    view_range is ((xmin, xmax), (ymin, ymax)).
    If `view` is None or lacks explicit limits, it will be estimated from cell positions.
    """
    T = len(field_frames)
    assert len(ids_per_frame) == T and len(pos_per_frame) == T, \
        "Recorder sequences length mismatch"

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

    view_range = resolve_plot_range_from_frames(pos_per_frame=pos_per_frame, spec=view)

    return field_frames, cell_frames, edges_per_frame, view_range
