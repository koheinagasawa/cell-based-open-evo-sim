from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# ---- helper: group positions by cell in 2D ---------------------------------
def _positions_by_cell_2d(recorder):
    """Return {cell_id: (T, X, Y)} sorted by time from recorder.positions rows."""
    rows = getattr(recorder, "positions", None) or []
    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 4:
            continue
        t = int(row[0])
        cid = row[1]
        x = float(row[2])
        y = float(row[3])
        by_cell[cid].append((t, x, y))

    out = {}
    for cid, lst in by_cell.items():
        lst.sort(key=lambda z: z[0])
        T = np.array([t for t, _, _ in lst], dtype=int)
        X = np.array([x for _, x, _ in lst], dtype=float)
        Y = np.array([y for _, _, y in lst], dtype=float)
        out[cid] = (T, X, Y)
    return out


def _timeline_union_2d(data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Return sorted unique times across all cells. Robust to missing frames per cell."""
    if not data:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([T for (T, _, _) in data.values()]))


def _axes_limits_from_data(
    series_xy: list[tuple[np.ndarray, np.ndarray]], pad_ratio: float = 0.05
):
    """Compute static axes limits with a small margin to avoid jitter during animation."""
    allX = np.concatenate([x for x, _ in series_xy])
    allY = np.concatenate([y for _, y in series_xy])
    xmin, xmax = float(allX.min()), float(allX.max())
    ymin, ymax = float(allY.min()), float(allY.max())
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pad = span * pad_ratio
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def plot_state_trajectories(recorder, show=True):
    """Plot per-cell state time series from recorder.states (list of [t, cell_id, *state])."""
    rows = getattr(recorder, "states", None) or []
    if not rows:
        print("No state data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        t = int(row[0])
        cid = row[1][:6]  # use first 6 chars for brevity
        s = np.asarray(row[2:], dtype=float)
        by_cell[cid].append((t, s))

    # Sort by time and stack
    series = []
    for cid, lst in by_cell.items():
        lst.sort(key=lambda x: x[0])
        T = np.array([t for t, _ in lst], dtype=int)
        S = np.vstack([s for _, s in lst])  # (T, state_dim)
        series.append((cid, T, S))

    import math

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (cid, T, S) in zip(axes, series):
        for j in range(S.shape[1]):
            ax.plot(T, S[:, j], label=f"state[{j}]")
        ax.set_title(f"Cell {cid}")
        ax.set_xlabel("t")
        ax.set_ylabel("state")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()


def plot_2D_position_trajectories(
    recorder, show=True, mark_start_end=True, equal_aspect=True
):
    """Plot per-cell 2D position trajectories from recorder.positions (list of [t, cell_id, x, y, ...])."""

    rows = getattr(recorder, "positions", None) or []
    if not rows:
        print("No position data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 4:  # need at least t, id, x, y
            continue
        t = int(row[0])
        cid = row[1][:6]  # use first 6 chars for brevity
        x = float(row[2])
        y = float(row[3])
        by_cell[cid].append((t, x, y))

    series = []
    for cid, lst in by_cell.items():
        lst.sort(key=lambda z: z[0])
        X = np.array([x for _, x, _ in lst], dtype=float)
        Y = np.array([y for _, _, y in lst], dtype=float)
        series.append((cid, X, Y))

    fig, ax = plt.subplots(figsize=(6, 6))
    for cid, X, Y in series:
        ax.plot(X, Y, label=f"cell {cid}", linewidth=1.5)
        if mark_start_end and X.size > 0:
            ax.scatter([X[0]], [Y[0]], marker="o", s=30)  # start
            ax.scatter([X[-1]], [Y[-1]], marker="*", s=80)  # end

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    if show:
        plt.show()


def plot_quiver_last_step(recorder, show=True, equal_aspect=True, scale=None):
    """Plot one arrow per cell using the last step velocity.
    Legend matches other plot functions: one entry per cell id (first 6 chars).
    """
    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    pxs, pys, us, vs, cs = [], [], [], [], []

    for cid_full, (T, X, Y) in data.items():
        if X.size < 2:
            continue

        # Use first 6 chars of the id, keep consistent with other plots
        cid6 = str(cid_full)[:6]

        # Draw trajectory line and record its color for the arrow
        (line_handle,) = ax.plot(X, Y, linewidth=1.5, label=f"cell {cid6}")
        color = line_handle.get_color()

        # Prepare last-step arrow payload (end of trajectory)
        dx, dy = X[-1] - X[-2], Y[-1] - Y[-2]
        pxs.append(X[-1])
        pys.append(Y[-1])
        us.append(dx)
        vs.append(dy)
        cs.append(color)  # color arrows by the line color

    if pxs:
        ax.quiver(
            np.array(pxs),
            np.array(pys),
            np.array(us),
            np.array(vs),
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=cs,  # per-cell color to match the line/legend
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)  # same style as other plots
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_quiver_along_trajectories(
    recorder, arrow_stride=3, show=True, equal_aspect=True, scale=None
):
    """Plot multiple arrows along each trajectory (every `arrow_stride` steps).
    Legend matches other plot functions: one entry per cell id (first 6 chars).
    """
    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    pxs, pys, us, vs, cs = [], [], [], [], []
    stride = max(int(arrow_stride), 1)

    for cid_full, (T, X, Y) in data.items():
        cid6 = str(cid_full)[:6]
        # Draw the trajectory with label (legend comes from these lines)
        (line_handle,) = ax.plot(X, Y, linewidth=1.5, label=f"cell {cid6}")
        color = line_handle.get_color()

        if X.size < 2:
            continue

        # Collect displacement arrows every 'stride' steps
        for i in range(1, X.size, stride):
            pxs.append(X[i - 1])
            pys.append(Y[i - 1])
            us.append(X[i] - X[i - 1])
            vs.append(Y[i] - Y[i - 1])
            cs.append(color)  # color per arrow sample for this cell

    if pxs:
        ax.quiver(
            np.array(pxs),
            np.array(pys),
            np.array(us),
            np.array(vs),
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=cs,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_3d_position_trajectories(
    recorder, show=True, mark_start_end=True, equal_aspect=True
):
    """Draw 3D trajectories from recorder.positions rows: [t, cell_id, x, y, z, ...]."""

    rows = getattr(recorder, "positions", None) or []
    if not rows:
        print("No position data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 5:
            continue
        t = int(row[0])
        cid = row[1][:6]  # use first 6 chars for brevity
        x = float(row[2])
        y = float(row[3])
        z = float(row[4])
        by_cell[cid].append((t, x, y, z))

    series = []
    for cid, lst in by_cell.items():
        lst.sort(key=lambda r: r[0])
        X = np.array([x for _, x, _, _ in lst], dtype=float)
        Y = np.array([y for _, _, y, _ in lst], dtype=float)
        Z = np.array([z for _, _, _, z in lst], dtype=float)
        series.append((cid, X, Y, Z))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cid, X, Y, Z in series:
        ax.plot(X, Y, Z, linewidth=1.5, label=f"cell {cid}")
        if mark_start_end and X.size:
            ax.scatter([X[0]], [Y[0]], [Z[0]], marker="o", s=30)  # start
            ax.scatter([X[-1]], [Y[-1]], [Z[-1]], marker="*", s=80)  # end

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if equal_aspect and series:
        # Simple equal aspect: set symmetric limits around data centroid
        allX = np.concatenate([s[1] for s in series])
        allY = np.concatenate([s[2] for s in series])
        allZ = np.concatenate([s[3] for s in series])
        cx, cy, cz = np.mean(allX), np.mean(allY), np.mean(allZ)
        r = (
            max(
                allX.max() - allX.min(),
                allY.max() - allY.min(),
                allZ.max() - allZ.min(),
            )
            * 0.5
        )
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def animate_2D_position_trajectories(
    recorder,
    tail: int | None = None,
    interval: int = 60,
    equal_aspect: bool = True,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 120,
    blit: bool = True,
    # Legend controls
    legend: str | bool = "auto",  # one of {True, False, "auto", "inside", "outside"}
    legend_cols: int = 1,
    label_shorten: int | None = 8,
):
    """
    Animate per-cell 2D trajectories over time from `recorder.positions`.

    Parameters
    ----------
    recorder : object
        Must expose `positions` as a list of rows `[t, cell_id, x, y, ...]`.
    tail : int | None
        If given, only the most recent `tail` points per cell are drawn (sliding window).
    interval : int
        Delay between frames in milliseconds.
    equal_aspect : bool
        Keep axes equal for geometric fidelity.
    show : bool
        Call `plt.show()` at the end.
    save_path : str | None
        If provided, save the animation. `.gif` uses PillowWriter; `.mp4` tries matplotlib's
        default writer and falls back to GIF when unavailable.
    dpi : int
        Output DPI when saving.
    blit : bool
        Use blitting for performance. Some backends may require `blit=False`.
    legend : {True, False, "auto", "inside", "outside"}
        Control legend rendering. `auto` puts legend outside if there are many cells.
    legend_cols : int
        Number of columns when the legend is drawn.
    label_shorten : int | None
        If set, shorten cell IDs to first `label_shorten` chars to keep legend compact.

    Returns
    -------
    (fig, anim) : matplotlib.figure.Figure, matplotlib.animation.FuncAnimation
    """
    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to animate.")
        return None, None

    times = _timeline_union_2d(data)

    # Precompute axes range to avoid autoscale jitter
    limits = _axes_limits_from_data([(X, Y) for (_, X, Y) in data.values()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # One line per cell
    # Sort by id for stable color assignment across runs
    artists: dict[str, dict] = {}
    for cid, (T, X, Y) in sorted(data.items(), key=lambda kv: str(kv[0])):
        label_cid = str(cid)
        if (
            isinstance(label_shorten, int)
            and label_shorten > 0
            and len(label_cid) > label_shorten
        ):
            label_cid = label_cid[:label_shorten]
        (line,) = ax.plot([], [], linewidth=1.5, label=f"cell {label_cid}")
        artists[cid] = {"line": line, "T": T, "X": X, "Y": Y}

    # Legend policy
    draw_legend = legend if isinstance(legend, bool) else True
    loc_kwargs = {}
    if legend in ("inside", True):
        loc = "best"
    elif legend == "outside" or (legend == "auto" and len(artists) > 8):
        loc = "upper left"
        loc_kwargs = {"bbox_to_anchor": (1.02, 1.0), "borderaxespad": 0.0}
    elif legend == "auto":
        loc = "best"
    else:  # legend is False or "none"
        draw_legend = False
        loc = "best"

    if draw_legend:
        ax.legend(
            loc=loc,
            fontsize=8,
            framealpha=0.85,
            ncol=max(1, int(legend_cols)),
            **loc_kwargs,
        )

    # Current head markers (scatter supports efficient .set_offsets)
    head_scatter = ax.scatter([], [], s=28, marker="o")

    # Build lookup: time -> index per cell (O(1) during updates)
    idxmaps = {
        cid: {int(t): i for i, t in enumerate(entry["T"])}
        for cid, entry in artists.items()
    }

    def init():
        for entry in artists.values():
            entry["line"].set_data([], [])
        head_scatter.set_offsets(np.empty((0, 2)))
        return [entry["line"] for entry in artists.values()] + [head_scatter]

    def update(frame_i: int):
        t = int(times[frame_i])
        heads = []
        for cid, entry in artists.items():
            idx = idxmaps[cid].get(t)
            if idx is None:
                continue  # no sample for this cell at time t
            start = 0 if tail is None else max(0, idx + 1 - int(tail))
            x = entry["X"][start : idx + 1]
            y = entry["Y"][start : idx + 1]
            entry["line"].set_data(x, y)
            heads.append([entry["X"][idx], entry["Y"][idx]])
        if heads:
            head_scatter.set_offsets(np.asarray(heads))
        return [entry["line"] for entry in artists.values()] + [head_scatter]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(times),
        init_func=init,
        interval=int(interval),
        blit=bool(blit),
    )

    if save_path:
        try:
            if save_path.lower().endswith(".gif"):
                fps = max(1, int(round(1000.0 / max(1, interval))))
                anim.save(save_path, writer=PillowWriter(fps=fps), dpi=int(dpi))
            else:
                # Try default writer (e.g., ffmpeg). If unavailable, fall back to GIF.
                anim.save(save_path, dpi=int(dpi))
        except Exception:
            alt = save_path.rsplit(".", 1)[0] + ".gif"
            fps = max(1, int(round(1000.0 / max(1, interval))))
            anim.save(alt, writer=PillowWriter(fps=fps), dpi=int(dpi))
            print(f"Saved as GIF instead: {alt}")

    if show:
        plt.show()
    return fig, anim


def animate_quiver_2D(
    recorder,
    tail_steps: int = 1,
    interval: int = 60,
    scale: float | None = None,
    equal_aspect: bool = True,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 120,
    blit: bool = True,
):
    """
    Animate per-step velocity arrows (quiver) for all cells.

    Robustness tweaks
    -----------------
    * Keep a fixed-size quiver (Nmax = #cells * tail_steps) to avoid shape changes.
    * Use masked arrays instead of NaNs so some Matplotlib backends don't choke
      and GIF encoders actually register per-frame changes.
    * **Also**: when `scale` is None we compute a finite default from the data so that
      Matplotlib does not try to autoscale from an empty (fully masked) first frame,
      which can yield NaN and produce a static-looking GIF.
    * If your GIF still appears static, try `blit=False` when calling this.
    """
    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to animate (quiver).")
        return None, None

    times = _timeline_union_2d(data)
    if len(times) <= 1:
        print("Only a single frame present; nothing to animate.")
        return None, None

    limits = _axes_limits_from_data([(X, Y) for (_, X, Y) in data.values()])

    # Prepare lookups per cell
    cache = {}
    for cid, (T, X, Y) in data.items():
        cache[cid] = {
            "T": T,
            "X": X,
            "Y": Y,
            "idx": {int(t): i for i, t in enumerate(T)},
        }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Optional faint full trajectories for context (kept subtle)
    for cid, (T, X, Y) in data.items():
        ax.plot(X, Y, linewidth=0.6, alpha=0.5)

    # --- Determine a safe scale if none is provided ---
    scale_used = scale
    if scale_used is None:
        mags = []
        for T, X, Y in data.values():
            if len(X) > 1:
                dX = np.diff(X)
                dY = np.diff(Y)
                m = np.hypot(dX, dY)
                if m.size:
                    mags.append(np.median(m))
        med = float(np.median(mags)) if mags else 1.0
        # Larger scale -> shorter arrows when scale_units="xy"
        scale_used = max(1e-6, med)

    M = len(cache)
    K = max(1, int(tail_steps))
    Nmax = M * K

    # Preallocate buffers
    off = np.zeros((Nmax, 2), dtype=float)
    U = np.zeros(Nmax, dtype=float)
    V = np.zeros(Nmax, dtype=float)
    mask = np.ones(Nmax, dtype=bool)  # True = hidden

    # Create quiver once
    quiv = ax.quiver(
        off[:, 0],
        off[:, 1],
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=scale_used,
        pivot="tail",
    )

    def init():
        # Start fully masked (no arrows visible)
        quiv.set_offsets(off)
        quiv.set_UVC(np.ma.array(U, mask=mask), np.ma.array(V, mask=mask))
        return [quiv]

    def update(frame_i: int):
        t = int(times[frame_i])
        # Reset to hidden
        mask[:] = True
        k = 0
        for cid, buf in cache.items():
            idx = buf["idx"].get(t)
            if idx is None or idx <= 0:
                continue
            j0 = max(1, idx + 1 - K)
            for j in range(j0, idx + 1):
                if k >= Nmax:
                    break
                x0, y0 = buf["X"][j - 1], buf["Y"][j - 1]
                dx, dy = buf["X"][j] - x0, buf["Y"][j] - y0
                off[k, 0] = x0
                off[k, 1] = y0
                U[k] = dx
                V[k] = dy
                mask[k] = False  # visible
                k += 1
        quiv.set_offsets(off)
        quiv.set_UVC(np.ma.array(U, mask=mask), np.ma.array(V, mask=mask))
        return [quiv]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(times),
        init_func=init,
        interval=int(interval),
        blit=bool(blit),
    )

    if save_path:
        try:
            if save_path.lower().endswith(".gif"):
                fps = max(1, int(round(1000.0 / max(1, interval))))
                anim.save(save_path, writer=PillowWriter(fps=fps), dpi=int(dpi))
            else:
                anim.save(save_path, dpi=int(dpi))
        except Exception:
            alt = save_path.rsplit(".", 1)[0] + ".gif"
            fps = max(1, int(round(1000.0 / max(1, interval))))
            anim.save(alt, writer=PillowWriter(fps=fps), dpi=int(dpi))
            print(f"Saved as GIF instead: {alt}")

    if show:
        plt.show()
    return fig, anim


def draw_connections(
    ax,
    cells: Iterable["Cell"],
    *,
    # edge styling
    scale: float = 1.0,
    color=None,
    alpha: float = 0.9,
    arrows: bool = True,
    # labeling / clarity
    show_nodes: bool = False,
    node_size: float = 40.0,
    node_color="black",
    node_labels: bool = True,
    labeler: Optional[Callable[["Cell"], str]] = None,
    weight_labels: bool = True,
    weight_fmt: str = "{:.2g}",
    min_abs_w: float = 0.0,
    # curvature for A<->B
    curve_bidirectional: bool = True,
    curve_rad: float = 0.15,
    # autoscale viewport
    autoscale: bool = True,
    pad_frac: float = 0.1,
):
    """
    Draw directed connections on a Matplotlib Axes.
    - 2D only: uses cell.position[:2]
    - Line width is proportional to |weight| * scale
    - Returns list of created artists for optional further styling
    """
    from matplotlib.patches import FancyArrowPatch

    def _stable_color_for_id(sid: str):
        """Map an id string to a stable color from tab20 palette."""
        import matplotlib as mpl

        palette = mpl.rcParams.get("axes.prop_cycle", None)
        if hasattr(palette, "by_key"):
            colors = palette.by_key().get("color", ["#1f77b4"])
        else:
            colors = ["#1f77b4"]
        h = abs(hash(sid))
        return colors[h % len(colors)]

    def _short_id(sid: str, n: int = 4) -> str:
        s = str(sid)
        return s if len(s) <= n else s[-n:]

    artists: List = []  # edge artists (returned)
    text_artists: List = []  # node/weight labels (not returned)
    pts = []  # collect endpoints for autoscale

    cells_list = list(cells)
    # Defensive: ensure unique ids (will raise early if duplicated)
    reg = {c.id: c for c in cells_list}
    if len(reg) != len(cells_list):
        raise ValueError("Duplicate cell ids detected while drawing connections.")

    # Prepare node positions (2D only)
    xy = {}
    for c in cells_list:
        p = np.asarray(c.position, float).ravel()
        if p.size >= 2:
            xy[c.id] = (float(p[0]), float(p[1]))

    # Optional: draw nodes first (so arrows overlay on top)
    if show_nodes and xy:
        xs, ys = zip(*xy.values())
        node_cols = node_color
        ax.scatter(xs, ys, s=node_size, c=node_cols, zorder=2.0)
        if node_labels:
            lab = labeler or (lambda cell: _short_id(cell.id))
            # Slight offset for readability (in axes fraction units)
            xspan = max(xs) - min(xs) if xs else 1.0
            yspan = max(ys) - min(ys) if ys else 1.0
            dx = 0.01 * (xspan if xspan > 0 else 1.0)
            dy = 0.01 * (yspan if yspan > 0 else 1.0)
            for c in cells_list:
                if c.id in xy:
                    x, y = xy[c.id]
                    t = ax.text(
                        x + dx,
                        y + dy,
                        lab(c),
                        fontsize=9,
                        color="black",
                        ha="left",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7
                        ),
                        zorder=3.0,
                    )
                    text_artists.append(t)

    # Pre-compute reverse-edge existence for curvature
    has_rev = set()
    if curve_bidirectional:
        for c in cells_list:
            for dst_id in getattr(c, "conn_out", {}) or {}:
                if dst_id in reg:
                    if c.id in getattr(reg[dst_id], "conn_out", {}):
                        has_rev.add((c.id, dst_id))

    for src in reg.values():
        if src.id not in xy:
            continue  # skip non-2D
        p = np.array(xy[src.id], float)
        for dst_id, w in getattr(src, "conn_out", {}).items():
            if abs(float(w)) < float(min_abs_w):
                continue
            dst = reg.get(dst_id)
            if dst is None or dst.id not in xy:
                continue  # unknown id; ignore silently (topology may be partial)
            q = np.array(xy[dst.id], float)
            pts.append((p[0], p[1]))
            pts.append((q[0], q[1]))
            lw = max(0.6, float(abs(w)) * float(scale) * 2.0)

            # Edge color: use provided `color` or stable per-source color
            col = color or _stable_color_for_id(src.id)

            # Optional curvature for bidirectional pairs (A->B, B->A)
            connectionstyle = "arc3"
            if curve_bidirectional and (src.id, dst.id) in has_rev and src.id != dst.id:
                rad = +curve_rad if str(src.id) < str(dst.id) else -curve_rad
                connectionstyle = f"arc3,rad={rad}"
            if arrows:
                # Arrow with small shrink to avoid covering node markers
                art = FancyArrowPatch(
                    posA=(p[0], p[1]),
                    posB=(q[0], q[1]),
                    arrowstyle="-|>",
                    mutation_scale=9.0,
                    linewidth=lw,
                    color=col,
                    alpha=alpha,
                    shrinkA=4.0,
                    shrinkB=6.0,
                    connectionstyle=connectionstyle,
                    zorder=2.5,
                )

                ax.add_patch(art)
                artists.append(art)
            else:
                (ln,) = ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    linewidth=lw,
                    color=col,
                    alpha=alpha,
                    zorder=2.5,
                )
                artists.append(ln)
            # Weight label at mid-point (on straight line; good enough for small curvature)
            if weight_labels:
                mx, my = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
                tt = ax.text(
                    mx,
                    my,
                    weight_fmt.format(float(w)),
                    fontsize=8,
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7
                    ),
                    zorder=3.0,
                )
                text_artists.append(tt)

    # Auto-fit axes to endpoints when requested
    if autoscale and pts:
        xs, ys = zip(*pts)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = max(1e-9, xmax - xmin)
        dy = max(1e-9, ymax - ymin)
        pad = pad_frac * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
    return artists


def plot_field_scalar_and_quiver(
    ax, world, channel: str, *, xlim=None, ylim=None, grid_n: int = 25
):
    """
    Render a scalar field (value) as an image and its gradient as a quiver.
    Requirements:
      - world.use_fields == True and world.field_router has 'channel'
      - 2D only (for now). 3D channels are not plotted here.
    """
    fr = getattr(world, "field_router", None)
    assert fr is not None and channel in fr.channels, "FieldRouter or channel missing"
    ch = fr.channels[channel]
    assert int(ch.dim_space) == 2, "Use this helper for 2D channels only"

    # Determine bounds
    xs = [c.position[0] for c in world.cells]
    ys = [c.position[1] for c in world.cells]
    if not xs or not ys:
        xs, ys = [0.0], [0.0]
    margin = 1.0
    xlo = xs[0] if xlim is None else xlim[0]
    xhi = xs[0] if xlim is None else xlim[1]
    ylo = ys[0] if ylim is None else ylim[0]
    yhi = ys[0] if ylim is None else ylim[1]
    if xlim is None:
        xlo, xhi = min(xs) - margin, max(xs) + margin
    if ylim is None:
        ylo, yhi = min(ys) - margin, max(ys) + margin

    X = np.linspace(xlo, xhi, grid_n)
    Y = np.linspace(ylo, yhi, grid_n)
    XX, YY = np.meshgrid(X, Y)
    V = np.zeros_like(XX)
    GX = np.zeros_like(XX)
    GY = np.zeros_like(XX)
    for i in range(grid_n):
        for j in range(grid_n):
            val, grad = ch.sample(np.array([XX[i, j], YY[i, j]], dtype=float))
            V[i, j] = val
            GX[i, j] = grad[0]
            GY[i, j] = grad[1]

    im = ax.imshow(V, extent=[xlo, xhi, ylo, yhi], origin="lower", alpha=0.8)
    ax.quiver(XX, YY, GX, GY, angles="xy", scale_units="xy", scale=1.0, width=0.002)
    ax.set_title(f"Field '{channel}': value (imshow) & grad (quiver)")
    return im
