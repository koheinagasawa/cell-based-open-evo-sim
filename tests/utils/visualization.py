# ---- helper: group positions by cell in 2D ---------------------------------
def _positions_by_cell_2d(recorder):
    """Return {cell_id: (T, X, Y)} sorted by time from recorder.positions rows."""
    from collections import defaultdict

    import numpy as np

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


def plot_state_trajectories(recorder, show=True):
    """Plot per-cell state time series from recorder.states (list of [t, cell_id, *state])."""
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    rows = getattr(recorder, "states", None) or []
    if not rows:
        print("No state data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        t = int(row[0])
        cid = row[1]
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
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np

    rows = getattr(recorder, "positions", None) or []
    if not rows:
        print("No position data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 4:  # need at least t, id, x, y
            continue
        t = int(row[0])
        cid = row[1]
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
    Requires at least two position samples per cell.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    pxs, pys, us, vs = [], [], [], []

    for cid, (T, X, Y) in data.items():
        if X.size >= 2:
            # trajectory (optional thin line for context)
            ax.plot(X, Y, linewidth=1.0)
            # last-step arrow
            dx, dy = X[-1] - X[-2], Y[-1] - Y[-2]
            pxs.append(X[-1])
            pys.append(Y[-1])
            us.append(dx)
            vs.append(dy)

    if pxs:
        ax.quiver(
            np.array(pxs),
            np.array(pys),
            np.array(us),
            np.array(vs),
            angles="xy",
            scale_units="xy",
            scale=scale,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return fig, ax


def plot_quiver_along_trajectories(
    recorder, arrow_stride=3, show=True, equal_aspect=True, scale=None
):
    """Plot multiple arrows along each trajectory (every `arrow_stride` steps).
    Arrow at step i represents the displacement from i-1 to i.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    data = _positions_by_cell_2d(recorder)
    if not data:
        print("No position data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    pxs, pys, us, vs = [], [], [], []
    stride = max(int(arrow_stride), 1)

    for cid, (T, X, Y) in data.items():
        ax.plot(X, Y, linewidth=1.0)  # path for context
        if X.size < 2:
            continue
        for i in range(1, X.size, stride):
            pxs.append(X[i - 1])
            pys.append(Y[i - 1])
            us.append(X[i] - X[i - 1])
            vs.append(Y[i] - Y[i - 1])

    if pxs:
        ax.quiver(
            np.array(pxs),
            np.array(pys),
            np.array(us),
            np.array(vs),
            angles="xy",
            scale_units="xy",
            scale=scale,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return fig, ax


# tests/viz/plot3d.py
def plot_3d_position_trajectories(
    recorder, show=True, mark_start_end=True, equal_aspect=True
):
    """Draw 3D trajectories from recorder.positions rows: [t, cell_id, x, y, z, ...]."""
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rows = getattr(recorder, "positions", None) or []
    if not rows:
        print("No position data to plot.")
        return

    by_cell = defaultdict(list)
    for row in rows:
        if len(row) < 5:
            continue
        t = int(row[0])
        cid = row[1]
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
        import numpy as np

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
        import matplotlib.pyplot as plt

        plt.show()
    return fig, ax
