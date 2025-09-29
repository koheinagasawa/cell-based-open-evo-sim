from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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
