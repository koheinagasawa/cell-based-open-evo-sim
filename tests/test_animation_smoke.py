import numpy as np

from tests.utils.visualization2d import animate_field_cells_connections


def test_animation_smoke(test_output_dir):
    T, H, W = 20, 64, 64  # short run
    # synthetic field: moving gaussian blob
    yy, xx = np.mgrid[0:H, 0:W]
    field_frames = []
    cell_frames = []
    edge_frames = []
    for t in range(T):
        cx = 16 + t * 1.2
        cy = 32
        field = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 9.0)))
        field_frames.append(field)

        # two cells orbiting a point
        a = (32 + 10 * np.cos(t * 0.3), 32 + 10 * np.sin(t * 0.3))
        b = (32 + 10 * np.cos(t * 0.3 + np.pi), 32 + 10 * np.sin(t * 0.3 + np.pi))
        cell_frames.append({"A": a, "B": b})

        # single edge with time-varying weight in [0,1]
        w = (np.sin(t * 0.2) + 1) / 2
        edge_frames.append([("A", "B", float(w))])

    out = test_output_dir / "smoke.gif"
    animate_field_cells_connections(
        out_path=str(out),
        field_frames=field_frames,
        cell_frames=cell_frames,
        edge_frames=edge_frames,
        fps=15,
        trail_len=10,
        figsize=(4, 4),
        cmap="viridis",
        show_colorbar=False,
    )
    assert out.exists() and out.stat().st_size > 0
