import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.utils.animation_loader import build_frames_from_recorder
from tests.utils.visualization2d import animate_field_cells_connections


def main(run_dir: str, out_gif: str):
    """
    Expected files (adjust to your recorder):
      - {run_dir}/field_frames.npy         -> list-like length T, each (H,W)
      - {run_dir}/cell_ids.npy             -> list-like length T, each list[str]
      - {run_dir}/cell_pos.npy             -> array/list length T, each (N_t,2)
      - {run_dir}/edges.npy  (optional)    -> list-like length T, each list[(idA,idB,w)]
    If your recorder uses different names/npz, edit here accordingly.
    """
    field_frames = np.load(f"{run_dir}/field_frames.npy", allow_pickle=True).tolist()
    ids_per_frame = np.load(f"{run_dir}/cell_ids.npy", allow_pickle=True).tolist()
    pos_per_frame = np.load(f"{run_dir}/cell_pos.npy", allow_pickle=True).tolist()

    try:
        edges_per_frame = np.load(f"{run_dir}/edges.npy", allow_pickle=True).tolist()
    except Exception:
        edges_per_frame = None

    # Try to read field metadata
    import json
    import os

    field_extent = None
    try:
        with open(os.path.join(run_dir, "field_metadata.json"), "r") as f:
            meta = json.load(f)
            field_extent = tuple(meta["bounds"])
    except Exception:
        pass

    ff, cf, ef, vr = build_frames_from_recorder(
        field_frames=field_frames,
        ids_per_frame=ids_per_frame,
        pos_per_frame=pos_per_frame,
        edges_per_frame=edges_per_frame,
    )

    out_path = animate_field_cells_connections(
        out_path=out_gif,
        field_frames=ff,
        cell_frames=cf,
        edge_frames=ef,
        view_range=vr,
        fps=15,
        trail_len=30,
        figsize=(6, 6),
        cmap="viridis",
        show_colorbar=True,
        field_extent=field_extent,
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    import sys

    assert (
        len(sys.argv) == 3
    ), "Usage: python scripts/make_animation_from_run.py <run_dir> <out.gif>"
    main(sys.argv[1], sys.argv[2])
