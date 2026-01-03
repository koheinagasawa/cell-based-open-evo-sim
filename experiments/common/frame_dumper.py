import os
from typing import Dict, List, Optional, Tuple

import numpy as np

CellId = str
Pos = Tuple[float, float]
Edge = Tuple[CellId, CellId, float]


class FrameDumper:
    """
    Collects per-frame snapshots and writes four NPY files:
      - field_frames.npy : object array length T, each (H,W) float
      - cell_ids.npy     : object array length T, each List[str]
      - cell_pos.npy     : object array length T, each (N_t,2) float
      - edges.npy        : object array length T, each List[(idA,idB,weight)]
    """

    def __init__(
        self,
        *,
        sample_every: int = 1,
        bounds: Tuple[float, float, float, float] = (-100.0, 100.0, -100.0, 100.0),
        resolution: Tuple[int, int] = (64, 64),
    ):
        self.sample_every = int(sample_every)
        self.bounds = bounds
        self.resolution = resolution
        self._field_frames: List[np.ndarray] = []
        self._ids_per_frame: List[List[CellId]] = []
        self._pos_per_frame: List[np.ndarray] = []
        self._edges_per_frame: List[List[Edge]] = []
        self._n = 0

    def on_step(self, world, time_step) -> None:
        """Call after world.step(time_step). `time_step` may be float; sampling uses integer counter."""
        # Use integer counter to decide sampling (captures at steps 0, N, 2N, ...)
        if (self._n % self.sample_every) != 0:
            self._n += 1
            return

        # ---- capture ----
        field = self._extract_field(world)
        id2pos = self._extract_id2pos(world)
        edges = self._extract_edges(world, id2pos)

        ids = list(id2pos.keys())
        pos = np.asarray([id2pos[i] for i in ids], dtype=float)

        self._field_frames.append(field)
        self._ids_per_frame.append(ids)
        self._pos_per_frame.append(pos)
        self._edges_per_frame.append(edges)

        self._n += 1

    def write_files(self, out_dir: str) -> Dict[str, str]:
        """Write .npy files under out_dir and return their paths."""
        os.makedirs(out_dir, exist_ok=True)

        # Save metadata about field sampling
        import json

        meta_path = os.path.join(out_dir, "field_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"bounds": self.bounds, "resolution": self.resolution}, f)

        # Use dtype=object so variable-size frames are allowed
        def _to_obj_array(pylist):
            arr = np.empty(len(pylist), dtype=object)
            arr[:] = pylist
            return arr

        paths = {}
        paths["field_frames"] = os.path.join(out_dir, "field_frames.npy")
        np.save(
            paths["field_frames"], _to_obj_array(self._field_frames), allow_pickle=True
        )

        paths["cell_ids"] = os.path.join(out_dir, "cell_ids.npy")
        np.save(
            paths["cell_ids"], _to_obj_array(self._ids_per_frame), allow_pickle=True
        )

        paths["cell_pos"] = os.path.join(out_dir, "cell_pos.npy")
        np.save(
            paths["cell_pos"], _to_obj_array(self._pos_per_frame), allow_pickle=True
        )

        paths["edges"] = os.path.join(out_dir, "edges.npy")
        np.save(paths["edges"], _to_obj_array(self._edges_per_frame), allow_pickle=True)

        return paths

    # ---------- helpers (best-effort world introspection) ----------
    def _extract_field(self, world) -> np.ndarray:
        """Return a 2D float array for background. Falls back to zeros."""
        # 1. Try FieldRouter (vectorized sampling)
        if hasattr(world, "field_router") and world.field_router:
            fr = world.field_router
            xmin, xmax, ymin, ymax = self.bounds
            H, W = self.resolution

            xs = np.linspace(xmin, xmax, W)
            ys = np.linspace(ymin, ymax, H)
            xx, yy = np.meshgrid(xs, ys)

            total = np.zeros((H, W), dtype=float)

            for ch in fr.channels.values():
                if ch.dim_space != 2 or not ch.sources:
                    continue

                # Use optimized sample_grid if available
                if hasattr(ch, "sample_grid"):
                    val, _ = ch.sample_grid(xx, yy)
                    total += val
                else:
                    # Fallback: manual sampling (slow)
                    # This path is just a safety net
                    for i in range(H):
                        for j in range(W):
                            v, _ = ch.sample(np.array([xx[i, j], yy[i, j]]))
                            total[i, j] += v

            return total

        f = None
        if hasattr(world, "field"):
            f = world.field
        elif hasattr(world, "fields"):
            try:
                f = world.fields[0]
            except Exception:
                pass
        elif hasattr(world, "get_field"):
            try:
                f = world.get_field()
            except Exception:
                pass
        if f is None:
            return np.zeros((64, 64), dtype=float)
        f = np.asarray(f, dtype=float)
        if f.ndim == 2:
            return f
        if f.ndim == 3:
            return f[..., 0]
        return np.asarray(f).reshape(64, 64)

    def _extract_id2pos(self, world) -> Dict[CellId, Pos]:
        """Build {cell_id: (x,y)} from world cells."""
        cells = (
            getattr(world, "cells", None) or getattr(world, "get_cells", lambda: None)()
        )
        if cells is None:
            raise RuntimeError("World has no 'cells' or 'get_cells()'.")
        id2pos: Dict[CellId, Pos] = {}
        for c in cells:
            cid = str(getattr(c, "id", ""))
            p = np.asarray(getattr(c, "position"), dtype=float).ravel()
            x = float(p[0])
            y = float(p[1] if p.size > 1 else 0.0)
            id2pos[cid] = (x, y)
        return id2pos

    def _extract_edges(self, world, id2pos: Dict[CellId, Pos]) -> List[Edge]:
        """
        Create unique undirected edges (id_low,id_high,weight) to avoid duplicates
        if links are bidirectional.
        """
        edges: List[Edge] = []
        seen = set()

        def add_edge(a: str, b: str, w) -> None:
            if a not in id2pos or b not in id2pos:
                return
            key = (a, b) if a < b else (b, a)
            if key in seen:
                return
            seen.add(key)
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
            edges.append((key[0], key[1], weight))

        cells = (
            getattr(world, "cells", None) or getattr(world, "get_cells", lambda: None)()
        )
        if cells is not None:
            for c in cells:
                if hasattr(c, "get_connections"):
                    try:
                        for nb_id, w in c.get_connections():
                            add_edge(str(c.id), str(nb_id), w)
                        continue
                    except Exception:
                        pass
                if hasattr(c, "connections"):
                    conn = getattr(c, "connections")
                    if isinstance(conn, dict):
                        for nb_id, w in conn.items():
                            add_edge(str(c.id), str(nb_id), w)
                    elif isinstance(conn, (list, tuple)):
                        for pair in conn:
                            try:
                                nb_id, w = pair
                                add_edge(str(c.id), str(nb_id), w)
                            except Exception:
                                pass

        if hasattr(world, "edges"):
            try:
                for a, b, w in world.edges:
                    add_edge(str(a), str(b), w)
            except Exception:
                pass

        return edges
