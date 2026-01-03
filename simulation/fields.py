# simulation/fields.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _gaussian_kernel(r2: float, sigma: float) -> float:
    """Unnormalized isotropic Gaussian: exp(-0.5 * r^2 / sigma^2)."""
    if sigma <= 0.0:
        return 0.0 if r2 > 0.0 else 1.0
    return float(np.exp(-0.5 * r2 / (sigma * sigma)))


@dataclass
class FieldChannel:
    """
    Continuous scalar field in R^D as a sum of decaying Gaussian sources.

    Value(x) = Σ a_i * K(||x - p_i||)
    Grad(x)  = Σ a_i * dK/dx = Σ a_i * (-(x - p_i)/sigma^2) * K

    Notes:
      * Naive O(N*M) without spatial acceleration (OK for small N).
      * dim_space can be 2 or 3 (future-proof).
    """

    name: str
    dim_space: int = 2
    sigma: float = 1.0
    decay: float = 0.97
    radius: float | None = None  # Optional cutoff
    prune_eps: float = 1e-8  # Cull tiny sources

    # Internal: list[(pos[D], amount)]
    sources: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    def deposit(self, position: np.ndarray, amount: float) -> None:
        """Append a new source; align/truncate pos to D if needed."""
        p = np.asarray(position, dtype=float).ravel()
        D = int(self.dim_space)
        if p.size < D:
            p = np.pad(p, (0, D - p.size))
        elif p.size > D:
            p = p[:D]
        a = float(amount)
        if not np.isfinite(a) or a == 0.0:
            return
        self.sources.append((p, a))

    def apply_decay(self) -> None:
        """Multiply all amounts by decay and prune tiny sources."""
        if not self.sources:
            return
        dec = float(self.decay)
        if dec <= 0.0:
            self.sources.clear()
            return
        new_sources: List[Tuple[np.ndarray, float]] = []
        for p, a in self.sources:
            aa = a * dec
            if abs(aa) >= self.prune_eps:
                new_sources.append((p, aa))
        self.sources = new_sources

    def is_active(self) -> bool:
        """Fast check for whether this channel has any surviving sources."""
        return bool(self.sources)

    def sample(self, position: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (value, grad[D]) at position (R^D)."""
        x = np.asarray(position, dtype=float).ravel()
        D = int(self.dim_space)
        if x.size < D:
            x = np.pad(x, (0, D - x.size))
        elif x.size > D:
            x = x[:D]

        val = 0.0
        grad = np.zeros(D, dtype=float)
        sig2 = float(self.sigma * self.sigma) if self.sigma > 0 else 1.0
        r2max = None if self.radius is None else float(self.radius * self.radius)

        for p, a in self.sources:
            d = x - p
            r2 = float(d @ d)
            if r2max is not None and r2 > r2max:
                continue
            k = _gaussian_kernel(r2, self.sigma)
            if k == 0.0:
                continue
            val += a * k
            grad += a * (-(d / sig2)) * k  # dK/dx

        return float(val), grad

    def sample_grid(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Vectorized sampling on a 2D grid (X, Y).
        Returns (V, (GX, GY)) where V, GX, GY have same shape as X, Y.
        """
        V = np.zeros_like(X, dtype=float)
        GX = np.zeros_like(X, dtype=float)
        GY = np.zeros_like(X, dtype=float)

        if not self.sources:
            return V, (GX, GY)

        sig2 = float(self.sigma * self.sigma) if self.sigma > 0 else 1.0
        r2max = None if self.radius is None else float(self.radius * self.radius)

        # Loop over sources and accumulate (vectorized over grid)
        for p, a in self.sources:
            # p is typically (2,) or (3,) but we only care about x,y for 2D grid
            px, py = p[0], p[1]

            dX = X - px
            dY = Y - py
            r2 = dX * dX + dY * dY

            if r2max is not None:
                # Optimization: only compute exp where r2 <= r2max
                mask = r2 <= r2max
                if not np.any(mask):
                    continue
                k = np.zeros_like(r2)
                k[mask] = np.exp(-0.5 * r2[mask] / sig2)
            else:
                k = np.exp(-0.5 * r2 / sig2)

            # Accumulate value
            V += a * k

            # Accumulate gradient: grad = a * (-(d / sig2)) * k
            # factor = -a / sig2 * k
            factor = (-a / sig2) * k
            GX += factor * dX
            GY += factor * dY

        return V, (GX, GY)


@dataclass
class FieldRouter:
    """
    World-attached router for environmental fields.

    Per frame:
      1) sample_into_cell(cell)  -> writes declared field inputs (val/grad) to cell.field_inputs
      2) apply_decay()           -> decay all channels
      3) collect_from_cells(...) -> consume 'emit_field:*' from output slots and deposit

    Contract: Effects of 'emit_field:*' appear in the NEXT frame.
    """

    channels: Dict[str, FieldChannel]
    # Lightweight per-step metrics (not persisted)
    sample_calls: int = 0
    last_total_sources: int = 0

    def sample_into_cell(self, cell) -> None:
        """
        Populate cell.field_inputs according to cell.field_layout declarations.
        Expected keys:
          - 'field:<name>:val'  (dim=1)
          - 'field:<name>:grad' (dim=D)
        Unknown channels/keys are zeroed.
        """
        layout: Dict[str, int] = getattr(cell, "field_layout", {}) or {}

        # Fast path: the cell does not declare any field inputs.
        if not layout:
            cell.field_inputs = {}
            return

        # Fast path: no active channels globally -> zero-fill without sampling.
        # (This avoids O(#channels) Gaussian evaluations per cell.)
        if not any(ch.is_active() for ch in self.channels.values()):
            out: Dict[str, np.ndarray] = {}
            for key, dim in sorted(layout.items()):
                out[key] = np.zeros(int(dim), dtype=float)
            cell.field_inputs = out
            return

        out: Dict[str, np.ndarray] = {}
        for key, dim in sorted(layout.items()):
            dim = int(dim)
            arr = np.zeros(dim, dtype=float)

            # Parse 'field:<name>:<kind>'
            try:
                _, name, kind = key.split(":", 2)
            except ValueError:
                out[key] = arr
                continue

            ch = self.channels.get(name)
            if ch is None:
                out[key] = arr
                continue

            val, grad = ch.sample(getattr(cell, "position", np.zeros(ch.dim_space)))

            # Count one sampling call (value+grad from the same channel counts as 1)
            self.sample_calls += 1

            if kind == "val":
                if dim > 0:
                    arr[0] = float(val)
            elif kind == "grad":
                g = np.asarray(grad, dtype=float).ravel()
                n = min(dim, g.size)
                if n > 0:
                    arr[:n] = g[:n]
            # else: leave zeros
            out[key] = arr

        cell.field_inputs = out

    def apply_decay(self) -> None:
        # Update per-step aggregate metrics
        self.last_total_sources = sum(len(ch.sources) for ch in self.channels.values())
        self.sample_calls = 0
        for ch in self.channels.values():
            ch.apply_decay()

    def collect_from_cells(self, cells: Iterable) -> None:
        """Consume 'emit_field:<name>' from output slots and deposit scalar amount."""
        for c in cells:
            slots = getattr(c, "output_slots", None)
            if not slots:
                continue
            for k, v in slots.items():
                if not isinstance(k, str) or not k.startswith("emit_field:"):
                    continue
                try:
                    _, name = k.split(":", 1)
                except ValueError:
                    continue
                ch = self.channels.get(name)
                if ch is None:
                    continue
                vec = np.asarray(v, dtype=float).ravel()
                if vec.size == 0:
                    continue
                ch.deposit(
                    getattr(c, "position", np.zeros(ch.dim_space)), float(vec[0])
                )
