"""Soft repulsion force between overlapping particles.

Each particle must expose:
  - position: np.ndarray (D-dimensional)
  - radius:   float
"""
import numpy as np


def compute_repulsion_forces(particles, *, stiffness: float = 1.0) -> np.ndarray:
    """Return (N, D) force array from pairwise soft-repulsion.

    Force is linear in overlap depth:
      F_ij = stiffness * overlap * direction   (overlap = r_i + r_j - dist)
    Only applies when overlap > 0 (particles intersecting).

    Coincident particles (dist == 0) receive a deterministic nudge
    along the first axis to avoid division by zero.
    """
    n = len(particles)
    if n == 0:
        return np.empty((0, 0), dtype=float)

    dim = len(particles[0].position)
    forces = np.zeros((n, dim), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            diff = particles[i].position - particles[j].position
            dist = float(np.linalg.norm(diff))
            sum_r = particles[i].radius + particles[j].radius
            overlap = sum_r - dist

            if overlap <= 0.0:
                continue

            if dist < 1e-15:
                # Coincident: deterministic nudge along first axis
                direction = np.zeros(dim, dtype=float)
                direction[0] = 1.0
            else:
                direction = diff / dist

            force_mag = stiffness * overlap
            f = force_mag * direction

            forces[i] += f
            forces[j] -= f

    return forces
