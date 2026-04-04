"""Spring (bond) forces between connected particles.

Each particle must expose:
  - position:  np.ndarray (D-dimensional)
  - radius:    float
  - id:        str
  - conn_out:  dict[str, float]  (dst_id -> weight)

Rest length for a bond (i, j) is defined as r_i + r_j.
"""
import numpy as np


def compute_spring_forces(particles, *, stiffness: float = 1.0) -> np.ndarray:
    """Return (N, D) force array from spring bonds.

    For each directed edge i->j in conn_out, a Hookean spring force is applied
    (edge weight is currently unused; only connectivity matters):
      displacement = dist - rest_length
      F_on_i = stiffness * displacement * direction_toward_j

    Both i and j receive equal-and-opposite contributions per edge.
    Coincident particles (dist == 0) with nonzero rest length receive a
    deterministic nudge along the first axis.
    Edges referencing unknown particle IDs are silently ignored.
    """
    n = len(particles)
    if n == 0:
        return np.empty((0, 0), dtype=float)

    dim = len(particles[0].position)
    forces = np.zeros((n, dim), dtype=float)

    # Build id -> index lookup
    id_to_idx: dict[str, int] = {}
    for idx, p in enumerate(particles):
        id_to_idx[p.id] = idx

    for i, p_i in enumerate(particles):
        for dst_id, _weight in p_i.conn_out.items():
            j = id_to_idx.get(dst_id)
            if j is None:
                continue  # dangling connection

            p_j = particles[j]
            diff = p_j.position - p_i.position  # from i toward j
            dist = float(np.linalg.norm(diff))
            rest_length = p_i.radius + p_j.radius
            displacement = dist - rest_length

            if abs(displacement) < 1e-15:
                continue  # at rest length

            if dist < 1e-15:
                # Coincident: deterministic nudge along first axis
                direction = np.zeros(dim, dtype=float)
                direction[0] = 1.0
            else:
                direction = diff / dist

            # Force on i: toward j if stretched (displacement > 0),
            #             away from j if compressed (displacement < 0)
            force_mag = stiffness * displacement
            f = force_mag * direction

            forces[i] += f
            forces[j] -= f

    return forces
