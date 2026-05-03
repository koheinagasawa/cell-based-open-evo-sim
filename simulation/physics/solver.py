"""Physics solver: aggregates forces and integrates positions.

Orchestrates repulsion and spring forces, then applies an
overdamped (first-order) Euler position update:

    position += total_force * dt
"""
import numpy as np

from simulation.physics.repulsion import compute_repulsion_forces
from simulation.physics.spring import compute_spring_forces


class PhysicsSolver:
    """Minimal overdamped physics integrator.

    Parameters
    ----------
    dt : float
        Time step size (displacement = force * dt).
    repulsion_stiffness : float
        Stiffness for soft repulsion.
    spring_stiffness : float
        Stiffness for bond springs.
    """

    def __init__(
        self,
        *,
        dt: float = 0.05,
        repulsion_stiffness: float = 1.0,
        spring_stiffness: float = 1.0,
    ):
        self.dt = float(dt)
        self.repulsion_stiffness = float(repulsion_stiffness)
        self.spring_stiffness = float(spring_stiffness)

    def step(self, particles) -> None:
        """Compute forces and update positions in-place."""
        if len(particles) == 0:
            return

        f_repulsion = compute_repulsion_forces(
            particles, stiffness=self.repulsion_stiffness
        )
        f_spring = compute_spring_forces(particles, stiffness=self.spring_stiffness)
        total = f_repulsion + f_spring

        for i, p in enumerate(particles):
            p.position = p.position + total[i] * self.dt
