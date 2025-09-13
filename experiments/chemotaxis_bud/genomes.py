import numpy as np

from simulation.input_layout import InputLayout

# ---- Simple genomes (declarative IO; no magic indices) ----------------------


class EmitterContinuous:
    """Emit 1.0 into the field every step; no movement, no budding."""

    def __init__(self, state_size: int = 4, field_key: str = "emit_field:pher"):
        self.S = state_size
        self.field_key = field_key

    def activate(self, inputs):
        return {
            "state": np.zeros(self.S),
            "move": np.zeros(2),
            self.field_key: np.array([1.0]),
            "bud": np.array([0.0]),
        }


class FollowerChemotaxisAndBud:
    """
    Move along the field gradient and always emit a bud signal.
    Bud acceptance is governed by the Bud/Energy policies (thresholds etc.).
    """

    def __init__(
        self,
        state_size: int,
        field_grad_key: str,
        grad_gain: float = 1.0,
        layout: InputLayout | None = None,
    ):
        self.S = state_size
        self.field_grad_key = field_grad_key
        self.grad_gain = float(grad_gain)
        # Optional, declaration-driven slicer. If absent, we fallback to "last 2 dims".
        self.layout = layout

    def activate(self, inputs):
        # Prefer declaration-driven slicing by key; fallback to last 2 dims.
        if self.layout is not None:
            grad = self.layout.get_vector(inputs, self.field_grad_key)
        else:
            x = np.asarray(inputs, dtype=float).ravel()
            grad = x[-2:]
        grad = np.asarray(grad, dtype=float) * self.grad_gain
        return {
            "state": np.zeros(self.S),
            "move": grad,
            "bud": np.array([1.0]),
        }
