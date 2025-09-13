import numpy as np

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

    def __init__(self, state_size: int, field_grad_key: str, grad_gain: float = 1.0):
        self.S = state_size
        self.field_grad_key = field_grad_key
        self.grad_gain = float(grad_gain)

    def activate(self, inputs):
        # The interpreter/cell composes inputs; tail contains field grad (declared).
        # We avoid magic indices by slicing from the tail length.
        x = np.asarray(inputs, dtype=float).ravel()
        # The last 2 elements are our declared grad (dim=2) by construction here.
        grad = x[-2:] * self.grad_gain
        return {
            "state": np.zeros(self.S),
            "move": grad,
            "bud": np.array([1.0]),
        }
