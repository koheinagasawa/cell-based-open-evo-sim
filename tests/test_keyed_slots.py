# tests/test_keyed_slots.py
import numpy as np

from simulation.cell import Cell


def test_vector_path_backcompat(interpreter4):
    vec = np.arange(6, dtype=float)
    out = interpreter4.interpret(vec)
    assert "state" in out


def test_dict_passthrough(interpreter4):
    raw = {
        "state": np.array([1, 2, 3, 4], float),
        "move": np.array([0.1, -0.2], float),
    }
    out = interpreter4.interpret(raw)
    assert set(out.keys()) >= {"state", "move"}
    np.testing.assert_allclose(out["state"], [1, 2, 3, 4])
    np.testing.assert_allclose(out["move"], [0.1, -0.2])


def test_cell_with_keyed_genome(interpreter4, world_factory):
    class DictGenome:
        def __init__(self, S=4):
            self.S = S

        def activate(self, inputs):
            return {
                "state": np.zeros(self.S, dtype=float) + 7.0,  # easy to assert
                "move": np.array([0.05, -0.05], dtype=float),
            }

    g = DictGenome(S=4)
    c = Cell([0, 0], g, state_size=4, interpreter=interpreter4)
    w = world_factory([c])
    w.step()  # two-phase commit applies 'state' after step
    assert np.allclose(c.state, [7, 7, 7, 7])
    # keyed slot is present in output_slots
    assert "move" in getattr(c, "output_slots", {})
