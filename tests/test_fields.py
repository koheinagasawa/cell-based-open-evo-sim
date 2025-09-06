# tests/test_field_basic.py
import numpy as np

from simulation.cell import Cell
from simulation.fields import FieldChannel, FieldRouter


def test_field_deposit_and_decay(world_factory, interpreter4):
    # Channel 'pher': 2D, slow decay
    fr = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=1.0, decay=0.9)}
    )

    class Emitter:
        def __init__(self):
            self.emitted = False

        def activate(self, inputs):
            # Emit only once to observe pure decay on later frames.
            amount = [1.0] if not self.emitted else [0.0]
            self.emitted = True
            return {
                "state": np.zeros(4),
                "move": np.zeros(2),
                "emit_field:pher": amount,
            }

    class Reader:
        def activate(self, inputs):
            # inputs tail includes 'field:pher:val' at index -3 (example)
            return {"state": np.zeros(4), "move": np.zeros(2)}

    A = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:val": 1},
    )
    B = Cell(
        [0.5, 0.0],
        Reader(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:val": 1},
    )

    w = world_factory([A, B])
    # Enable fields after world creation
    w.field_router = fr
    w.use_fields = True

    # Step 1: sample old field (empty) -> 0; then deposit from A
    w.step()
    v0_A = A.field_inputs["field:pher:val"][0]
    v0_B = B.field_inputs["field:pher:val"][0]
    assert v0_A == 0.0 and v0_B == 0.0

    # Step 2: now readings see previous deposit
    w.step()
    v1_A = A.field_inputs["field:pher:val"][0]
    v1_B = B.field_inputs["field:pher:val"][0]
    assert v1_A > 0.0 and v1_B > 0.0 and v1_B < v1_A  # farther is weaker

    # Step 3: decay reduces values
    w.step()
    v2_A = A.field_inputs["field:pher:val"][0]
    assert v2_A < v1_A


def test_field_gradient_guides_motion(world_factory, interpreter4):
    fr = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=0.8, decay=0.95)}
    )

    class Emitter:
        def activate(self, inputs):
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:pher": [1.0]}

    class Follower:
        def activate(self, inputs):
            # Assume field tail is last; use last 2 dims as 'grad'
            grad = np.array(inputs)[-2:]
            return {"state": np.zeros(4), "move": grad}

    A = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:grad": 2},
    )
    B = Cell(
        [1.0, 0.0],
        Follower(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:grad": 2},
    )

    w = world_factory([A, B])
    w.field_router = fr
    w.use_fields = True

    p0 = B.position.copy()
    w.step()  # deposit, but B still sees old field (empty) -> no movement
    assert np.allclose(B.position, p0)

    w.step()  # now B sees grad pointing toward A (negative x), so x decreases
    assert B.position[0] < p0[0]
