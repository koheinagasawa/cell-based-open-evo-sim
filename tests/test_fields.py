# tests/test_field_basic.py
import matplotlib.pyplot as plt
import numpy as np

from simulation.cell import Cell
from simulation.fields import FieldChannel, FieldRouter
from simulation.input_layout import InputLayout
from tests.conftest import interpreter4
from tests.utils.visualization import plot_field_scalar_and_quiver


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
        def __init__(self, layout):
            self.layout = layout

        def activate(self, inputs):
            grad = self.layout.get_vector(inputs, "field:pher:grad")
            return {"state": np.zeros(4), "move": grad}

    field_layout = {"field:pher:grad": 2}
    A = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
    )
    B = Cell(
        [1.0, 0.0],
        Follower(layout=InputLayout.from_dicts({}, field_layout)),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout=field_layout,
    )

    w = world_factory([A, B])
    w.field_router = fr
    w.use_fields = True

    p0 = B.position.copy()
    w.step()  # deposit, but B still sees old field (empty) -> no movement
    assert np.allclose(B.position, p0)

    w.step()  # now B sees grad pointing toward A (negative x), so x decreases
    assert B.position[0] < p0[0]


def test_field_steady_state_convergence(world_factory, interpreter4):
    # decay=0.9, deposit=1.0 => steady limit = 10.0 at the emitter location
    fr = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=1.0, decay=0.9)}
    )

    class Emitter:
        """Emit 1.0 every step to drive the field toward steady state."""

        def activate(self, inputs):
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:pher": [1.0]}

    E = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:val": 1},
    )

    w = world_factory([E], field_router=fr, use_fields=True)

    # Step 1: read old field (empty) -> 0, then deposit 1.0
    w.step()
    v_prev = E.field_inputs["field:pher:val"][0]
    assert v_prev == 0.0

    # Iterate more steps and check against the analytical solution:
    # v_i = limit * (1 - d**i), where d=decay, deposit=1 -> limit = 1/(1-d)
    d = 0.9
    deposit = 1.0
    limit = deposit / (1.0 - d)
    N = 70  # number of post-initialization steps
    for i in range(1, N + 1):
        w.step()
        v = E.field_inputs["field:pher:val"][0]
        expected = limit * (1.0 - d**i)
        # Monotonic non-decreasing and bounded by the steady-state limit
        assert v >= v_prev - 1e-12, "Should be non-decreasing under continuous emission"
        assert v <= limit + 1e-9, "Must never exceed the steady-state limit"
        # Match the analytical curve closely (exact for this configuration)
        assert np.allclose(v, expected, rtol=1e-12, atol=1e-12)
        v_prev = v

    # After enough steps, it should be close to the limit
    assert abs(v_prev - limit) < 1e-2


def test_field_multi_channels_isolated(world_factory, interpreter4):
    # Two independent channels
    fr = FieldRouter(
        {
            "A": FieldChannel(name="A", dim_space=2, sigma=1.0, decay=0.9),
            "B": FieldChannel(name="B", dim_space=2, sigma=1.0, decay=0.9),
        }
    )

    class EmitterAOnce:
        """Emit on the first actionable frame only, then stop -> pure decay afterward."""

        def __init__(self):
            self.done = False

        def activate(self, inputs):
            amt = [1.0] if not self.done else [0.0]
            self.done = True
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:A": amt}

    class EmitterBContinuous:
        """Emit 1.0 every step -> monotonic increase toward steady state."""

        def activate(self, inputs):
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:B": [1.0]}

    class Reader:
        def activate(self, inputs):
            # Reader does not move; we inspect field_inputs directly
            return {"state": np.zeros(4), "move": np.zeros(2)}

    A = Cell(
        [0.0, 0.0],
        EmitterAOnce(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:A:val": 1, "field:B:val": 1},
    )
    B = Cell(
        [0.0, 0.0],
        EmitterBContinuous(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:A:val": 1, "field:B:val": 1},
    )
    R = Cell(
        [0.0, 0.0],
        Reader(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:A:val": 1, "field:B:val": 1},
    )

    w = world_factory([A, B, R], field_router=fr, use_fields=True)

    # Step 1: read zeros, then A deposits once and B deposits once
    w.step()
    a0 = R.field_inputs["field:A:val"][0]
    b0 = R.field_inputs["field:B:val"][0]
    assert a0 == 0.0 and b0 == 0.0

    # Step 2: both channels visible
    w.step()
    a1 = R.field_inputs["field:A:val"][0]
    b1 = R.field_inputs["field:B:val"][0]
    assert a1 > 0.0 and b1 > 0.0

    # Step 3: A should decay (no further emission), B should increase (continuous emission)
    w.step()
    a2 = R.field_inputs["field:A:val"][0]
    b2 = R.field_inputs["field:B:val"][0]
    assert a2 < a1, "Channel A must decay after one-shot emission"
    assert b2 > b1, "Channel B must increase under continuous emission"

    # And there must be no cross-talk: A's value is unaffected by B's ongoing emission
    # (i.e., A continues to decay even though B is still being emitted)
    for _ in range(5):
        w.step()
        a_next = R.field_inputs["field:A:val"][0]
        b_next = R.field_inputs["field:B:val"][0]
        assert a_next <= a2 + 1e-9
        assert b_next >= b2 - 1e-9
        a2, b2 = a_next, b_next


def test_field_3d_sampling_smoke(world_factory, interpreter4):
    # 3D channel; Cell.position is 2D but router pads to 3D internally.
    fr = FieldRouter(
        {"pher3": FieldChannel(name="pher3", dim_space=3, sigma=1.0, decay=0.95)}
    )

    class Emitter:
        def activate(self, inputs):
            return {
                "state": np.zeros(4),
                "move": np.zeros(2),
                "emit_field:pher3": [1.0],
            }

    class Reader:
        def activate(self, inputs):
            # Not moving; we inspect field_inputs directly (val + 3D grad)
            return {"state": np.zeros(4), "move": np.zeros(2)}

    E = Cell(
        [0.0, 0.0],
        Emitter(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher3:val": 1, "field:pher3:grad": 3},
    )
    R = Cell(
        [0.5, 0.0],
        Reader(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher3:val": 1, "field:pher3:grad": 3},
    )

    w = world_factory([E, R], field_router=fr, use_fields=True)

    # Step 1: no field yet
    w.step()
    v0 = R.field_inputs["field:pher3:val"][0]
    g0 = R.field_inputs["field:pher3:grad"]
    assert v0 == 0.0 and g0.shape == (3,)

    # Step 2: field visible; grad should point roughly from R toward E (negative x, zero-ish z)
    w.step()
    v1 = R.field_inputs["field:pher3:val"][0]
    g1 = R.field_inputs["field:pher3:grad"]
    assert v1 > 0.0
    assert g1.shape == (3,)
    assert g1[0] < 0.0  # x-gradient points toward E at origin


def test_plot_field_scalar_and_quiver_smoke(
    world_factory, interpreter4, test_output_dir
):
    # Prepare a simple 2D field and a one-shot emitter at origin.
    fr = FieldRouter(
        {"pher": FieldChannel(name="pher", dim_space=2, sigma=1.0, decay=0.95)}
    )

    class EmitterOnce:
        def __init__(self):
            self.done = False

        def activate(self, inputs):
            amt = [1.0] if not self.done else [0.0]
            self.done = True
            return {"state": np.zeros(4), "move": np.zeros(2), "emit_field:pher": amt}

    E = Cell(
        [0.0, 0.0],
        EmitterOnce(),
        interpreter=interpreter4,
        max_neighbors=0,
        recv_layout={},
        field_layout={"field:pher:val": 1},
    )

    w = world_factory([E], field_router=fr, use_fields=True)

    # Step once to deposit, second step to read non-zero field
    w.step()
    w.step()

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    im = plot_field_scalar_and_quiver(ax, w, channel="pher", grid_n=21)
    # The drawn scalar field should have positive values somewhere.
    arr = im.get_array()
    assert float(np.max(arr)) > 0.0

    # Save to ensure no I/O or rendering issues occur.
    out = test_output_dir / "field_vis.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
