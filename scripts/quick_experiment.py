import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# 1. Import actual implementations
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.runner_generic import world_factory
from experiments.run_experiment import run_experiment_quick
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ParentChildLinkWrapper, SimpleBudding
from simulation.input_layout import InputLayout


# ------------------------------------------------------------
# 2. Define custom genomes locally
# ------------------------------------------------------------
class SmoothRandomWalker:
    """
    Moves with momentum (smooth random walk) and emits field.
    Uses state[0], state[1] to store current velocity.
    """

    def __init__(self, state_size: int = 4, field_key: str = "emit_field:C"):
        self.S = state_size
        self.field_key = field_key
        self.speed = 0.5
        self.turn_noise = 0.2

    def activate(self, inputs):
        # inputs structure depends on layout, but usually starts with [self_pos, self_state...]
        # We can assume standard layout or just parse from tail if needed.
        # For simplicity in this quick experiment, we assume we can read our own state from inputs.
        # However, standard inputs: [pos(2), state(S), ...]
        # So state is at index 2:2+S
        
        # Extract current velocity from state (memory)
        # inputs is 1D array.
        current_state = inputs[2 : 2 + self.S]
        vx, vy = current_state[0], current_state[1]

        # If zero (first step), pick random direction
        if vx == 0 and vy == 0:
            ang = np.random.uniform(0, 2 * np.pi)
            vx, vy = np.cos(ang), np.sin(ang)
        
        # Add noise to direction
        ang = np.arctan2(vy, vx)
        ang += np.random.normal(0, self.turn_noise)
        
        # Update velocity
        vx = np.cos(ang) * self.speed
        vy = np.sin(ang) * self.speed

        # Write back to state
        new_state = np.zeros(self.S)
        new_state[0] = vx
        new_state[1] = vy

        return {
            "state": new_state,
            "move": np.array([vx, vy]),
            self.field_key: np.array([1.0]),
            "bud": np.array([0.0]),
        }


class SmartFollower:
    """
    Moves towards field gradient with some noise.
    Buds only if field value > threshold and cooldown is ready.
    Uses state[0] as cooldown timer.
    """

    def __init__(
        self,
        state_size: int,
        field_val_key: str,
        field_grad_key: str,
        grad_gain: float = 1.0,
        bud_threshold: float = 0.5,
        cooldown_steps: int = 20,
    ):
        self.S = state_size
        self.field_val_key = field_val_key
        self.field_grad_key = field_grad_key
        self.grad_gain = grad_gain
        self.bud_threshold = bud_threshold
        self.cooldown_steps = cooldown_steps
        
        # Helper to parse inputs by key
        self.layout_helper = None

    def activate(self, inputs):
        # Inputs layout: [pos(2), state(S), num_neighbors(1), field_inputs...]
        # Field inputs are sorted by key: "field:C:grad" comes before "field:C:val"
        # So the tail is: [grad_x, grad_y, val]
        
        grad = inputs[-3:-1]  # Take 2 elements starting from -3
        val = inputs[-1]      # Take the last element
        
        # 1. Movement: Gradient + Noise
        move_vec = grad * self.grad_gain
        # Add random noise (reduced to allow following)
        noise = np.random.normal(0, 0.1, size=2)
        move_vec += noise
        
        # 2. Budding logic
        # Read cooldown from state[0]
        current_state = inputs[2 : 2 + self.S]
        cooldown = current_state[0]
        
        bud_signal = 0.0
        new_cooldown = max(0, cooldown - 1)
        
        if val > self.bud_threshold and cooldown <= 0:
            bud_signal = 1.0
            new_cooldown = self.cooldown_steps
            
        # Update state
        new_state = np.zeros(self.S)
        new_state[0] = new_cooldown

        return {
            "state": new_state,
            "move": move_vec,
            "bud": np.array([bud_signal]),
        }


# ------------------------------------------------------------
# 3. Define positioners for initial placement
# ------------------------------------------------------------
def at_origin(i):
    """Single emitter at the origin."""
    return (0.0, 0.0)


def ring_positions(n=12, R=1.5):
    """Arrange follower cells on a ring."""

    def pos(i):
        ang = 2 * np.pi * i / max(1, n)
        return float(R * np.cos(ang)), float(R * np.sin(ang))

    return pos


# ------------------------------------------------------------
# 3. Define interpreter and other components
# ------------------------------------------------------------
# Example: interpreter defines slots for state/move/bud outputs.
interpreter = lambda: SlotBasedInterpreter(
    {
        "state": slice(0, 4),
        "move": slice(4, 6),
        "bud": 6,
    }
)

# Genome factories (functions returning new genome instances)
emitter_genome = lambda: SmoothRandomWalker(state_size=4, field_key="emit_field:C")
follower_genome = lambda: SmartFollower(
    state_size=4,
    field_val_key="field:C:val",
    field_grad_key="field:C:grad",
    grad_gain=5.0,
    bud_threshold=0.1,
    cooldown_steps=30,
)

# Reproduction policy
policy = lambda: ParentChildLinkWrapper(
    SimpleBudding(threshold=0.5, cost=0.3, init_energy=0.5),
    weight=0.5,
    bidirectional=True,
)


# ------------------------------------------------------------
# 4. Run experiment
# ------------------------------------------------------------
def make_unique_outdir(base="outputs/demo_quickrun"):
    """Create a unique timestamped output folder like outputs/demo_quickrun_20251013_2145_a7f3."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = np.random.randint(0, 0xFFFF)
    out_dir = f"{base}_{ts}_{suffix:04x}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


result = run_experiment_quick(
    out_dir=make_unique_outdir(),  # output folder for all artifacts
    steps=300,  # number of simulation steps
    seed=42,  # random seed
    world_factory=world_factory(),  # callable returning a World
    interpreter=interpreter,  # interpreter instance or factory
    populations=[
        dict(
            count=1,
            positioner=at_origin,
            genome_factory=emitter_genome,
            field_layout={"field:C:val": 1},
            recv_layout={},
            max_neighbors=0,
        ),
        dict(
            count=20,
            positioner=ring_positions(20, 4.0),
            genome_factory=follower_genome,
            field_layout={"field:C:val": 1, "field:C:grad": 2},
            recv_layout={},
            max_neighbors=0,
        ),
    ],
    policy=policy,  # budding + link policy
    fields=[("C", 2.0, 0.02)], 
    sample_every=1,  # record every step
    make_gif=True,  # automatically create GIF
)

# ------------------------------------------------------------
# 5. Outputs
# ------------------------------------------------------------
print("\n=== Experiment finished ===")
print("Output directory:", result["out_dir"])
print("GIF animation:", result.get("gif"))
print("Metrics keys:", [k for k in result.keys() if k not in ("out_dir", "gif")])
