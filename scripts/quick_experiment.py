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

from experiments.chemotaxis_bud.genomes import (
    EmitterContinuous,
    FollowerChemotaxisAndBud,
)
from experiments.common.runner_generic import world_factory
from experiments.run_experiment import run_experiment_quick
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import ParentChildLinkWrapper, SimpleBudding


# ------------------------------------------------------------
# 2. Define positioners for initial placement
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
emitter_genome = lambda: EmitterContinuous(state_size=4, field_key="emit_field:C")
follower_genome = lambda: FollowerChemotaxisAndBud(
    state_size=4,
    field_grad_key="field:C:grad",
    grad_gain=1.0,
)

# Reproduction policy
policy = lambda: ParentChildLinkWrapper(
    SimpleBudding(),
    weight=0.8,
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
    steps=200,  # number of simulation steps
    seed=0,  # random seed
    world_factory=world_factory(),  # callable returning a World
    interpreter=interpreter,  # interpreter instance or factory
    populations=[
        (1, at_origin, emitter_genome),  # one emitter
        (12, ring_positions(12, 1.5), follower_genome),  # twelve followers
    ],
    policy=policy,  # budding + link policy
    fields=[("C", 1.5, 0.02)],  # [(name, sigma, decay)]
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
