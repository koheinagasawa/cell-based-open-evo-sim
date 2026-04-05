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
from simulation.physics.solver import PhysicsSolver
from simulation.policies import ParentChildLinkWrapper, SimpleBudding
from simulation.input_layout import InputLayout


# ============================================================
# Scenario registry
# ============================================================
# Each scenario is a function returning kwargs for run_experiment_quick.
# Usage: python scripts/quick_experiment.py [scenario_name]
# Default: chemotaxis_bud
# ============================================================


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
        grad_gain: float = 0.2,
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
    grad_gain=0.5,
    bud_threshold=10.0,
    cooldown_steps=50,
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


def scenario_chemotaxis_bud():
    """Chemotaxis + budding scenario (no physics)."""
    return dict(
        out_dir=make_unique_outdir(),
        steps=200,
        seed=42,
        world_factory=world_factory(),
        interpreter=interpreter,
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
                count=4,
                positioner=ring_positions(20, 4.0),
                genome_factory=follower_genome,
                field_layout={"field:C:val": 1, "field:C:grad": 2},
                recv_layout={},
                max_neighbors=0,
            ),
        ],
        policy=policy,
        fields=[("C", 20.0, 0.9)],
        sample_every=5,
        make_gif=True,
    )


class _LeaderGenome:
    """Random-walking leader cell. Drives the body by voluntary movement."""

    def __init__(self, speed=0.15):
        self.speed = speed

    def activate(self, inputs):
        state = inputs[2:6]
        vx, vy = state[0], state[1]
        if vx == 0 and vy == 0:
            ang = np.random.uniform(0, 2 * np.pi)
            vx, vy = np.cos(ang), np.sin(ang)
        ang = np.arctan2(vy, vx) + np.random.normal(0, 0.3)
        vx, vy = np.cos(ang) * self.speed, np.sin(ang) * self.speed
        new_state = np.zeros(4)
        new_state[0], new_state[1] = vx, vy
        return {"state": new_state, "move": np.array([vx, vy])}


class _PassiveGenome:
    """Passive cell: no voluntary movement. Relies on physics forces."""

    def activate(self, inputs):
        return {"state": np.zeros(4), "move": np.zeros(2)}


def scenario_physics_body():
    """Physics body demo: leader cell pulls connected passive cells via springs."""
    interp_body = lambda: SlotBasedInterpreter(
        {"state": slice(0, 4), "move": slice(4, 6)}
    )

    def hex_body_positions(n_ring=6, R=0.8):
        """Center at origin + ring of n_ring cells."""
        positions = [(0.0, 0.0)]
        for i in range(n_ring):
            ang = 2 * np.pi * i / n_ring
            positions.append((R * np.cos(ang), R * np.sin(ang)))
        return positions

    # Pre-build cells with connections so we can set up bonds
    from simulation.cell import Cell

    positions = hex_body_positions()
    n_ring = 6
    center_id = "leader"
    ring_ids = [f"body_{i}" for i in range(n_ring)]

    cells = []
    # Leader (center)
    leader = Cell(
        positions[0],
        _LeaderGenome(speed=0.15),
        id=center_id,
        interpreter=interp_body(),
        state_size=4,
        max_neighbors=0,
        radius=0.4,
    )
    leader.set_connections({rid: 1.0 for rid in ring_ids})
    cells.append(leader)

    # Ring (passive)
    for i in range(n_ring):
        c = Cell(
            positions[i + 1],
            _PassiveGenome(),
            id=ring_ids[i],
            interpreter=interp_body(),
            state_size=4,
            max_neighbors=0,
            radius=0.4,
        )
        conns = {center_id: 1.0}
        conns[ring_ids[(i - 1) % n_ring]] = 1.0
        conns[ring_ids[(i + 1) % n_ring]] = 1.0
        c.set_connections(conns)
        cells.append(c)

    # We need a custom world_factory that injects pre-built cells
    # instead of using populations. Use run_experiment_quick with a
    # single "population" of count=0 and manually add cells.
    # Simpler: build the scenario manually.
    from experiments.common.runner_generic import world_factory as wf_module

    out_dir = make_unique_outdir()
    solver = PhysicsSolver(dt=0.05, repulsion_stiffness=3.0, spring_stiffness=3.0)

    w = wf_module()(
        cells,
        seed=42,
        physics_solver=solver,
    )

    # Run and record frames manually
    from tests.utils.visualization2d import animate_field_cells_connections

    field_frames, cell_frames, edge_frames = [], [], []
    steps = 300
    for _ in range(steps):
        cell_frame = {c.id: (float(c.position[0]), float(c.position[1])) for c in w.cells}
        edges = []
        for c in w.cells:
            for dst_id, wt in c.conn_out.items():
                edges.append((c.id, dst_id, float(wt)))
        field_frames.append(np.zeros((32, 32), dtype=float))
        cell_frames.append(cell_frame)
        edge_frames.append(edges)
        w.step()

    # Final frame
    cell_frame = {c.id: (float(c.position[0]), float(c.position[1])) for c in w.cells}
    edges = []
    for c in w.cells:
        for dst_id, wt in c.conn_out.items():
            edges.append((c.id, dst_id, float(wt)))
    field_frames.append(np.zeros((32, 32), dtype=float))
    cell_frames.append(cell_frame)
    edge_frames.append(edges)

    gif_path = os.path.join(out_dir, "physics_body.gif")
    animate_field_cells_connections(
        out_path=gif_path,
        field_frames=field_frames,
        cell_frames=cell_frames,
        edge_frames=edge_frames,
        fps=20,
        trail_len=40,
        figsize=(6, 6),
        cmap="gray",
        show_colorbar=False,
    )
    return {"out_dir": out_dir, "gif": gif_path}


SCENARIOS = {
    "chemotaxis_bud": scenario_chemotaxis_bud,
    "physics_body": scenario_physics_body,
}

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "chemotaxis_bud"
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
        sys.exit(1)

    print(f"Running scenario: {name}")
    scenario_fn = SCENARIOS[name]

    if name == "physics_body":
        # physics_body builds its own world, returns result dict directly
        result = scenario_fn()
    else:
        kwargs = scenario_fn()
        result = run_experiment_quick(**kwargs)

    print("\n=== Experiment finished ===")
    print("Output directory:", result["out_dir"])
    print("GIF animation:", result.get("gif"))
    if "metrics_csv_path" in result:
        print("Metrics keys:", [k for k in result.keys() if k not in ("out_dir", "gif")])
