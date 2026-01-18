import numpy as np
import pytest

from simulation.agent import Agent
from simulation.cell import Cell
from simulation.interpreter import ProfiledInterpreter, SlotBasedInterpreter
from simulation.policies import AgentProfileBudding


class ConstantGenome:
    def __init__(self, output_vector):
        self.output_vector = np.array(output_vector, dtype=float)

    def activate(self, inputs):
        return self.output_vector


def test_profile_switches_interpretation():
    # 1. Setup Interpreters for two profiles: "mover" and "emitter"
    # Genome output size = 4.
    # Mover uses indices [0,1] for move. Emitter uses [2,3] for emit.
    # Both use [0,4] (overlapping) for state, just for complexity.

    mover_interp = SlotBasedInterpreter(
        {"state": slice(0, 2), "move": slice(0, 2)}  # Uses first 2 dims as move
    )

    emitter_interp = SlotBasedInterpreter(
        {"state": slice(0, 2), "emit_field:A": slice(2, 4)}  # Uses last 2 dims as emit
    )

    # 2. Create ProfiledInterpreter
    p_interp = ProfiledInterpreter(
        profiles={"mover": mover_interp, "emitter": emitter_interp},
        default_profile="mover",
    )

    # 3. Create Genome outputting [1.0, 1.0, 5.0, 5.0]
    # Mover should see move=[1,1]. Emitter should see emit=[5,5].
    genome = ConstantGenome([1.0, 1.0, 5.0, 5.0])

    # 4. Create Cells with different profiles
    cell_m = Cell([0, 0], genome, interpreter=p_interp, profile="mover", state_size=2)
    cell_e = Cell([0, 0], genome, interpreter=p_interp, profile="emitter", state_size=2)
    cell_d = Cell(
        [0, 0], genome, interpreter=p_interp, profile="unknown", state_size=2
    )  # Default -> mover

    # 5. Step (Act)
    # We can call act() directly with dummy inputs
    cell_m.act([])
    cell_e.act([])
    cell_d.act([])

    # 6. Verify Output Slots
    # Mover
    assert "move" in cell_m.output_slots
    assert "emit_field:A" not in cell_m.output_slots
    np.testing.assert_array_equal(cell_m.output_slots["move"], [1.0, 1.0])

    # Emitter
    assert "move" not in cell_e.output_slots
    assert "emit_field:A" in cell_e.output_slots
    np.testing.assert_array_equal(cell_e.output_slots["emit_field:A"], [5.0, 5.0])

    # Default (should be Mover)
    assert "move" in cell_d.output_slots
    np.testing.assert_array_equal(cell_d.output_slots["move"], [1.0, 1.0])


class MockGenome:
    def activate(self, inputs, rng=None):
        return [0.0] * 10


@pytest.fixture
def agent_setup(world_factory):
    interp = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})
    agent = Agent(genome=MockGenome(), interpreter=interp, id="prof_agent_01")
    parent = agent.spawn_cell(position=[0.0, 0.0], state_size=4, profile="stem")
    parent.energy = 1.0
    return agent, parent, world_factory


def test_profile_inheritance(agent_setup):
    """Test that profile is inherited by default."""
    agent, parent, world_factory = agent_setup

    # inherit_profile=True
    policy = AgentProfileBudding(threshold=0.5, cost=0.1, inherit_profile=True)
    w = world_factory([parent], agents=[agent], reproduction_policy=policy)

    # Signal: [gate=1.0, x=1.0, y=0.0, profile_idx=999 (ignored)]
    bud_signal = np.array([1.0, 1.0, 0.0, 999.0])

    def _spawn(child, parent, meta=None):
        w.add_cell(child)

    policy.apply(w, parent, bud_signal, _spawn)

    assert len(w.cells) == 2
    child = w.cells[1]
    assert child.profile == "stem"  # Same as parent


def test_profile_switching(agent_setup):
    """Test switching profile based on genome output."""
    agent, parent, world_factory = agent_setup

    # Map: 1 -> muscle, 2 -> neuron
    p_map = {1: "muscle", 2: "neuron"}
    policy = AgentProfileBudding(
        threshold=0.5, cost=0.1, inherit_profile=False, profile_map=p_map
    )
    w = world_factory([parent], agents=[agent], reproduction_policy=policy)

    # Signal A: Switch to Muscle (index 1)
    # [gate=1.0, x=1.0, y=0.0, profile=1.0]
    signal_muscle = np.array([1.0, 1.0, 0.0, 1.0])

    def _spawn(child, parent, meta=None):
        w.add_cell(child)

    policy.apply(w, parent, signal_muscle, _spawn)

    child_1 = w.cells[1]
    assert child_1.profile == "muscle"
    assert child_1.agent_id == "prof_agent_01"

    # Signal B: Switch to Neuron (index 2)
    parent.energy = 1.0  # Recharge parent
    signal_neuron = np.array([1.0, -1.0, 0.0, 2.0])

    policy.apply(w, parent, signal_neuron, _spawn)

    child_2 = w.cells[2]
    assert child_2.profile == "neuron"


def test_profile_fallback(agent_setup):
    """Test fallback to parent profile if index is not in map."""
    agent, parent, world_factory = agent_setup

    p_map = {1: "muscle"}
    policy = AgentProfileBudding(
        threshold=0.5, inherit_profile=False, profile_map=p_map
    )
    w = world_factory([parent], agents=[agent], reproduction_policy=policy)

    # Signal: index 5 (not in map)
    signal_unknown = np.array([1.0, 1.0, 0.0, 5.0])

    def _spawn(child, parent, meta=None):
        w.add_cell(child)

    policy.apply(w, parent, signal_unknown, _spawn)

    child = w.cells[1]
    assert child.profile == "stem"  # Fallback to parent
