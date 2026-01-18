import numpy as np
import pytest

from simulation.agent import Agent
from simulation.interpreter import SlotBasedInterpreter
from simulation.policies import AgentBudding


class MockGenome:
    def activate(self, inputs):
        return [0.0] * 6


def test_agent_initialization():
    genome = MockGenome()
    interpreter = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})

    agent = Agent(genome, interpreter)

    assert agent.id is not None
    assert agent.genome is genome
    assert agent.interpreter is interpreter
    assert agent.cells == []


def test_agent_spawn_cell_shares_resources():
    genome = MockGenome()
    interpreter = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})
    agent = Agent(genome, interpreter)

    # Spawn two cells with specific configurations
    c1 = agent.spawn_cell(position=[0.0, 0.0], state_size=4)
    c2 = agent.spawn_cell(position=[1.0, 1.0], state_size=4)

    # Check ownership
    assert len(agent.cells) == 2
    assert c1 in agent.cells
    assert c2 in agent.cells

    # Check shared resources
    assert c1.genome is agent.genome
    assert c2.genome is agent.genome
    assert c1.interpreter is agent.interpreter

    # Check agent ID tagging
    assert getattr(c1, "agent_id") == agent.id
    assert getattr(c2, "agent_id") == agent.id


@pytest.fixture
def agent_setup(world_factory):
    # Setup dependencies
    interp = SlotBasedInterpreter({"state": slice(0, 4), "move": slice(4, 6)})
    genome = MockGenome()

    # Create the Agent
    agent = Agent(genome=genome, interpreter=interp, id="test_agent_01")

    # Create the Parent Cell (initially belonging to the agent)
    parent = agent.spawn_cell(position=[0.0, 0.0], state_size=4)
    parent.energy = 1.0  # Sufficient energy

    return agent, parent, world_factory


def test_budding_via_agent_lookup(agent_setup):
    """
    Test that AgentBudding correctly looks up the agent from the world
    and uses it to spawn the child.
    """
    agent, parent, world_factory = agent_setup

    # 1. Create World with Agent registered
    policy = AgentBudding(threshold=0.5, cost=0.1)
    w = world_factory([parent], agents=[agent], reproduction_policy=policy)

    # Verify pre-conditions
    assert parent.agent_id == "test_agent_01"
    assert len(agent.cells) == 1

    # 2. Trigger Budding
    # Signal: [gate=1.0, off_x=1.0, off_y=0.0]
    bud_signal = np.array([1.0, 1.0, 0.0])

    def _spawn(child, parent, meta=None):
        w.add_cell(child)

    policy.apply(w, parent, bud_signal, _spawn)

    # 3. Verify Child Properties
    assert len(w.cells) == 2
    child = w.cells[1]

    # Child should have the same agent_id
    assert child.agent_id == "test_agent_01"

    # Child should be registered in the Agent's internal list
    # (This proves agent.spawn_cell was called)
    assert len(agent.cells) == 2
    assert child in agent.cells

    # Child should share the exact same genome instance
    assert child.genome is agent.genome
