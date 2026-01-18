import pytest

from simulation.agent import Agent
from simulation.interpreter import SlotBasedInterpreter


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
