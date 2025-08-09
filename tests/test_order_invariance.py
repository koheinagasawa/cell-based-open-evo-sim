
import numpy as np

class FirstNeighborChaserGenome:
    """Move toward the first neighbor's relative position."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
    def activate(self, inputs):
        # inputs = [self_pos(2), self_state(S), n0_rel(2), n0_state(S), ...]
        dx, dy = inputs[2 + self.state_size : 2 + self.state_size + 2]
        return [0.0] * self.state_size + [dx, dy]

def test_two_phase_is_order_invariant(world_line_factory):
    w1 = world_line_factory(order="normal", genome_builder=lambda i: FirstNeighborChaserGenome(4, 2))
    w2 = world_line_factory(order="reversed", genome_builder=lambda i: FirstNeighborChaserGenome(4, 2))
    w1.step(); w2.step()
    pos1 = np.stack([c.position for c in w1.cells], axis=0)
    pos2 = np.stack([c.position for c in w2.cells], axis=0)
    np.testing.assert_allclose(pos1, pos2, atol=1e-12)
