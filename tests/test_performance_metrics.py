import numpy as np

from experiments.common.metrics_hook import PerformanceMetricsHook
from simulation.cell import Cell
from simulation.interpreter import SlotBasedInterpreter
from simulation.world import World


def test_performance_metrics_hook(world_factory):
    # Setup simple world
    S = 4
    interp = SlotBasedInterpreter({"state": slice(0, S), "move": slice(S, S + 2)})

    # Create cells close enough to find each other
    c1 = Cell([0, 0], genome=None, state_size=S, interpreter=interp, max_neighbors=5)
    c2 = Cell(
        [0.5, 0.5], genome=None, state_size=S, interpreter=interp, max_neighbors=5
    )

    # Mock genome that does nothing but allows execution
    class MockGenome:
        def activate(self, inputs, rng=None):
            return [0.0] * (S + 2)

    c1.genome = MockGenome()
    c2.genome = MockGenome()

    world = world_factory([c1, c2], use_neighbors=True)
    hook = PerformanceMetricsHook(prefix="perf_")

    # Run a step
    world.step()

    # Check metrics from hook
    metrics = hook.on_step(world, 0)

    # We expect neighbor search to have happened twice (once for each cell)
    assert metrics["perf_neighbor_search_count"] == 2.0

    # Phase 1 time should be recorded
    assert "perf_time_phase1_decide_ms" in metrics
    assert metrics["perf_time_phase1_decide_ms"] >= 0.0

    # Phase 2 time should be recorded
    assert "perf_time_phase2_commit_state_ms" in metrics
    assert metrics["perf_time_phase2_commit_state_ms"] >= 0.0

    # Phase 3 time should be recorded
    assert "perf_time_phase3_apply_actions_ms" in metrics
    assert metrics["perf_time_phase3_apply_actions_ms"] >= 0.0

    # Phase 4 time should be recorded
    assert "perf_time_phase4_maintenance_ms" in metrics
    assert metrics["perf_time_phase4_maintenance_ms"] >= 0.0

    # Phase 5 time should be recorded
    assert "perf_time_phase5_connected_messaging_ms" in metrics
    assert metrics["perf_time_phase5_connected_messaging_ms"] >= 0.0

    # Neighbor search time (ms) should be recorded
    assert "perf_time_neighbor_search_ms" in metrics
    assert metrics["perf_time_neighbor_search_ms"] >= 0.0
