# tests/test_field_action.py
import os
import shutil

import numpy as np

from experiments.common.event_logger import EventLogger
from simulation.cell import Cell
from simulation.fields import FieldRouter
from simulation.interpreter import SlotBasedInterpreter


def test_add_field_action(world_factory):
    # Setup: Interpreter maps slot "add_field"
    S = 4
    # "add_field" slot at index S+2 (after 2 move dims)
    # Value format: [gate, seed, sigma, decay]
    interp = SlotBasedInterpreter(
        {
            "state": slice(0, S),
            "move": slice(S, S + 2),
            "add_field": slice(S + 2, S + 6),
        }
    )

    class DiscoveryGenome:
        def __init__(self, trigger=False):
            self.trigger = trigger

        def activate(self, inputs):
            # [gate=1.0, seed=0.5, sigma=1.2, decay=0.9]
            val = [1.0, 0.5, 1.2, 0.9] if self.trigger else [0.0, 0.0, 0.0, 0.0]
            return [0.0] * S + [0.0, 0.0] + val

    c = Cell([0, 0], DiscoveryGenome(trigger=True), state_size=S, interpreter=interp)

    # Initial fields empty
    fr = FieldRouter({})

    # Capture callback
    events = []

    def on_add(world, info):
        events.append(info)

    w = world_factory(
        [c], use_fields=True, field_router=fr, field_added_callback=on_add
    )

    # Step 1: Action triggers
    w.step()

    # Verify field created
    expected_name = "dynamic_5"  # int(0.5 * 10)
    assert expected_name in fr.channels
    ch = fr.channels[expected_name]
    assert ch.sigma == 1.2
    assert ch.decay == 0.9

    # Verify callback
    assert len(events) == 1
    assert events[0]["field_name"] == expected_name
    assert events[0]["cell"] is c

    # Step 2: Trigger again (same seed) -> Should NOT recreate or duplicate
    w.step()
    # Callback should NOT be called again because field already exists
    assert len(events) == 1


def test_event_logger_field_integration(tmp_path):
    # Verify EventLogger writes the new CSV correctly
    logger = EventLogger(str(tmp_path))

    # Log one field add
    logger.log_field_add(
        time_step=10,
        cell_id="cell_A",
        pos_xy=(1.0, 2.0),
        field_name="dynamic_7",
        sigma=0.5,
        decay=0.9,
    )

    paths = logger.write_csv("events.csv")

    assert "events_field_csv_path" in paths
    p = paths["events_field_csv_path"]
    assert os.path.exists(p)

    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
        # header
        assert "time_step,cell_id,x,y,field_name,sigma,decay" in content
        # row
        assert "10,cell_A,1.0,2.0,dynamic_7,0.5,0.9" in content
