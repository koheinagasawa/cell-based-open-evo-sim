from abc import ABC, abstractmethod

import numpy as np


class Interpreter(ABC):
    @abstractmethod
    def interpret(self, output: np.ndarray) -> dict:
        """
        Extracts interpreted values from raw genome output.
        Returns a dictionary of key-value pairs (e.g. "state", "action", "move", etc).
        """
        pass


class SlotBasedInterpreter(Interpreter):
    def __init__(self, slot_defs: dict[str, slice | int]):
        """
        slot_defs: dict mapping slot names to slices or single indices.
        Example: {"state": slice(0, 4), "move": slice(4, 6), "emit_field": 6}
        """
        self.slot_defs = slot_defs

    def interpret(self, output: np.ndarray) -> dict:
        result = {}
        for key, sl in self.slot_defs.items():
            if isinstance(sl, slice):
                val = output[sl]
            elif isinstance(sl, int):
                val = output[sl]
            else:
                raise TypeError(
                    f"Slot definition for key '{key}' must be int or slice, got {type(sl)}"
                )
            result[key] = val

        return result
