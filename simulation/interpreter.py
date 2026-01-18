from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Interpreter(ABC):
    @abstractmethod
    def interpret(self, output: np.ndarray, profile: Any = None) -> dict:
        """
        Extracts interpreted values from raw genome output.
        Returns a dictionary of key-value pairs. (e.g. "state", "action", "move", etc).

        Args:
            output: Raw vector or dict from genome.
            profile: Optional context (e.g. cell profile ID) to switch interpretation rules.
        """
        pass


# --- Keyed Slots naming convention (minimal, non-enforced) -------------------
# Reserved: "state"  (required when genome returns a dict)
# Conventional keys seen today: "move", "bud_gate", etc. (project-specific)
# Recommended for multi-cell I/O:
#   - Outbound signals: "emit:<name>"  e.g., "emit:a", "emit:chem1"
#   - Inbound  signals: "recv:<name>"  e.g., "recv:a", "recv:chem1"
# These are NOT enforced here; we simply document the convention.


class SlotBasedInterpreter(Interpreter):
    def __init__(self, slot_defs: dict[str, slice | int]):
        """
        slot_defs: dict mapping slot names to slices or single indices.
        Example: {"state": slice(0, 4), "move": slice(4, 6), "emit_field": 6}
        """
        self.slot_defs = slot_defs

    def interpret(self, output, profile: Any = None) -> dict:
        # Note: SlotBasedInterpreter ignores 'profile' by default (stateless mapping)

        """
        Dict path:
          - If `output` is a dict, pass it through (values -> np.asarray(float)).
          - 'state' key is required (for two-phase commit).
        Vector path:
          - If `output` is a vector, slice it by slot_defs.
          - slot_defs supports: int | slice | index array (list/tuple/ndarray).
        """
        # 1) Dict passthrough: accept keyed slots directly from genome
        if isinstance(output, dict):
            out = {}
            for k, v in output.items():
                out[k] = np.asarray(v, dtype=float)
            if "state" not in out:
                raise KeyError("Keyed output must include 'state'")
            return out

        # 2) Vector path: slice by slot_defs
        vec = np.asarray(output, dtype=float)
        result = {}
        for key, sl in self.slot_defs.items():
            if isinstance(sl, slice):
                val = vec[sl]
            elif isinstance(sl, int):
                val = vec[sl]
            elif isinstance(sl, (list, tuple, np.ndarray)):
                # Advanced indexing with an index array (1-D gather).
                # NOTE: indices must be in-bounds; coerced to int.
                idx = np.asarray(sl, dtype=int)
                val = vec[idx]
            else:
                raise TypeError(
                    f"Slot definition for key '{key}' must be int or slice, got {type(sl)}"
                )
            result[key] = val

        return result

    def to_metadata(self):
        """Return JSON-serializable metadata for provenance/logging."""
        import numpy as np

        # Helper to serialize
        def _serialize_slotdef(sd):
            # Normalize different slot definition formats into JSON-friendly dicts
            if isinstance(sd, slice):
                return {
                    "kind": "slice",
                    "start": sd.start,
                    "stop": sd.stop,
                    "step": sd.step,
                }
            if isinstance(sd, (int, np.integer)):
                return {"kind": "index", "index": int(sd)}
            if isinstance(sd, (list, tuple, np.ndarray)):
                return {"kind": "indices", "indices": [int(x) for x in sd]}
            # Fallback (rare): unknown type -> stringified
            return {"kind": "unknown", "value": str(sd)}

        def _slot_len(sd):
            # Best-effort length computation
            if isinstance(sd, slice):
                start = 0 if sd.start is None else int(sd.start)
                stop = 0 if sd.stop is None else int(sd.stop)
                step = 1 if sd.step in (None, 0) else int(sd.step)
                if stop < start or step <= 0:
                    return 0
                return (stop - start + step - 1) // step
            if isinstance(sd, (int, np.integer)):
                return 1
            if isinstance(sd, (list, tuple, np.ndarray)):
                return len(sd)
            return None  # unknown

        slot_defs_meta = {
            name: _serialize_slotdef(sd) for name, sd in self.slot_defs.items()
        }
        slot_lengths = {name: _slot_len(sd) for name, sd in self.slot_defs.items()}
        total_dim = None
        if all(v is not None for v in slot_lengths.values()):
            total_dim = int(sum(slot_lengths.values()))

        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "slot_defs": slot_defs_meta,  # JSON-friendly view
            "slot_lengths": slot_lengths,  # per-slot lengths (int or None)
            "total_output_dim": total_dim,  # sum of lengths if computable
        }


class ProfiledInterpreter(Interpreter):
    """
    Switches between different SlotBasedInterpreters (or definitions) based on 'profile'.
    """

    def __init__(
        self, profiles: Dict[Any, SlotBasedInterpreter], default_profile: Any = None
    ):
        self.profiles = profiles
        self.default_profile = default_profile

    def interpret(self, output: np.ndarray, profile: Any = None) -> dict:
        key = profile if profile in self.profiles else self.default_profile

        interpreter = self.profiles.get(key)
        if interpreter is None:
            # Fallback or empty if no matching profile found
            return {}

        return interpreter.interpret(output, profile=profile)

    def to_metadata(self):
        return {
            "class": self.__class__.__name__,
            "profiles": {
                str(k): (v.to_metadata() if hasattr(v, "to_metadata") else str(v))
                for k, v in self.profiles.items()
            },
            "default_profile": str(self.default_profile),
        }
