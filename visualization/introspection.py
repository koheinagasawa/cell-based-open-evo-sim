from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def build_registry(cells) -> Dict[str, "Cell"]:
    """Return {cell_id: cell} ensuring IDs are unique."""
    reg: Dict[str, "Cell"] = {}
    for c in cells:
        if c.id in reg:
            raise ValueError(f"Duplicate cell id detected: {c.id}")
        reg[c.id] = c
    return reg


def export_connections_json(cells, include_positions: bool = True) -> Dict[str, Any]:
    """
    Export a static snapshot of outgoing connections for all cells.

    Returns:
      {
        "nodes": [{"id": "...", "pos":[...]}...],
        "edges": [{"src":"A","dst":"B","w":0.7}, ...]
      }
    """
    nodes = []
    edges = []
    for c in cells:
        nodes.append(
            {
                "id": c.id,
                **(
                    {"pos": c.position.astype(float).tolist()}
                    if include_positions
                    else {}
                ),
            }
        )
        for dst_id, w in getattr(c, "conn_out", {}).items():
            edges.append({"src": c.id, "dst": str(dst_id), "w": float(w)})
    return {"nodes": nodes, "edges": edges}


def export_connections_dot(cells, label_weights: bool = True) -> str:
    """
    Export Graphviz DOT string (directed graph). Edge label shows weight if enabled.
    Node label is the cell id.
    """
    lines: List[str] = ["digraph G {"]
    # Nodes
    for c in cells:
        lines.append(f'  "{c.id}" [label="{c.id}"];')
    # Edges
    for c in cells:
        for dst_id, w in getattr(c, "conn_out", {}).items():
            if label_weights:
                lines.append(f'  "{c.id}" -> "{dst_id}" [label="{float(w):.3g}"];')
            else:
                lines.append(f'  "{c.id}" -> "{dst_id}";')
    lines.append("}")
    return "\n".join(lines)


def degree_stats(cells) -> Dict[str, Dict[str, float]]:
    """
    Compute per-node in/out degree counts and weight sums.

    Returns:
      {
        "A": {"out_deg":2, "in_deg":1, "out_w":1.7, "in_w":0.9},
        ...
      }
    """
    # Initialize
    stats: Dict[str, Dict[str, float]] = {
        c.id: {"out_deg": 0.0, "in_deg": 0.0, "out_w": 0.0, "in_w": 0.0} for c in cells
    }
    # Outgoing
    for c in cells:
        outs = getattr(c, "conn_out", {})
        stats[c.id]["out_deg"] = float(len(outs))
        stats[c.id]["out_w"] = float(sum(outs.values())) if outs else 0.0
        for dst_id, w in outs.items():
            if dst_id in stats:
                stats[dst_id]["in_deg"] += 1.0
                stats[dst_id]["in_w"] += float(w)
    return stats


def to_json_str(obj: Dict[str, Any]) -> str:
    """Just a thin wrapper to ensure stable JSON formatting."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
