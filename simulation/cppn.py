"""Immutable feedforward CPPNs with direct evaluation and fixed-graph mutation."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from numbers import Integral, Real
from typing import Literal

import numpy as np


def _gaussian(value: float) -> float:
    return math.exp(-(value * value))


_ACTIVATIONS: dict[str, Callable[[float], float]] = {
    "identity": lambda value: value,
    "tanh": math.tanh,
    "sin": math.sin,
    "gaussian": _gaussian,
}


def _is_id(value: object) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and value >= 0
    )


def _finite_real(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _check_fields(data: dict[str, object], expected: set[str], label: str) -> None:
    if data.keys() != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")


@dataclass(frozen=True)
class NodeGene:
    """A computed node; IDs below the genome's input_size are implicit inputs."""

    id: int
    activation: str
    bias: float = 0.0


@dataclass(frozen=True)
class ConnectionGene:
    source: int
    target: int
    weight: float


@dataclass(frozen=True)
class MutationConfig:
    """Explicit per-gene probabilities and Gaussian perturbation scales."""

    weight_probability: float
    bias_probability: float
    activation_probability: float
    weight_std: float
    bias_std: float

    def __post_init__(self) -> None:
        for name in (
            "weight_probability",
            "bias_probability",
            "activation_probability",
            "weight_std",
            "bias_std",
        ):
            value = _finite_real(getattr(self, name), name)
            if name.endswith("_probability"):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
            elif value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MutationRecord:
    """One actual change; gene_id is a node ID or a (source, target) edge."""

    parameter: Literal["weight", "bias", "activation"]
    gene_id: int | tuple[int, int]
    before: float | str
    after: float | str


@dataclass(frozen=True)
class CPPNGenome:
    """Immutable graph parameters plus a derived feedforward evaluation order.

    Input IDs are range(input_size); computed node IDs are integers >=input_size.
    output_ids selects output_size distinct computed nodes in Raw Vector order.
    Both sizes are positive integers supplied by the caller and fixed at construction.
    All nodes, including nodes outside output ancestry, must form a DAG.
    Activation names are identity, tanh, sin and gaussian (exp(-x*x)).
    No output label or Interpreter mapping belongs to this evaluator.
    """

    input_size: int
    output_size: int
    nodes: Sequence[NodeGene]
    connections: Sequence[ConnectionGene]
    output_ids: Sequence[int]
    _order: tuple[NodeGene, ...] = field(init=False, repr=False, compare=False)
    _incoming: tuple[tuple[ConnectionGene, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("input_size", "output_size"):
            size = getattr(self, name)
            if not _is_id(size) or size == 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(size))

        for name in ("nodes", "connections", "output_ids"):
            if not isinstance(getattr(self, name), Sequence):
                raise ValueError(f"{name} must be a sequence")

        nodes: dict[int, NodeGene] = {}
        for node in self.nodes:
            if not isinstance(node, NodeGene):
                raise ValueError("nodes must contain NodeGene values")
            if not _is_id(node.id) or node.id < self.input_size:
                raise ValueError(
                    f"computed node id must be an integer >= {self.input_size}"
                )
            if node.id in nodes:
                raise ValueError(f"duplicate node id: {node.id}")
            if (
                not isinstance(node.activation, str)
                or node.activation not in _ACTIVATIONS
            ):
                raise ValueError(
                    f"unknown activation for node {node.id}: {node.activation!r}"
                )
            nodes[node.id] = NodeGene(
                int(node.id),
                node.activation,
                _finite_real(node.bias, f"node {node.id} bias"),
            )

        if len(self.output_ids) != self.output_size:
            raise ValueError(
                f"genome must declare exactly {self.output_size} output ids"
            )
        output_ids: list[int] = []
        for node_id in self.output_ids:
            if not _is_id(node_id) or node_id not in nodes:
                raise ValueError(
                    f"output id must identify a computed node: {node_id!r}"
                )
            if node_id in output_ids:
                raise ValueError(f"duplicate output id: {node_id}")
            output_ids.append(int(node_id))

        incoming: dict[int, list[ConnectionGene]] = {node_id: [] for node_id in nodes}
        outgoing: dict[int, list[int]] = {node_id: [] for node_id in nodes}
        indegree = {node_id: 0 for node_id in nodes}
        connections: dict[tuple[int, int], ConnectionGene] = {}
        for edge in self.connections:
            if not isinstance(edge, ConnectionGene):
                raise ValueError("connections must contain ConnectionGene values")
            if not _is_id(edge.source) or (
                edge.source >= self.input_size and edge.source not in nodes
            ):
                raise ValueError(f"unknown connection source: {edge.source!r}")
            if not _is_id(edge.target) or edge.target not in nodes:
                raise ValueError(
                    f"connection target must be a computed node: {edge.target!r}"
                )
            key = (int(edge.source), int(edge.target))
            if key in connections:
                raise ValueError(f"duplicate connection: {key}")
            normalized = ConnectionGene(
                key[0], key[1], _finite_real(edge.weight, f"connection {key} weight")
            )
            connections[key] = normalized
            incoming[normalized.target].append(normalized)
            if normalized.source in nodes:
                indegree[normalized.target] += 1
                outgoing[normalized.source].append(normalized.target)

        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        heapify(ready)
        order: list[NodeGene] = []
        while ready:
            node_id = heappop(ready)
            order.append(nodes[node_id])
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heappush(ready, target)
        if len(order) != len(nodes):
            raise ValueError("CPPN graph contains a cycle")

        object.__setattr__(
            self, "nodes", tuple(nodes[node_id] for node_id in sorted(nodes))
        )
        object.__setattr__(
            self, "connections", tuple(connections[key] for key in sorted(connections))
        )
        object.__setattr__(self, "output_ids", tuple(output_ids))
        object.__setattr__(self, "_order", tuple(order))
        object.__setattr__(
            self,
            "_incoming",
            tuple(
                tuple(sorted(incoming[node.id], key=lambda edge: edge.source))
                for node in order
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Export fresh JSON-compatible genotype data, excluding derived caches."""
        return {
            "schema_version": 1,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "nodes": [
                {"id": node.id, "activation": node.activation, "bias": node.bias}
                for node in self.nodes
            ],
            "connections": [
                {"source": edge.source, "target": edge.target, "weight": edge.weight}
                for edge in self.connections
            ],
            "output_ids": list(self.output_ids),
        }

    @classmethod
    def from_dict(cls, data: object) -> "CPPNGenome":
        """Restore schema version 1; malformed data raises ValueError."""
        if not isinstance(data, dict):
            raise ValueError("genotype must be a dictionary")
        _check_fields(
            data,
            {
                "schema_version",
                "input_size",
                "output_size",
                "nodes",
                "connections",
                "output_ids",
            },
            "genotype",
        )
        version = data["schema_version"]
        if type(version) is not int or version != 1:
            raise ValueError("schema_version must be the integer 1")
        for name in ("nodes", "connections", "output_ids"):
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be a JSON array")

        nodes: list[NodeGene] = []
        for record in data["nodes"]:
            if not isinstance(record, dict):
                raise ValueError("nodes must contain dictionaries")
            _check_fields(record, {"id", "activation", "bias"}, "nodes")
            nodes.append(
                NodeGene(
                    id=record["id"],
                    activation=record["activation"],
                    bias=record["bias"],
                )
            )
        connections: list[ConnectionGene] = []
        for record in data["connections"]:
            if not isinstance(record, dict):
                raise ValueError("connections must contain dictionaries")
            _check_fields(record, {"source", "target", "weight"}, "connections")
            connections.append(
                ConnectionGene(
                    source=record["source"],
                    target=record["target"],
                    weight=record["weight"],
                )
            )
        return cls(
            input_size=data["input_size"],
            output_size=data["output_size"],
            nodes=nodes,
            connections=connections,
            output_ids=data["output_ids"],
        )

    def copy(self) -> "CPPNGenome":
        """Rebuild an independent immutable genotype and its derived caches."""
        return type(self)(
            input_size=self.input_size,
            output_size=self.output_size,
            nodes=self.nodes,
            connections=self.connections,
            output_ids=self.output_ids,
        )

    def mutate(
        self, config: MutationConfig, rng: np.random.Generator
    ) -> tuple["CPPNGenome", tuple[MutationRecord, ...]]:
        """Return an independent child and actual changes, keeping graph/I/O fixed.

        Visit nodes by ID (bias, then activation), then edges by (source, target).
        Disabled parameters consume no draws. Enabled parameters draw a gate even
        at probability one, followed by noise or an alternative activation index.
        Non-finite mutations raise FloatingPointError without clipping or retry;
        the parent stays unchanged, but consumed RNG draws are not rolled back.
        """
        if not isinstance(config, MutationConfig):
            raise ValueError("config must be a MutationConfig")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be an explicit numpy.random.Generator")
        records: list[MutationRecord] = []

        def mutate_number(
            value: float,
            probability: float,
            std: float,
            parameter: Literal["weight", "bias"],
            gene_id: int | tuple[int, int],
        ) -> float:
            if probability == 0.0 or std == 0.0:
                return value
            if rng.random() >= probability:
                return value
            changed = value + float(rng.normal(0.0, std))
            if not math.isfinite(changed):
                raise FloatingPointError(
                    f"non-finite mutation of {parameter} at gene {gene_id}"
                )
            if changed != value:
                records.append(MutationRecord(parameter, gene_id, value, changed))
            return changed

        nodes: list[NodeGene] = []
        for node in self.nodes:
            bias = mutate_number(
                node.bias, config.bias_probability, config.bias_std, "bias", node.id
            )
            activation = node.activation
            if (
                config.activation_probability > 0.0
                and rng.random() < config.activation_probability
            ):
                alternatives = tuple(
                    name for name in sorted(_ACTIVATIONS) if name != activation
                )
                activation = alternatives[int(rng.integers(len(alternatives)))]
                records.append(
                    MutationRecord("activation", node.id, node.activation, activation)
                )
            nodes.append(NodeGene(node.id, activation, bias))
        connections = [
            ConnectionGene(
                edge.source,
                edge.target,
                mutate_number(
                    edge.weight,
                    config.weight_probability,
                    config.weight_std,
                    "weight",
                    (edge.source, edge.target),
                ),
            )
            for edge in self.connections
        ]
        child = type(self)(
            input_size=self.input_size,
            output_size=self.output_size,
            nodes=nodes,
            connections=connections,
            output_ids=self.output_ids,
        )
        return child, tuple(records)

    def activate(
        self,
        inputs: Sequence[float] | np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Evaluate directly; rng is accepted for Cell compatibility but never used.

        Validation failures raise ValueError. Numerical divergence raises
        FloatingPointError before an activation can hide it by saturation.
        All execution values are local to this call.
        """
        raw = np.asarray(inputs)
        if raw.shape != (self.input_size,) or raw.dtype.kind not in "iuf":
            raise ValueError(
                f"inputs must be a one-dimensional vector of {self.input_size} real numbers"
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                array = np.array(raw, dtype=np.float64, copy=True)
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError("inputs must be finite float64 values") from exc
        if not np.isfinite(array).all():
            raise ValueError("inputs must be finite float64 values")

        values = {i: float(value) for i, value in enumerate(array)}
        for node, incoming in zip(self._order, self._incoming):
            total = node.bias
            for edge in incoming:
                total += edge.weight * values[edge.source]
            if not math.isfinite(total):
                raise FloatingPointError(f"non-finite weighted sum at node {node.id}")
            value = _ACTIVATIONS[node.activation](total)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite activation at node {node.id}")
            values[node.id] = value
        return np.array(
            [values[node_id] for node_id in self.output_ids], dtype=np.float64
        )
