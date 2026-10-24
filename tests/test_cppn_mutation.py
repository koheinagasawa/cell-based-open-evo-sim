import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from simulation.cppn import (
    ConnectionGene,
    CPPNGenome,
    MutationConfig,
    MutationRecord,
    NodeGene,
)


def make_parent():
    return CPPNGenome(
        input_size=2,
        output_size=2,
        nodes=(
            NodeGene(2, "tanh", 0.25),
            NodeGene(3, "sin", -0.125),
            NodeGene(4, "identity", 0.5),
            NodeGene(5, "gaussian", 0.75),
        ),
        connections=(
            ConnectionGene(0, 2, 0.75),
            ConnectionGene(1, 2, -0.5),
            ConnectionGene(2, 3, 1.25),
            ConnectionGene(3, 4, -0.75),
        ),
        output_ids=(4, 3),
    )


def make_config(**overrides):
    values = {
        "weight_probability": 0.7,
        "bias_probability": 0.6,
        "activation_probability": 0.8,
        "weight_std": 0.125,
        "bias_std": 0.25,
    }
    values.update(overrides)
    return MutationConfig(**values)


@pytest.mark.parametrize(
    "field",
    [
        "weight_probability",
        "bias_probability",
        "activation_probability",
        "weight_std",
        "bias_std",
    ],
)
@pytest.mark.parametrize(
    "value", [-0.1, np.nan, np.inf, -np.inf, True, np.bool_(False), "0.1", None]
)
def test_mutation_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        make_config(**{field: value})


@pytest.mark.parametrize(
    "field", ["weight_probability", "bias_probability", "activation_probability"]
)
def test_mutation_probabilities_cannot_exceed_one(field):
    with pytest.raises(ValueError, match=field):
        make_config(**{field: 1.01})


def test_config_is_immutable_explicit_and_normalizes_numpy_scalars():
    with pytest.raises(TypeError):
        MutationConfig()
    config = make_config(
        weight_probability=np.float64(1.0),
        bias_probability=np.float32(0.0),
        weight_std=np.float64(1e308),
    )
    assert type(config.weight_probability) is float
    assert type(config.bias_probability) is float
    assert type(config.weight_std) is float
    with pytest.raises(FrozenInstanceError):
        config.activation_probability = 0.0


def test_mutation_matches_declared_rng_order_and_gaussian_addition():
    parent = make_parent()
    config = make_config()
    expected_rng = np.random.default_rng(37)
    expected_nodes = []
    expected_connections = []
    expected_records = []
    activations = ("gaussian", "identity", "sin", "tanh")
    for node in parent.nodes:
        bias = node.bias
        if expected_rng.random() < config.bias_probability:
            bias += float(expected_rng.normal(0.0, config.bias_std))
            expected_records.append(MutationRecord("bias", node.id, node.bias, bias))
        activation = node.activation
        if expected_rng.random() < config.activation_probability:
            alternatives = [name for name in activations if name != activation]
            activation = alternatives[int(expected_rng.integers(len(alternatives)))]
            expected_records.append(
                MutationRecord("activation", node.id, node.activation, activation)
            )
        expected_nodes.append(NodeGene(node.id, activation, bias))
    for edge in parent.connections:
        weight = edge.weight
        if expected_rng.random() < config.weight_probability:
            weight += float(expected_rng.normal(0.0, config.weight_std))
            expected_records.append(
                MutationRecord(
                    "weight", (edge.source, edge.target), edge.weight, weight
                )
            )
        expected_connections.append(ConnectionGene(edge.source, edge.target, weight))
    rng = np.random.default_rng(37)
    child, records = parent.mutate(config, rng)
    assert child.nodes == tuple(expected_nodes)
    assert child.connections == tuple(expected_connections)
    assert records == tuple(expected_records)
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_same_parent_config_and_seed_reproduce_child_records_and_rng_state():
    parent = make_parent()
    first_rng = np.random.default_rng(41)
    second_rng = np.random.default_rng(41)
    first, first_records = parent.mutate(make_config(), first_rng)
    second, second_records = parent.mutate(make_config(), second_rng)
    assert first == second
    assert first is not second
    assert first_records == second_records
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    np.testing.assert_array_equal(
        first.activate([0.5, -0.25]), second.activate([0.5, -0.25])
    )


def test_mutation_is_independent_of_graph_declaration_order():
    parent = make_parent()
    reordered = replace(
        parent,
        nodes=tuple(reversed(parent.nodes)),
        connections=tuple(reversed(parent.connections)),
    )
    first, first_records = parent.mutate(make_config(), np.random.default_rng(7))
    second, second_records = reordered.mutate(make_config(), np.random.default_rng(7))
    assert first.to_dict() == second.to_dict()
    assert first_records == second_records


@pytest.mark.parametrize(
    "config",
    [
        make_config(
            weight_probability=0.0, bias_probability=0.0, activation_probability=0.0
        ),
        make_config(
            weight_probability=1.0,
            bias_probability=1.0,
            activation_probability=0.0,
            weight_std=0.0,
            bias_std=0.0,
        ),
    ],
)
def test_disabled_mutation_returns_independent_copy_without_consuming_rng(config):
    parent = make_parent()
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    child, records = parent.mutate(config, rng)
    assert child == parent
    assert child is not parent
    assert all(a is not b for a, b in zip(parent.nodes, child.nodes))
    assert all(a is not b for a, b in zip(parent.connections, child.connections))
    assert records == ()
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("parameter", ["weight", "bias", "activation"])
def test_each_mutation_category_can_be_enabled_independently(parameter):
    parent = make_parent()
    config = make_config(
        weight_probability=float(parameter == "weight"),
        bias_probability=float(parameter == "bias"),
        activation_probability=float(parameter == "activation"),
    )
    child, records = parent.mutate(config, np.random.default_rng(17))
    assert child != parent
    assert len(records) == 4
    assert all(record.parameter == parameter for record in records)
    assert child.input_size == parent.input_size
    assert child.output_size == parent.output_size
    assert child.output_ids == parent.output_ids
    assert [node.id for node in child.nodes] == [node.id for node in parent.nodes]
    assert [(edge.source, edge.target) for edge in child.connections] == [
        (edge.source, edge.target) for edge in parent.connections
    ]
    if parameter != "weight":
        assert child.connections == parent.connections
    if parameter != "bias":
        assert [node.bias for node in child.nodes] == [
            node.bias for node in parent.nodes
        ]
    if parameter != "activation":
        assert [node.activation for node in child.nodes] == [
            node.activation for node in parent.nodes
        ]


@pytest.mark.parametrize("parameter", ["weight", "bias", "activation"])
def test_each_category_can_change_the_direct_raw_vector_output(parameter):
    parent = CPPNGenome(
        input_size=1,
        output_size=1,
        nodes=(NodeGene(1, "identity", 0.25),),
        connections=(ConnectionGene(0, 1, 0.5),),
        output_ids=(1,),
    )
    config = make_config(
        weight_probability=float(parameter == "weight"),
        bias_probability=float(parameter == "bias"),
        activation_probability=float(parameter == "activation"),
    )
    child, records = parent.mutate(config, np.random.default_rng(17))
    assert len(records) == 1
    assert not np.array_equal(child.activate([0.25]), parent.activate([0.25]))
    np.testing.assert_array_equal(parent.activate([0.25]), [0.375])


def test_activation_can_switch_to_every_other_supported_function():
    parent = make_parent()
    config = make_config(
        weight_probability=0.0, bias_probability=0.0, activation_probability=1.0
    )
    observed = {node.id: set() for node in parent.nodes}
    for seed in range(32):
        child, records = parent.mutate(config, np.random.default_rng(seed))
        assert len(records) == len(parent.nodes)
        for original, changed, record in zip(parent.nodes, child.nodes, records):
            assert changed.activation != original.activation
            assert record == MutationRecord(
                "activation", original.id, original.activation, changed.activation
            )
            observed[original.id].add(changed.activation)
    for node in parent.nodes:
        assert observed[node.id] == {"identity", "tanh", "sin", "gaussian"} - {
            node.activation
        }


def test_mutation_records_exactly_describe_changes_and_parent_stays_unchanged():
    parent = make_parent()
    saved = parent.to_dict()
    output = parent.activate([0.5, -0.25])
    child, records = parent.mutate(make_config(), np.random.default_rng(19))
    actual = []
    for before, after in zip(parent.nodes, child.nodes):
        if before.bias != after.bias:
            actual.append(MutationRecord("bias", before.id, before.bias, after.bias))
        if before.activation != after.activation:
            actual.append(
                MutationRecord(
                    "activation", before.id, before.activation, after.activation
                )
            )
    for before, after in zip(parent.connections, child.connections):
        if before.weight != after.weight:
            actual.append(
                MutationRecord(
                    "weight",
                    (before.source, before.target),
                    before.weight,
                    after.weight,
                )
            )
    assert records == tuple(actual)
    assert records
    assert parent.to_dict() == saved
    np.testing.assert_array_equal(parent.activate([0.5, -0.25]), output)
    restored = CPPNGenome.from_dict(json.loads(json.dumps(child.to_dict())))
    assert restored == child
    np.testing.assert_array_equal(
        restored.activate([0.5, -0.25]), child.activate([0.5, -0.25])
    )
    with pytest.raises(FrozenInstanceError):
        records[0].after = 99.0
    with pytest.raises(FrozenInstanceError):
        child.nodes[0].bias = 99.0


def test_rounding_away_a_perturbation_does_not_report_a_change():
    parent = CPPNGenome(
        input_size=1,
        output_size=1,
        nodes=(NodeGene(1, "identity", 1e308),),
        connections=(),
        output_ids=(1,),
    )
    config = make_config(
        weight_probability=0.0,
        bias_probability=1.0,
        activation_probability=0.0,
        bias_std=1e-300,
    )
    child, records = parent.mutate(config, np.random.default_rng(1))
    assert child == parent
    assert records == ()


@pytest.mark.parametrize("seed", [1, 3])
def test_nonfinite_mutation_fails_without_clipping_or_changing_parent(seed):
    parent = CPPNGenome(
        input_size=1,
        output_size=1,
        nodes=(NodeGene(1, "identity"),),
        connections=(ConnectionGene(0, 1, 1.7e308),),
        output_ids=(1,),
    )
    config = make_config(
        weight_probability=1.0,
        bias_probability=0.0,
        activation_probability=0.0,
        weight_std=1.7e308,
    )
    saved = parent.to_dict()
    with pytest.raises(FloatingPointError, match="weight"):
        parent.mutate(config, np.random.default_rng(seed))
    assert parent.to_dict() == saved


@pytest.mark.parametrize("rng", [None, 1, np.random.RandomState(0)])
def test_mutation_requires_an_explicit_numpy_generator(rng):
    with pytest.raises(ValueError, match="rng"):
        make_parent().mutate(make_config(), rng)


def test_invalid_config_argument_fails_before_consuming_rng():
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="config"):
        make_parent().mutate(None, rng)
    assert rng.bit_generator.state == before
