import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from simulation.agent import Agent
from simulation.cppn import ConnectionGene, CPPNGenome, NodeGene
from simulation.interpreter import ProfiledInterpreter, SlotBasedInterpreter

OUTPUT_IDS = tuple(range(20, 29))


def constant_nodes():
    return tuple(
        NodeGene(node_id, "identity", float(i)) for i, node_id in enumerate(OUTPUT_IDS)
    )


def make_genome(nodes=None, connections=(), output_ids=OUTPUT_IDS):
    return CPPNGenome(
        input_size=18,
        output_size=9,
        nodes=constant_nodes() if nodes is None else nodes,
        connections=connections,
        output_ids=output_ids,
    )


@pytest.mark.parametrize(("input_size", "output_size"), [(1, 1), (2, 3), (24, 12)])
def test_io_sizes_are_declared_by_the_caller(input_size, output_size):
    output_ids = tuple(range(input_size, input_size + output_size))
    genome = CPPNGenome(
        input_size=input_size,
        output_size=output_size,
        nodes=tuple(
            NodeGene(node_id, "identity", i) for i, node_id in enumerate(output_ids)
        ),
        connections=tuple(
            ConnectionGene(input_size - 1, node_id, i + 1)
            for i, node_id in enumerate(output_ids)
        ),
        output_ids=output_ids,
    )
    inputs = np.zeros(input_size)
    inputs[-1] = 2.0
    assert genome.input_size == input_size
    assert genome.output_size == output_size
    result = genome.activate(inputs)
    assert result.shape == (output_size,)
    np.testing.assert_array_equal(result, 2.0 + 3.0 * np.arange(output_size))
    with pytest.raises(ValueError, match="inputs"):
        genome.activate(np.zeros(input_size + 1))
    with pytest.raises(FrozenInstanceError):
        genome.input_size = input_size + 1
    with pytest.raises(FrozenInstanceError):
        genome.output_size = output_size + 1


@pytest.mark.parametrize("field", ["input_size", "output_size"])
@pytest.mark.parametrize("value", [0, -1, True, np.bool_(True), 1.5, "2", None])
def test_io_sizes_must_be_positive_integers(field, value):
    sizes = {"input_size": 2, "output_size": 1}
    sizes[field] = value
    with pytest.raises(ValueError, match=field):
        CPPNGenome(
            **sizes,
            nodes=(NodeGene(2, "identity"),),
            connections=(),
            output_ids=(2,),
        )


@pytest.mark.parametrize("output_ids", [(), (2, 3)])
def test_output_count_must_match_declared_size(output_ids):
    with pytest.raises(ValueError, match="1 output"):
        CPPNGenome(
            input_size=2,
            output_size=1,
            nodes=(NodeGene(2, "identity"), NodeGene(3, "identity")),
            connections=(),
            output_ids=output_ids,
        )


def test_node_ids_and_connections_follow_declared_input_boundary():
    with pytest.raises(ValueError, match="node id"):
        CPPNGenome(
            input_size=24,
            output_size=1,
            nodes=(NodeGene(20, "identity"),),
            connections=(),
            output_ids=(20,),
        )
    with pytest.raises(ValueError, match="source"):
        CPPNGenome(
            input_size=2,
            output_size=1,
            nodes=(NodeGene(20, "identity"),),
            connections=(ConnectionGene(17, 20, 1.0),),
            output_ids=(20,),
        )
    with pytest.raises(ValueError, match="target"):
        CPPNGenome(
            input_size=24,
            output_size=1,
            nodes=(NodeGene(24, "identity"),),
            connections=(ConnectionGene(24, 20, 1.0),),
            output_ids=(24,),
        )


def test_differently_sized_genomes_do_not_change_each_others_layout():
    small = CPPNGenome(
        input_size=2,
        output_size=1,
        nodes=(NodeGene(2, "identity"),),
        connections=(ConnectionGene(1, 2, 3.0),),
        output_ids=(2,),
    )
    experiment = make_genome()
    np.testing.assert_array_equal(small.activate([0.0, 2.0]), [6.0])
    np.testing.assert_array_equal(experiment.activate(np.zeros(18)), np.arange(9))
    np.testing.assert_array_equal(small.activate([0.0, 2.0]), [6.0])


def test_constant_outputs_have_fixed_shape_and_declared_order():
    genome = make_genome(output_ids=tuple(reversed(OUTPUT_IDS)))
    result = genome.activate(np.zeros(18))
    assert isinstance(result, np.ndarray)
    assert result.shape == (9,)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.arange(8, -1, -1))


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("identity", 0.75),
        ("tanh", np.tanh(0.75)),
        ("sin", np.sin(0.75)),
        ("gaussian", np.exp(-(0.75**2))),
    ],
)
def test_activation_applies_after_weighted_sum_and_bias(activation, expected):
    nodes = (NodeGene(18, activation, 0.25), *constant_nodes())
    connections = (
        ConnectionGene(0, 18, 2.0),
        ConnectionGene(1, 18, -0.5),
        ConnectionGene(18, 20, 3.0),
    )
    inputs = np.zeros(18)
    inputs[:2] = (0.5, 1.0)
    result = make_genome(nodes, connections).activate(inputs)
    np.testing.assert_allclose(result[0], 3.0 * expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(result[1:], np.arange(1, 9))


def test_topological_evaluation_and_sum_order_ignore_declaration_order():
    nodes = (NodeGene(40, "identity"), NodeGene(19, "identity"), *constant_nodes())
    connections = (
        ConnectionGene(0, 40, 1e16),
        ConnectionGene(1, 40, -1e16),
        ConnectionGene(2, 40, 1.0),
        ConnectionGene(40, 19, 2.0),
        ConnectionGene(19, 20, 3.0),
    )
    inputs = np.ones(18)
    forward = make_genome(nodes, connections).activate(inputs)
    reversed_order = make_genome(
        tuple(reversed(nodes)), tuple(reversed(connections))
    ).activate(inputs)
    np.testing.assert_array_equal(forward, reversed_order)
    assert forward[0] == 6.0


def test_evaluator_does_not_consume_rng_or_retain_execution_state():
    genome = make_genome(connections=(ConnectionGene(0, 20, 2.0),))
    inputs_a = np.ones(18)
    inputs_b = np.full(18, 7.0)
    original = inputs_a.copy()
    rng = np.random.default_rng(12)
    rng_before = rng.bit_generator.state
    first = genome.activate(inputs_a, rng=rng)
    genome.activate(inputs_b, rng=rng)
    again = genome.activate(inputs_a, rng=rng)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(inputs_a, original)
    assert rng.bit_generator.state == rng_before
    assert not np.shares_memory(first, again)
    first[:] = -123.0
    np.testing.assert_array_equal(genome.activate(inputs_a), again)
    independent = make_genome(connections=(ConnectionGene(0, 20, 2.0),))
    np.testing.assert_array_equal(independent.activate(inputs_a), again)


def test_genome_captures_immutable_parameter_sequences():
    nodes = list(constant_nodes())
    edges = [ConnectionGene(0, 20, 2.0)]
    output_ids = list(OUTPUT_IDS)
    genome = make_genome(nodes, edges, output_ids)
    expected = genome.activate(np.ones(18))
    nodes.clear()
    edges.clear()
    output_ids.clear()
    np.testing.assert_array_equal(genome.activate(np.ones(18)), expected)
    assert isinstance(genome.nodes, tuple)
    assert isinstance(genome.connections, tuple)
    assert isinstance(genome.output_ids, tuple)
    with pytest.raises(FrozenInstanceError):
        genome.nodes = ()
    with pytest.raises(FrozenInstanceError):
        genome.nodes[0].bias = 99.0
    with pytest.raises(FrozenInstanceError):
        genome.connections[0].weight = 99.0


@pytest.mark.parametrize(
    "inputs",
    [
        np.zeros(17),
        np.zeros(19),
        np.zeros((1, 18)),
        np.zeros((18, 1)),
        np.zeros(18, dtype=bool),
        np.zeros(18, dtype=complex),
        ["0"] * 18,
        np.zeros(18, dtype=object),
        [np.nan] + [0.0] * 17,
        [np.inf] + [0.0] * 17,
        [-np.inf] + [0.0] * 17,
        None,
    ],
)
def test_invalid_inputs_are_rejected(inputs):
    with pytest.raises(ValueError, match="input"):
        make_genome().activate(inputs)


@pytest.mark.parametrize(
    ("nodes", "edges", "outputs", "message"),
    [
        (
            (*constant_nodes(), NodeGene(20, "identity")),
            (),
            OUTPUT_IDS,
            "duplicate node",
        ),
        ((*constant_nodes(), NodeGene(0, "identity")), (), OUTPUT_IDS, "node id"),
        ((*constant_nodes(), NodeGene(-1, "identity")), (), OUTPUT_IDS, "node id"),
        ((*constant_nodes(), NodeGene(18.5, "identity")), (), OUTPUT_IDS, "node id"),
        ((*constant_nodes(), NodeGene(True, "identity")), (), OUTPUT_IDS, "node id"),
        ((*constant_nodes(), NodeGene(18, "unknown")), (), OUTPUT_IDS, "activation"),
        ((*constant_nodes(), NodeGene(18, None)), (), OUTPUT_IDS, "activation"),
        ((*constant_nodes(), NodeGene(18, "identity", np.nan)), (), OUTPUT_IDS, "bias"),
        ((*constant_nodes(), NodeGene(18, "identity", "1")), (), OUTPUT_IDS, "bias"),
        (constant_nodes(), (ConnectionGene(0, 20, np.inf),), OUTPUT_IDS, "weight"),
        (constant_nodes(), (ConnectionGene(0, 20, True),), OUTPUT_IDS, "weight"),
        (constant_nodes(), (ConnectionGene(999, 20, 1.0),), OUTPUT_IDS, "source"),
        (constant_nodes(), (ConnectionGene(0, 999, 1.0),), OUTPUT_IDS, "target"),
        (constant_nodes(), (ConnectionGene(20, 0, 1.0),), OUTPUT_IDS, "target"),
        (constant_nodes(), (ConnectionGene(False, 20, 1.0),), OUTPUT_IDS, "source"),
        (constant_nodes(), (ConnectionGene(0, 20.0, 1.0),), OUTPUT_IDS, "target"),
        (
            constant_nodes(),
            (ConnectionGene(0, 20, 1.0), ConnectionGene(0, 20, 2.0)),
            OUTPUT_IDS,
            "duplicate connection",
        ),
        (constant_nodes(), (), OUTPUT_IDS[:-1], "9 output"),
        (constant_nodes(), (), (*OUTPUT_IDS, 20), "9 output"),
        (constant_nodes(), (), (*OUTPUT_IDS[:-1], 20), "duplicate output"),
        (constant_nodes(), (), (*OUTPUT_IDS[:-1], 999), "output"),
        (constant_nodes(), (), (*OUTPUT_IDS[:-1], 0), "output"),
        (constant_nodes(), (), (*OUTPUT_IDS[:-1], 28.0), "output"),
        (("not a node",), (), OUTPUT_IDS, "NodeGene"),
        (constant_nodes(), ("not an edge",), OUTPUT_IDS, "ConnectionGene"),
    ],
)
def test_invalid_genotype_is_rejected(nodes, edges, outputs, message):
    with pytest.raises(ValueError, match=message):
        make_genome(nodes, edges, outputs)


@pytest.mark.parametrize(
    "edges",
    [
        (ConnectionGene(20, 20, 1.0),),
        (ConnectionGene(20, 21, 1.0), ConnectionGene(21, 20, 1.0)),
        (ConnectionGene(18, 19, 1.0), ConnectionGene(19, 18, 1.0)),
    ],
)
def test_cycles_are_rejected_even_outside_output_ancestry(edges):
    nodes = (*constant_nodes(), NodeGene(18, "identity"), NodeGene(19, "identity"))
    with pytest.raises(ValueError, match="cycle"):
        make_genome(nodes, edges)


@pytest.mark.parametrize("activation", ["identity", "tanh", "sin", "gaussian"])
def test_nonfinite_weighted_sum_fails_even_if_activation_could_saturate(activation):
    nodes = (NodeGene(18, activation), *constant_nodes())
    edges = (ConnectionGene(0, 18, 1e308), ConnectionGene(18, 20, 1.0))
    inputs = np.full(18, 2.0)
    with pytest.raises(FloatingPointError, match="node 18"):
        make_genome(nodes, edges).activate(inputs)


def test_shared_genome_uses_existing_cell_and_profiled_interpreter_vector_path():
    interpreter = ProfiledInterpreter(
        {
            "default": SlotBasedInterpreter(
                {
                    "state": slice(0, 4),
                    "bud": slice(4, 7),
                    "contraction": 7,
                    "substrate_resistance": 8,
                }
            )
        },
        default_profile="default",
    )
    genome = make_genome(connections=(ConnectionGene(2, 20, 2.0),))
    agent = Agent(genome, interpreter, id="agent")
    first = agent.spawn_cell([0, 0], id="first", profile="default")
    second = agent.spawn_cell([1, 0], id="second", profile="default")
    # Phase 2 will build these 18 inputs from the body; Cell.sense is unchanged here.
    inputs_a = np.zeros(18)
    inputs_b = np.zeros(18)
    inputs_a[2] = 1.0
    inputs_b[2] = 3.0
    first.act(inputs_a)
    second.act(inputs_b)
    assert first.genome is second.genome is genome
    assert first.interpreter is second.interpreter is interpreter
    assert isinstance(first.raw_output, np.ndarray)
    np.testing.assert_array_equal(first.raw_output, [2, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(first.next_state, [2, 1, 2, 3])
    np.testing.assert_array_equal(second.next_state, [6, 1, 2, 3])
    np.testing.assert_array_equal(first.state, np.zeros(4))
    np.testing.assert_array_equal(second.state, np.zeros(4))
    np.testing.assert_array_equal(first.output_slots["bud"], [4, 5, 6])
    assert first.output_slots["contraction"] == 7.0
    assert first.output_slots["substrate_resistance"] == 8.0
    assert not np.shares_memory(first.next_state, second.next_state)
    first.next_state[0] = -1.0
    assert second.next_state[0] == 6.0


def make_serialization_genome():
    return CPPNGenome(
        input_size=2,
        output_size=2,
        nodes=(
            NodeGene(20, "identity", -0.25),
            NodeGene(9, "gaussian", 0.75),
            NodeGene(5, "sin", -0.5),
            NodeGene(2, "tanh", 0.125),
        ),
        connections=(
            ConnectionGene(5, 9, 0.25),
            ConnectionGene(2, 5, -2.0),
            ConnectionGene(1, 2, -0.75),
            ConnectionGene(0, 2, 1.5),
        ),
        output_ids=(9, 5),
    )


def test_serialization_declares_only_versioned_genotype_data():
    assert make_serialization_genome().to_dict() == {
        "schema_version": 1,
        "input_size": 2,
        "output_size": 2,
        "nodes": [
            {"id": 2, "activation": "tanh", "bias": 0.125},
            {"id": 5, "activation": "sin", "bias": -0.5},
            {"id": 9, "activation": "gaussian", "bias": 0.75},
            {"id": 20, "activation": "identity", "bias": -0.25},
        ],
        "connections": [
            {"source": 0, "target": 2, "weight": 1.5},
            {"source": 1, "target": 2, "weight": -0.75},
            {"source": 2, "target": 5, "weight": -2.0},
            {"source": 5, "target": 9, "weight": 0.25},
        ],
        "output_ids": [9, 5],
    }


@pytest.mark.parametrize("genome", [make_serialization_genome(), make_genome()])
def test_standard_json_file_round_trip_preserves_genotype_and_outputs(tmp_path, genome):
    path = tmp_path / "genotype.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(genome.to_dict(), stream, allow_nan=False)
    with path.open(encoding="utf-8") as stream:
        restored = CPPNGenome.from_dict(json.load(stream))
    assert restored == genome
    assert restored is not genome
    for value in (-1.0, 0.0, 0.75):
        inputs = np.full(genome.input_size, value)
        np.testing.assert_array_equal(
            restored.activate(inputs), genome.activate(inputs)
        )


def test_serialization_is_independent_of_node_and_connection_declaration_order():
    genome = make_serialization_genome()
    reordered = replace(
        genome,
        nodes=tuple(reversed(genome.nodes)),
        connections=tuple(reversed(genome.connections)),
    )
    assert reordered.to_dict() == genome.to_dict()
    data = genome.to_dict()
    data["nodes"].reverse()
    data["connections"].reverse()
    assert CPPNGenome.from_dict(data).to_dict() == genome.to_dict()


def test_serialization_normalizes_numpy_parameters_to_json_native_values():
    genome = CPPNGenome(
        input_size=np.int64(1),
        output_size=np.int64(1),
        nodes=[NodeGene(np.int64(1), "identity", np.float32(0.25))],
        connections=[ConnectionGene(np.int64(0), np.int64(1), np.float64(1.5))],
        output_ids=[np.int64(1)],
    )
    restored = CPPNGenome.from_dict(json.loads(json.dumps(genome.to_dict())))
    assert restored == genome


def test_exported_data_and_restored_genome_do_not_share_mutable_storage():
    parent = make_serialization_genome()
    expected = parent.to_dict()
    data = parent.to_dict()
    child = CPPNGenome.from_dict(data)
    data["nodes"][0]["bias"] = 0.875
    data["connections"][0]["weight"] = -0.5
    data["output_ids"].reverse()
    changed_child = CPPNGenome.from_dict(data)
    data["nodes"].clear()
    data["connections"].clear()
    data["output_ids"].clear()
    assert parent.to_dict() == child.to_dict() == expected
    assert changed_child.nodes[0].bias == 0.875
    assert changed_child.connections[0].weight == -0.5
    assert changed_child.output_ids == (5, 9)
    assert not np.array_equal(
        changed_child.activate([0.5, -0.25]), parent.activate([0.5, -0.25])
    )


def test_copy_is_an_independent_immutable_genotype():
    parent = make_serialization_genome()
    expected = parent.activate([0.5, -0.25])
    child = parent.copy()
    assert child == parent
    assert child is not parent
    assert all(a is not b for a, b in zip(parent.nodes, child.nodes))
    assert all(a is not b for a, b in zip(parent.connections, child.connections))
    changed_child = replace(
        child, nodes=(replace(child.nodes[0], bias=0.875), *child.nodes[1:])
    )
    assert not np.array_equal(changed_child.activate([0.5, -0.25]), expected)
    np.testing.assert_array_equal(parent.activate([0.5, -0.25]), expected)
    np.testing.assert_array_equal(child.activate([0.5, -0.25]), expected)
    with pytest.raises(FrozenInstanceError):
        child.nodes[0].bias = 99.0
    with pytest.raises(FrozenInstanceError):
        child.connections[0].weight = 99.0


@pytest.mark.parametrize("data", [None, [], "genotype", 1])
def test_restore_rejects_non_dictionary_data(data):
    with pytest.raises(ValueError, match="genotype"):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize("section", ["genotype", "nodes", "connections"])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_restore_rejects_missing_and_unknown_fields(section, operation):
    data = make_serialization_genome().to_dict()
    record = data if section == "genotype" else data[section][0]
    if operation == "missing":
        record.pop(next(iter(record)))
    else:
        record["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1", None])
def test_restore_rejects_unsupported_or_mistyped_schema_version(version):
    data = make_serialization_genome().to_dict()
    data["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize("field", ["nodes", "connections", "output_ids"])
@pytest.mark.parametrize("value", [None, {}, "invalid", ()])
def test_restore_requires_json_arrays(field, value):
    data = make_serialization_genome().to_dict()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize("field", ["nodes", "connections"])
@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_restore_requires_dictionary_records(field, value):
    data = make_serialization_genome().to_dict()
    data[field][0] = value
    with pytest.raises(ValueError, match=field):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("genotype", "input_size", True, "input_size"),
        ("genotype", "output_size", 3, "output"),
        ("genotype", "output_ids", [9, 9], "duplicate output"),
        ("genotype", "output_ids", [9, 999], "output"),
        ("nodes", "id", 0, "node id"),
        ("nodes", "activation", "unknown", "activation"),
        ("nodes", "bias", float("nan"), "bias"),
        ("nodes", "bias", "0.0", "bias"),
        ("connections", "source", 999, "source"),
        ("connections", "target", 0, "target"),
        ("connections", "weight", float("inf"), "weight"),
        ("connections", "weight", True, "weight"),
    ],
)
def test_restore_reuses_genotype_validation(section, field, value, message):
    data = make_serialization_genome().to_dict()
    record = data if section == "genotype" else data[section][0]
    record[field] = value
    with pytest.raises(ValueError, match=message):
        CPPNGenome.from_dict(data)


@pytest.mark.parametrize(
    "invalidity", ["duplicate node", "duplicate connection", "cycle"]
)
def test_restore_rejects_invalid_graph_topology(invalidity):
    data = make_serialization_genome().to_dict()
    if invalidity == "duplicate node":
        data["nodes"].append(dict(data["nodes"][0]))
    elif invalidity == "duplicate connection":
        data["connections"].append(dict(data["connections"][0]))
    else:
        data["connections"].append({"source": 9, "target": 2, "weight": 1.0})
    with pytest.raises(ValueError, match=invalidity):
        CPPNGenome.from_dict(data)
