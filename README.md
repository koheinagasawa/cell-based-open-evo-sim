# open-evo-sim

This project is an experimental platform for simulating artificial life (ALife) through cell-level dynamics and evolution using NEAT (NeuroEvolution of Augmenting Topologies).

## Motivation

In natural organisms, a single genome gives rise to both morphology and behavior through development and neural activity. Inspired by this, this simulation uses **a single NEAT-evolved neural network** shared by all cells of an individual to govern both:

- **Morphogenesis**: how cells divide, move, and form structure
- **Behavior**: how cells contract, signal, or otherwise act in the environment

Each cell acts autonomously, receiving local inputs such as signals from neighbors or spatial cues, and applies the shared neural network to determine its own behavior in each time step. No global controller or predefined agent structure is assumed.

Over time, structures and motion emerge organically through interactions between cells and their environment. Evolution is driven by local survival and reproduction dynamics, without top-down objectives or reward functions.

## Goals

- Explore whether complex, life-like structures and behaviors can emerge purely from bottom-up, cell-level dynamics
- Investigate unified neural representations of development and control
- Create a minimal framework for open-ended evolution without centralized evaluation

## Status

Very early stage. Currently implementing the core building blocks of the system in Python. C++ backend for physics and high-performance simulation may be introduced later.

## License

MIT

## Details

This is a loose but coherent summary of the design principles. It’s meant to serve as the conceptual foundation of the open-evo-sim being developed.

### Core Components

#### Genome = Meaning-agnostic structure generator
- Uses indirect encoding (CPPN: Compositional Pattern Producing Network)
- Outputs raw vector fields with no explicit semantics
- Naturally supports spatial structure, symmetry, modularity
- Each input/output node is treated as a coordinate in “semantic space,” allowing position-dependent connection patterns to evolve
- Evolves via mutation/crossover at the Agent level

#### Interpreter = Context-dependent meaning assigner
- Uses direct encoding (slot assignment)
- Assigns semantics to input/output slots depending on cell context (profile)
- Handles both output interpretation and input construction (e.g., “this input slot receives neighbor state”)
- Evolves with the Agent

#### Profile = Fixed per-cell context derived during development
- Generated from Genome output field during development phase
- Stored as a continuous vector, not a predefined label
- Determines how the cell’s outputs are interpreted and how it interacts with the world or other cells
- Interpreters use profile to apply context-sensitive interpretation
- Profile cost is introduced to prevent all-purpose “super cells”
- Cost may be based on norm, slot count, functional overlap, or maintenance overhead
- Cost affects survival via energy drain or efficiency penalties

#### Agent = Evolutionary unit
- Contains Genome and Interpreter
- Reproduces via mating or self-replication
- Develops into a body composed of many cells
- Subject to evolutionary operations like mutation and crossover

#### Cell = Behavioral and interactive unit
- Holds mutable state and fixed profile
- Communicates with nearby cells, interacts with fields
- Actions are generated from Genome via Interpreter
- Collection of cells forms the morphology and functionality of the Agent

#### Field = Auxiliary continuous layer over physical space
- Defined on top of the physical simulation where cells exist
- Physics (collisions, constraints) handles mechanical interactions, while fields capture non-mechanical phenomena (e.g., diffusion, chemical gradients, heat, odor)
- Cells interact with fields by emitting signals (emit) or sensing gradients (sense), enabling indirect interaction channels
- Fields evolve with their own dynamics such as diffusion, decay, and propagation
- Multiple types of fields can coexist; Profile and Interpreter determine which fields a cell engages with and how
- Provides a substrate for behavioral diversity and exploration strategies, acting as a driver for evolutionary variation

### Long-Term Goal: Contextual Evolution System
- Genome outputs are meaningless until interpreted
- Interpreter assigns meaning based on local profile
- Profile is derived during development and starts without explicit meaning
- Meaning arises through interaction, selection, and evolutionary pressure
- Structure and semantics co-evolve in an open-ended fashion

