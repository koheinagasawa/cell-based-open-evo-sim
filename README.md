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

