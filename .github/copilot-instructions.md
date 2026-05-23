# Project Context & Guidelines

## 1. Instructions for AI Assistant
Act as a "Co-pilot" for a busy developer.
- VERY IMPORTANT: Suggest the *smallest possible step* to move forward.
- VERY IMPORTANT: DO NOT make any code changes immediately without confirming the direction is OK by the user. ALWAYS ASK FIRST.
- VERY IMPORTANT: DO NOT rush. DO NOT make multiple changes in one go. Each change should be assigned to individual commit.
- Conversation in chat should be in Japanese, but all of code comments should be in English.
- Provide complete, copy-pasteable code for tests and visualization.
- Do not suggest major architectural changes unless requested.

## 2. Project Identity
- **Name:** cell-based-open-evo-sim
- **Core Concept:** Open-ended evolution of ALife in a 3D environment (currently 2D dev).
- **Key Architecture:**
    - **Genome (CPPN)** generates structure/behavior (meaning-agnostic).
    - **Interpreter** assigns meaning based on context (**Profile**).
    - **Implicit Fitness:** Evolution driven purely by survival, no explicit reward functions.
    - **Emergent Semantics Strategy:**
        - **Genome:** Pure structure generator (raw vectors, no fixed semantics).
        - **Profile:** Context derived during development (e.g., "Muscle", "Neuron").
        - **Interpreter:** Maps Genome output to Semantics (Actions) *depending on* the Profile.
        - **Goal:** Allow the same genetic structure to function differently in different contexts (Exaptation).
        - **Evolution:** Start with discrete preset Profiles, then evolve towards continuous context.
- **Reference Docs:**
    - `README.md`: General overview.
    - `doc/ProjectGrandDesign.md`: Core architectural definitions (Genome, Interpreter, Profile, Agent).
    - `doc/roadmap_ja.md`: **The authoritative roadmap.**

## 3. Development Philosophy (Strict Rules)
1.  **Test-First & Determinism:**
    - Every feature must be reproducible.
    - Determinism (same seed = same result) is the highest priority.
    - Write tests for every new logic.
    - Whenever changes are made, run all tests to ensure no regressions.
2.  **Slow & Steady:**
    - This is a hobby project with limited time.
    - **Do not rush.** Avoid complex libraries (like physics engines) until absolutely necessary.
    - Code maintainability > New features.
3.  **Architecture First:**
    - Validate logic (Profile, Messaging) before Physics.
    - Physics will be introduced late, starting with simple soft-collision.

## 4. Current Status & Immediate Focus
- **Current Implementation State:**
    - Basic concepts such as `Cell`, `World`, `Agent`, `Profile` and `Interpreter` structure exist.
    - **No physics/collision yet.**

## 5. Short-term Roadmap (Sequential)
- *Follow `doc/roadmap_ja.md`.*
- If there are any other tasks that should be tackled or if any existing tasks in the roadmap should be reordered in oder to proceed this project while maintaining its maintainability, testability (e.g. logging, determinism, visualization), and flexibility, suggest any time.
