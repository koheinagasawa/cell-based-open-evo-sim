# Project context and working agreement

## Purpose

This is a long-term artificial-life project: evolve diverse "creatures that might have existed" without hand-designing their final bodies or movement scripts. Humans define world laws, available operations, genetic representations, and experiment conditions. Open-ended evolution is a long-term direction, not the only valuable outcome. Do not replace this purpose with publication goals or benchmark optimization.

## Sources of truth

- `doc\ProjectGrandDesign.md`: design intent and long-term concepts, not an implementation inventory.
- `doc\roadmap_ja.md`: authoritative task order, dependencies, experiment gates, and non-goals.
- Source code and tests: evidence of current behavior; `README.md`: general introduction.
- Before design or feature work, read the relevant design and roadmap sections and inspect the affected code. Report conflicting or stale information; do not silently choose a different architecture or rewrite the roadmap.
- Follow the decision-recording rules below so future work does not depend on chat history.

## Decisions and project knowledge

- Keep design intent in `doc\ProjectGrandDesign.md`, and task order, dependencies, and gates in `doc\roadmap_ja.md`. Put concrete I/O, inheritance, and experiment contracts in the relevant design or experiment document. Prefer an existing appropriate document; create a new one only when the approved task needs it.
- For consequential decisions, briefly record the decision, its status, date and related task, rationale, important alternatives and why they were not chosen, and relevant constraints or conditions for revisiting it. Do not require a formal record for every routine implementation detail.
- Distinguish accepted decisions, proposals, unresolved questions, and superseded decisions. Never promote an agent suggestion or assumption to an accepted specification without agreement. Separate observations from interpretation and hypotheses, linking supporting artifacts when available.
- Once a decision is approved, updating the relevant documentation is part of completing that agreed task. Do not leave essential decisions only in chat or pause for separate approval of routine recordkeeping. Ask again if recording the decision would introduce an unapproved design change or materially expand scope.
- When a decision changes, make clear what it replaces and why. Update affected current contracts so old decisions do not remain apparently authoritative, and preserve useful rationale through a short history or reference. Read existing rationale before reopening a decision; new evidence or changed constraints can justify reconsideration.
- Keep one authoritative home for each decision and reference it elsewhere instead of duplicating its explanation across instructions, roadmaps, and notes. Historical reviews and conversation summaries are background, not current specifications.
- Do not assume session history or uncommitted/untracked notes will be available in another session or checkout. Put information required to continue in maintained repository documents and identify any pending documentation changes in the handoff. Recording does not authorize staging, committing, or pushing.
- Keep records proportional to the task. Do not automatically archive entire conversations, generate a handoff document after every exchange, or introduce a separate decision-log framework. Finish with the relevant documents updated and remaining material questions explicitly identified.

## Core design contract

- Genome is the CPPN itself, evaluated directly for each Cell. Do not assume a HyperNEAT-style CPPN that generates a separate controller's weights. Genotype storage and a compiled evaluator may be separate representations of the same network.
- Genome generates raw vectors; the directly encoded Interpreter maps them to next State and Actions using the Cell's Profile. Separating generation from interpretation creates routes to reuse existing capabilities and add uses; it does not guarantee stable or cumulative evolution.
- Use the same Genome + Interpreter framework for development and behavior. Observation windows may differ or overlap; a separate mature-body controller is not required.
- Share network parameters and Interpreter within an Agent; keep mutable execution State per Cell. Generate or select Profile at Cell birth and keep it fixed for that Cell's lifetime.
- Distinguish growth within an Agent from reproduction of a descendant Agent. Inherit Genome, Interpreter, and Profile generation/selection rules; normally redevelop descendants from seed Cells rather than copying adult bodies and current State.
- External evaluation and generation replacement are allowed experimental scaffolding. Keep them outside World laws and distinguish their results from autonomous reproduction and implicit selection.
- Distinguish existing validation scaffolding from the intended evolutionary and body-mediated paths. Preserve it unless the approved task replaces it, and do not treat a hand-coded or direct-control path as evidence for a different target mechanism.

## Slow and steady

- This is a hobby project with limited time. Prioritize sustainable, incremental progress over development speed or feature count.
- Do not rush. Complete and review one small logical change before starting the next; do not bundle independent improvements. A logical change may span related files; it is not a single file edit or tool call.
- Maintainability and understandability take priority over new features. Avoid complex libraries and speculative abstractions until a concrete need justifies them.

## Working agreement

- Discuss in Japanese; write code comments in English. Implement tests and visualizations as runnable repository files by default. Show full copy-and-paste code listings in chat only when explicitly requested.
- Agree on the smallest useful task before editing. Once approved, implement, run the relevant checks, and fix failures introduced by that change without pausing for each routine step. Ask again before expanding scope or making unapproved design, behavior, or dependency decisions.
- Complete the approved task in the repository rather than stopping at a proposal or sample code. Preserve unrelated behavior and report blockers or unrelated failures instead of silently expanding the task.
- Keep one logical change per commit when committing is requested. Do not stage, commit, push, or mark roadmap tasks complete merely because implementation was proposed.
- Work only in the assigned worktree. Preserve existing user edits and outputs.
- Add deterministic tests before implementing new logic. Run relevant tests during iteration and the existing full suite before handing off code changes. Documentation/instruction-only changes need metadata, path, and consistency checks rather than simulation runs unless executable behavior is affected.
- Reuse existing helpers and interfaces. Do not add a physics engine, general experiment framework, parallel execution, or backend migration without an approved need supported by the relevant experiment.
- Clarify unresolved experiment parameters instead of silently inventing defaults. Record failures explicitly; do not mask invalid input, numerical divergence, or missing artifacts with success-shaped fallback values.
- Separate observations from interpretation. Mechanism demonstrations, numerical correctness, heritable variation, selection response, and OEE are different claims.
- At handoff, briefly describe the actual changes, artifact locations, and unresolved limitations. Do not leave routine implementation steps for the user to perform manually when they are within the approved scope and the agent can complete them.

## Repository map and execution

| Area | Entry points |
|---|---|
| Cell and World | `simulation\cell.py`, `simulation\world.py`, `simulation\input_layout.py` |
| Interpretation and lifecycle | `simulation\interpreter.py`, `simulation\agent.py`, `simulation\policies.py`, `simulation\lifecycle.py` |
| Interactions | `simulation\physics`, `simulation\fields.py`, `simulation\messaging.py` |
| Experiment harness | `experiments\common\experiment_spec.py`, `experiments\common\runner_generic.py` |
| Tests and fixtures | `tests`, `tests\conftest.py`, `tests\utils` |
| Visualization | `scripts\make_animation_from_run.py`, `visualization\introspection.py` |

Run Python commands from the repository root using the active environment; do not hard-code machine-specific interpreter paths. The existing suite is `python -m pytest -q`; select relevant test files for narrower runs. For automated plotting, set `MPLBACKEND=Agg` for the command/session without changing the user's global configuration. Saved images and animations are sufficient; do not require a GUI.

Inspect `requirements-dev.txt` and actual imports before dependency changes. Do not assume the manifest covers every optional plotting dependency, silently install into a global interpreter, or change package sources.

## Task-specific guidance

- `.github\instructions\simulation.instructions.md`: simulation invariants.
- `.github\instructions\experiments.instructions.md`: experiments, tests, and visualization.
- `experiment-design` skill: turn one approved project question or task into a small experiment contract.
- `reproduce-experiment` skill: rerun an existing experiment and preserve comparable artifacts.
- `simulation-reviewer` agent: read-only review of a supplied change set and its intended behavior.

Use skills only for matching workflows and the reviewer for substantive review, not every small lookup. Keep detailed procedures in those files rather than duplicating them here.
