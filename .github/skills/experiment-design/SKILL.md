---
name: experiment-design
description: Design one small artificial-life experiment for an approved project question or task. Use when defining an I/O or inheritance contract, choosing controls and budgets, or deciding how to distinguish a mechanism demonstration from evolutionary evidence. Does not implement or run the experiment.
---

# Design one bounded experiment

## Context

Read `.github\copilot-instructions.md`, the relevant accepted design and roadmap sections, and the existing code needed to establish feasibility. Resolve repository paths from the repository root, not this skill directory.

Use the user's selected question or task. If none was selected, consult the roadmap to identify the next eligible step and propose only that step. Do not silently advance the roadmap.

## Procedure

1. State one question and the kind of evidence sought: mechanism, numerical behavior, inheritance, selection response, or ecological behavior. Do not substitute a benchmark for the project's purpose.
2. Identify prerequisites that exist, missing capabilities, and decisions requiring user approval. Distinguish facts from proposals.
3. Define only the parts of the contract needed for that task, such as initialization and reset, inputs and coordinate frames, raw outputs and interpretation, Profile generation, State ownership, Actions, topology distinctions, and inherited information. Do not force unrelated lifecycle or I/O decisions into the current scope.
4. When development or behavior is in scope, keep them in one Genome + Interpreter framework. Define applicable observation windows without requiring non-overlapping life stages or a separate controller.
5. Declare fixed and variable factors. Follow accepted staged constraints on Genome, Interpreter, and Profile rules; do not vary everything at once.
6. Choose only controls and comparable budgets needed for the question. Do not let an unrelated unresolved design decision block a diagnostic that does not depend on it.
7. Specify logical limits, seeds, measurements, failure outcomes, and artifacts. Wall-clock limits are operational safeguards; do not use timeout as silent selection pressure.
8. Define what observations would support or weaken the hypothesis and what the experiment cannot establish. Do not promise improvement, statistical significance, exaptation, or OEE.

For physical experiments, establish a local Action and environmental reaction that can make the body matter. Do not prescribe a solver, final morphology, or successful motion script without approval.

Reuse the existing harness where appropriate. If a missing capability is necessary, propose its smallest separate change; do not design a general experiment DSL or require an unrelated general evolutionary framework.

## Output

Respond in Japanese with a compact contract, normally around one page:

| Item | Required content |
|---|---|
| Scope and question | Approved task or scope, one question, evidence level |
| Existing / missing | Relevant implementation references and prerequisites |
| Contract | Inputs, outputs, ownership, inheritance/reset, observation window as applicable |
| Fixed / variable | Factors held constant and the one change being studied |
| Controls | Purpose of each baseline, or why none is needed for this task |
| Budget | Applicable seeds and logical resource limits; separate timeout |
| Evidence | Measurements, artifacts, failure categories, acceptance/comparison method |
| Limits | What this experiment does not establish |
| Approval needed | Unresolved decisions and the smallest proposed implementation step |

Make only the items needed by the selected task concrete, and reuse existing approved contracts. Leave decisions assigned to later work explicitly unresolved rather than pulling them into the current scope.

Ask one material unresolved question at a time using the available user-question tool. Present the proposal before making repository changes. Do not implement, execute, commit, or save a design document unless that action has been explicitly approved. Once the requested contract is delivered, stop.
