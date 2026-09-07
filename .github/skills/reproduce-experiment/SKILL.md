---
name: reproduce-experiment
description: Reproduce or compare an existing simulation experiment and preserve metrics, events, frames, and file-based visualizations. Use for reruns, seed comparisons, regression reproduction, or regenerating an animation from a trusted run. Does not add missing experiment infrastructure automatically.
---

# Reproduce an existing experiment

## Establish scope

Read `.github\copilot-instructions.md` and `.github\instructions\experiments.instructions.md`. Resolve paths from the repository root. Identify the requested experiment, original configuration/artifacts, expected comparison, and approved budget.

Inspect the relevant entry point before running it:

| Need | Existing entry points |
|---|---|
| Generic experiment | `experiments\common\experiment_spec.py`, `experiments\common\runner_generic.py` |
| Existing scene/config | `experiments\chemotaxis_bud\config.py`, `experiments\chemotaxis_bud\runner.py` |
| Quick API | `experiments\run_experiment.py` |
| Existing regression coverage | `tests\test_experiment_harness.py`, `tests\test_determinism.py`, `tests\test_animation_smoke.py` |
| Saved-run animation | `scripts\make_animation_from_run.py` |
| Output contracts | `experiments\common\frame_dumper.py`, `experiments\common\event_logger.py`, `tests\conftest.py` |

Do not assume an importable module is a CLI, invent command-line options, or run a large default scenario just to discover its behavior. Running an approved existing experiment is not permission to modify source, install global dependencies, or change its success criteria.

## Execute and record

1. Establish the effective config, all RNG sources, initial-state/reset behavior, and deterministic work limits. Check genomes, positioners, and factories for global randomness. If the necessary contract is missing, ask for it or use `experiment-design` rather than choosing silently.
2. Record the repository revision and relevant uncommitted state, exact command, working directory, interpreter/library versions, seeds, sampling settings, and budget. Use existing metadata outputs where supported; otherwise include this information in the handoff or an approved run-local metadata file.
3. Use the active Python environment from the repository root. A targeted test invocation uses `python -m pytest -q` followed by existing test paths/selectors. For headless plotting, set `MPLBACKEND=Agg` only for this execution context and preserve any prior setting if the shell persists.
4. Allocate a separate output directory per run. Reuse test fixtures for pytest outputs. Do not delete or overwrite the source run.
5. For repeatability requests, run independently initialized instances with the same declared configuration and seeds. For comparisons, keep budgets and all non-target factors constant. Do not automatically launch a sweep.
6. Record completion or failure explicitly. A timeout, missing dependency, swallowed required hook failure, invalid input, or numerical divergence must not be reported as a successful reproduction or silently converted to fitness.
7. Check the required artifacts against the actual writer and sampling contract. For recorded animations, inspect the script's documented inputs and output arguments before invoking it. Only load trusted object-array/pickled recordings.
8. Inspect numerical data as well as images. Any required but unavailable lineage, restart, initial-state, or limit information remains a stated limitation until separately implemented; this skill does not supply it.

Determine whether the selected runner and its events describe Cells, Agents, or another unit. Do not infer lineage from aggregate population size or net count differences; inspect event producers before interpreting an event as biological reproduction.

## Compare meaningfully

- Compare simulation state and relevant metrics with the agreed numerical tolerances.
- Exclude wall-clock timing and output-directory timestamps from state equality. Treat them as separate operational metadata.
- IDs can vary, but identity relationships may affect neighbor ties, messaging, or inheritance. Use an explicit identity correspondence where justified; do not sort positions alone and claim full-world or lineage equivalence.
- Verify that independent runs do not share mutable Cells, genotypes, or RNG state.
- Label hand-coded demonstrations, inherited variation, and selected behavior separately. Do not infer exaptation or OEE from a visually interesting run.
- If evidence differs, preserve both runs and report the earliest useful discrepancy; propose a minimal reproducer before changing implementation or tolerances.

## Handoff

Respond in Japanese with the result or blocker first, followed by the exact reproducible command/config, seeds, code/environment identity, artifact paths, comparison method, and limitations relevant to the request. If only regenerating a visualization, distinguish that from rerunning or validating the simulation.

Provide complete runnable commands or code when requested, not pseudocode with invented APIs. Do not commit outputs or source changes. Stop after delivering the requested reproduction or comparison.
