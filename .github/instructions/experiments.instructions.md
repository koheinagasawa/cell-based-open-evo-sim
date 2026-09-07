---
applyTo: "experiments/**/*.py,tests/**/*.py,scripts/**/*.py,visualization/**/*.py"
---

# Experiments, tests, and visualization

These rules supplement `.github\copilot-instructions.md`. Inspect and reuse the applicable experiment specifications, runners, hooks, writers, and test fixtures where they fit; do not turn their current names or layouts into permanent requirements.

## Reproducible setup

- Declare the question, fixed/variable factors, seed set, baseline, and logical run limits before interpreting results. Ask about unresolved choices that affect the experiment.
- Use explicit RNGs in every experiment component that draws randomness, not only in World. Inspect construction and callback behavior; a seed field on a specification does not seed arbitrary code.
- Rebuild independent objects for repeated runs. For inheritance experiments, distinguish inherited genotype/rules from reset Cell State and seed-body initialization.
- Keep external selection outside World. Do not make unrelated future capabilities prerequisites for a smaller diagnostic that does not need them.
- Prefer fixed steps/evaluation counts for comparable runs. Record wall-clock timeout as an operational interruption, not silently as fitness or deterministic selection.

## Tests and evidence

- Use small deterministic tests for new logic and regressions. Reuse applicable existing fixtures and output management; do not assume existing determinism tests cover every new path.
- Separate numerical correctness, repeatability, heritable variation, and selection response. Hand-coded examples are mechanism demonstrations, not evolved organisms.
- Match controls to the question: random/no-selection measures selection effects; mutation-off can examine inheritance and does not imply selection is impossible.
- Do not redefine success, increase limits, or select favorable seeds after seeing results without labeling a new experiment condition.
- Compare semantic state with declared tolerances, not whole output files byte-for-byte. Timing and directory timestamps are not simulation state. Preserve ID/lineage relationships; do not discard IDs that affect ordering or behavior.
- Performance gates are environment-sensitive. Record workload and timing separately; do not change thresholds just to obtain a pass.

## Artifacts and failure handling

- Follow the repository's applicable output-management contract for test and experiment artifacts. Keep distinct runs separate and do not overwrite previous evidence.
- Record effective configuration, commands, seeds, code revision plus relevant uncommitted changes, and runtime/library versions needed for reproduction.
- Verify the outputs required by the question are present and coherent with the recording and sampling settings. If required information or behavior is unavailable, state that limitation instead of claiming the runner supplies it.
- Treat extinction, invalid input, numerical divergence, missing artifacts, and hook failures explicitly and distinctly. Existing broad catches are not a pattern to extend.
- Save the visual and numerical artifacts required by the question using applicable existing helpers and formats. Automated rendering must not require a display. Verify data behind a plot rather than judging correctness from an attractive animation.
- Load pickled/object-array recordings only from trusted runs; do not deserialize arbitrary downloaded artifacts.
- Keep conclusions bounded by the controls and observations actually collected. Different action labels alone do not establish exaptation or cumulative evolution.
