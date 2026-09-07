---
applyTo: "simulation/**/*.py"
---

# Simulation invariants

These rules supplement `.github\copilot-instructions.md`. Read the affected implementation and its tests; do not treat long-term design goals as existing functionality.

## State transitions and ownership contracts

- Before changing simulation scheduling, inspect the current implementation, tests, and applicable specification to identify when State becomes visible, when Actions and world effects apply, and when lifecycle, message, and field updates take effect. Preserve those contracts unless the approved task changes them; update the relevant documentation and tests together with an intentional contract change.
- Make state visibility and lifecycle timing explicit. Avoid exposing partially updated state or silently changing birth, death, maintenance, or communication timing; cover the intended behavior with deterministic tests.
- Reuse the applicable existing I/O definition and interpretation path instead of duplicating layout or ordering rules. Keep raw-vector I/O validation at the appropriate boundary.
- Network parameters may be shared within an Agent; mutable Cell State, recurrent execution state, and descendant genotype mutations must not leak between owners.
- Preserve the distinction between Profile values fixed at Cell birth and the inherited rules that generate/select them.
- Keep policy injection and optional feature paths behavior-compatible. Keep external experimental evaluation outside World.

## Determinism

- Make RNG ownership and seed derivation explicit. Avoid global `np.random` draws and shared mutable RNG state in new simulation paths.
- A fixed seed alone is insufficient: examine iteration order, neighbor ties, IDs, connection ordering, and spawn identity.
- Inspect how identities and derived seeds are formed before changing them. Reproduce the relevant collision or tie behavior, and do not replace an order-independent scheme with an order-dependent one unless the approved contract requires it.
- Test independent worlds and independent parent/child objects, not two containers sharing the same Cells.
- State the numerical comparison contract. Do not promise cross-platform bitwise equality or loosen tolerances merely to hide a regression.

## Physics and lifecycle boundaries

- Distinguish CPPN graph topology, body topology, physical bonds, and directed message links. Shared storage does not establish identical semantics.
- Distinguish direct-control scaffolding from body-mediated locomotion. An actuator intended to demonstrate body-mediated behavior must act through the approved body/environment contract.
- Use the chosen world laws to judge conservation and motion. Internal actuation alone need not translate a body's center of mass; lack of translation is not automatically a solver defect.
- Specify numerical budgets and check finite values for new solver behavior. PBD/XPBD and inertia are choices to justify, not universal requirements.
- Distinguish Cell budding from descendant Agent birth. Keep membership, bond/message cleanup, and lifecycle events consistent when changing either path.
- Reject invalid genotype/I/O and numerical failures explicitly. Do not copy existing broad exception swallowing into new code.
- Measure the relevant bottleneck before optimizing; preserve deterministic ordering and the agreed numerical error budget.
