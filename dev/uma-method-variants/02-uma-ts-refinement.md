# 02 — Add UMA TS refinement

## Goal

Turn retained TS guesses into unconstrained UMA stationary points with
numerical frequencies. Keep the constrained UMA optimization before the TS
search so the rest of each candidate can settle.

## Work

1. Reuse or extend the existing `uma_opt` constrained screening stage. Accept
   several retained candidates per chemical target, keep their parent row and
   guess/constraint profile, and preserve structural diversity through the
   handoff. Do not silently choose only the lowest screening energy.
2. Release the reactive-core constraints after `uma_opt`. Build a UMA Hessian
   or numerical-frequency seed as needed by ORCA, then run UMA `OptTS` and a
   final UMA numerical frequency. Decide whether the seed Hessian can be
   reused and record the decision in the saved stage plan.
3. Use the same UMA model and environment for SP, constrained Opt, TS search,
   and frequency displacements. In ALPB mode the GFN2-xTB chloroform
   correction must affect both energies and gradients at every geometry;
   inspect the generated ORCA files to confirm the flag is present.
4. Reuse one job-scoped UMA server across all stages and numerical-frequency
   calls in a submitted job. Ensure failure and cancellation stop it. Do not
   start a second server when switching from optimization to NumFreq.
5. Carry the final geometry, frequencies, displacement vectors, and
   provenance into the portable result. Apply the existing TS quality checks:
   one imaginary mode is necessary, and its displacement must be reviewable
   against the intended reaction coordinate. Keep a `review` result when the
   mode is ambiguous; a calculator exit code alone is insufficient.

## Acceptance

- The path is `UMA Opt [C] → release constraints → UMA Hessian/OptTS → UMA
  NumFreq`, with distinct UMA stage and result names selected in Task 01.
- Multiple candidates can be refined and traced to their source guesses.
  Gas and ALPB variants use the requested potential consistently.
- Focused tests cover constraint release, stage order, server reuse and
  cleanup, saved frequency data, and quality flags. Use mocks or saved data
  here; Task 05 is the bounded live chemistry check.

## Completion record

Completed 2026-09-30 on `feature/uma-screening`. Full UMA TS refinement is
available in `ft.workflows.screen_ts(...)` with `method="uma-gas"` or
`method="uma-alpb-chloroform"` and `calculation_level="full"`. The composed
`ft.workflows.catalyst_screen(...)` accepts `ts_refine_n`; its UMA reference
and final analysis branches remain Task 03.

```python
import frust as ft

wf = ft.workflows.screen_ts(
    dataframe=systems,
    ts_types=["TS1"],
    method="uma-gas",
    calculation_level="full",
    top_n=10,
    ts_refine_n=3,
)
wf.show_stages()[["stage", "constraint", "lowest"]]
```

After `uma_opt`, an RMSD prune on `uma_opt-oc` preserves distinct geometries,
then `uma_refine_filter` keeps up to `ts_refine_n` per structure by
`uma_opt-EE`. Existing `structure_id` and `cid` values continue through all
stages. The screening `top_n` and refinement `ts_refine_n` are recorded in
workflow provenance. `auto` guess-profile selection resolves to the reviewed
`omol-uma-s-1p2p1/gas` profile for either UMA calculation environment; the
resolved profile is saved with the result. No ALPB geometry profile is claimed.

The released path is `uma_hessian` (`ExtOpt NumFreq`, retrieving `input.hess`)
→ `uma_ts_opt` (`ExtOpt OptTS`, `constraint=False`, reading that Hessian) →
`uma_freq` (`ExtOpt NumFreq`). The seed Hessian is reused for the initial
`OptTS` search, then final frequencies are recalculated at the optimized
geometry. The same pinned `omol@uma-s-1p2p1` model and gas or ALPB setting is
required at all UMA stages. A mixed screening/refinement environment is
rejected. UMA's Stepper path now passes the Hessian reuse flag to ORCA, and
generated input includes `--xtb-alpb chloroform` when selected. A live ORCA
compatibility check remains Task 05.

The existing job-scoped server covers the complete stage group, including
both `NumFreq` calls, and closes on success, failure, or cancellation. Final
`uma_freq-vibs` vectors and `uma_freq-frequencies_cm1` values are preserved in
portable full UMA TS results, including compact screening artifacts, so the
imaginary mode remains inspectable. Existing TS quality rules require one
imaginary frequency and keep unreviewed modes in `review`.

Checks: focused mock tests cover stage order, candidate identities, constraint
release, Hessian handoff, ALPB input, saved mode data, and vibration validity.
Existing UMA server tests cover single-server reuse and cleanup. The full fast
suite passed: 380 tests; 13 slow tests were deselected. No live chemistry was
run for this task.
