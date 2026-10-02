# 06 — Document the UMA method choices

## Goal

Teach users the four workflow choices and the optional ωB97 comparison. Base
examples on the implemented public API and the verified functional checks.

## Work

1. Update the relevant workflow, catalyst-screen, UMA, and vibration guides.
   Start with a compact input and the four stage flows, then explain the
   method choices. Show focused calls using `import frust as ft` and current
   `ft.workflows` names.
2. Give a small result table showing representative stage columns, energy
   source, geometry source, `low_cost`, `uma_ranked`, `dft_ranked`, and `full`
   tier meanings. State which path gives an UMA barrier and which gives an
   ωB97 barrier. Show how to inspect mode review and missing thermochemistry.
3. Explain gas and ALPB choices, the separate guess/constraint profile,
   candidate limits before and after ranking, and why the focused TS path
   retains `UMA Opt [C]` before releasing constraints. Show how to inspect
   each candidate's barrier and quality, and how optional ωB97 comparison
   pairs independently calculated results.
4. Keep cluster job IDs, node names, repository revisions, repair history,
   and test anecdotes in this development plan. The user guide should show
   inputs, expected outputs, and scientific interpretation. Do not imply that
   the bounded functional check establishes benchmark accuracy.
5. Verify example calls against the completed API and run
   `conda run -n UMA mkdocs build --strict`. Fix broken links and examples.

## Acceptance

- A reader can select each path, identify the saved method at each stage,
  understand which result tiers exist, and tell whether a barrier needs mode
  review.
- The docs use the current public API, concrete FRUST input/output examples,
  and accurate gas/ALPB profile language. Strict MkDocs build passes.

## Completion record

**Complete, 2026-10-02.** The four paths now lead the
[UMA method choices guide](../../docs/catalyst-screens/uma-screening.md) with
the N-methylpyrrole/TMP boron catalyst input, runnable constructors, stage
flows, candidate limits, a method-labelled TS1 result table, tier meanings,
quality values, mode review, missing thermochemistry inspection, and the
optional independently calculated ωB97 comparison. Related explanations were
aligned in the [workflow method guide](../../docs/workflows/workflow-methods.md),
[end-to-end catalyst screen](../../docs/catalyst-screens/end-to-end.md),
[catalyst overview](../../docs/catalyst-screens/overview.md),
[UMA lower-level guide](../../docs/external-tools/uma.md),
[vibrations](../../docs/visualization/vibrations.md), and
[result inspection](../../docs/workflows/inspecting-results.md).

The four example `ft.workflows.catalyst_screen(...)` calls were instantiated
with the documented CSV and their `show_stages()`/`targets()` outputs checked
without calculators. The g-xTB path included `uma_rank_sp` and no `uma_opt`;
full UMA paths ended in `uma_hessian`, `uma_ts_opt`, and `uma_freq`.
`conda run -n UMA mkdocs build --strict` passed. The bounded calculation
is a functional check, not an accuracy benchmark; the gas UMA example remains
marked invalid because its ligand reference has an imaginary mode.
