# 06 — Document the UMA method choices

## Goal

Teach users when to choose UMA final characterization, UMA ranking after
g-xTB, or a focused multi-candidate UMA TS search. Base examples on the
implemented public API and the verified functional checks.

## Work

1. Update the relevant workflow, catalyst-screen, UMA, and vibration guides.
   Start with a compact input and the three stage flows, then explain the
   method choices. Show focused calls using `import frust as ft` and current
   `ft.workflows` names.
2. Give a small result table showing representative stage columns, energy
   source, geometry source, `low_cost`, `uma_ranked`, `dft_ranked`, and `full`
   tier meanings. State which path gives an UMA barrier and which gives an
   ωB97 barrier. Show how to inspect mode review and missing thermochemistry.
3. Explain gas and ALPB choices, the separate guess/constraint profile,
   candidate limits before and after ranking, and why the focused TS path
   retains `UMA Opt [C]` before releasing constraints.
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

Pending. Record updated pages, example verification, build result, and any
remaining scientific limitation here.
