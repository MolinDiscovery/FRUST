# 01 — Define the stage and result contracts

## Goal

Make the three paths in the [plan](README.md) selectable without changing
existing UMA screening or ωB97 runs. Decide the public options and saved
result meanings before adding calculators.

## Work

1. Inspect the existing `MethodPlan`, workflow factories, stage graph, result
   tiers, analysis selectors, portable-run review, resume manifests, and
   reference fingerprints. Write a compact stage table for each path showing
   its input, calculator, selection point, and output columns.
2. Choose a small public API for (a) final method, (b) screening method, and
   (c) optional ranking method. Keep the existing screening choices and the
   `include_dft_rank_sp` behavior compatible. State exactly how a user asks
   for gas or ALPB UMA, UMA final characterization, and g-xTB → UMA SP → ωB97.
3. Give UMA ranking its own result tier, such as `uma_ranked`, with a clear
   energy and geometry source. Preserve `dft_ranked` for an actual DFT
   ranking SP. Define `low_cost`, `uma_ranked`, and method-specific `full`
   status and selection semantics. A UMA `full` result must say UMA, including
   in filenames, stage columns, `ft.show_steps`, manifests, and analysis.
4. Specify separate candidate limits before and after UMA SP reranking.
   Define how several geometrically diverse candidates reach the focused TS
   refinement path and how their identities survive filtering and collection.
5. Add the minimal stage-plan and result-contract code needed by later tasks.
   Keep chemistry target expansion unchanged and `targets()` inexpensive.
   Record the chosen names and options below, with one runnable construction
   example using `import frust as ft`.

## Acceptance

- The three paths have unambiguous stage sequences and saved tiers. No UMA
  calculation is written into a `dft_*` column or presented as independent DFT
  validation.
- Existing UMA screening and ωB97 calls retain their behavior and can read
  existing result bundles. New options, selections, and reference identities
  survive submit, collect, resume, and local execution.
- Focused tests check stage ordering, tier labels, both candidate limits, and
  local/cluster plan equivalence without running a new chemistry calculation.

## Completion record

Pending. Record the exact public API, stage/result table, test command and
result, and any compatibility decision here before starting Task 02.
