# 04 — Rerank g-xTB candidates with UMA single points

## Goal

Provide the hybrid path that keeps the current g-xTB screening stages, uses
UMA SP to choose which constrained g-xTB geometries advance, then runs the
established ωB97 full characterization.

## Work

1. Insert UMA SP **after** g-xTB constrained optimization. Retain a broad,
   diverse set at the g-xTB filter, evaluate UMA at those optimized
   geometries, and apply a separate post-UMA selection limit. There is no
   UMA optimization in this variant.
2. Implement the `uma_ranked` tier and result selection specified in Task
   01. Preserve both g-xTB and UMA energies and the optimized geometry used
   by the UMA SP, with clear stage names and provenance.
3. Allow gas or ALPB UMA ranking explicitly. Keep ωB97 refinement and
   reference calculations unchanged after selection. Mark the final result
   as ωB97 while showing that UMA made the candidate-selection decision.
4. Ensure the stage graph, submit manifests, cached results, resume behavior,
   and `ft.show_steps` agree for local and cluster execution. Test ties,
   missing UMA SP values, and the two selection limits on a small saved or
   mocked candidate set.

## Acceptance

- The stage order is `GFN-FF Opt [C] → g-xTB SP/filter → g-xTB Opt [C] →
  UMA SP/rerank → ωB97 full`.
- A requested UMA ranking tier is distinct from `dft_ranked`; the final
  ωB97 tier still uses the existing DFT stage names and analysis.
- Existing g-xTB-only screening, UMA screening, and ωB97 validation paths
  retain their prior defaults and pass focused regression tests.

## Completion record

Pending. Record public options, stage/tier example, tests, and any migration
note here before starting Task 05.
