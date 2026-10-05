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

Completed 2026-09-30 on `feature/uma-screening`. The hybrid path is
selectable with an independent UMA ranking plan and two candidate limits:

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    dataframe=components,
    ts_types=["TS1"],
    screening="gxtb-default",
    ranking="uma-alpb-chloroform",  # or "uma-gas"
    method="wb97xd3-631g",
    level="full",
    top_n=20,
    uma_rank_top_n=3,
)
wf.show_stages()[["branch", "stage", "lowest", "solvent"]]
```

The active order is initial RMSD prune → constrained GFN-FF Opt → g-xTB SP
→ `xtb_sp_filter` (broad `top_n`) → constrained `xtb_opt` → `uma_rank_sp`
on `xtb_opt-oc` → `uma_rank_filter` (`uma_rank_top_n`) → the established
ωB97 refinement. There is no `uma_opt` in this path. Missing/non-finite
UMA energies cannot win the post-UMA filter; ties retain stable input order.
The gas or ALPB(chloroform) choice is explicit in the ranking plan and saved
manifest. `ft.show_steps(df)` exposes both filter cutoffs without widening
the main calculation dataframe.

| Tier | Analysis energy | Geometry | Meaning |
| --- | --- | --- | --- |
| `low_cost` | `xtb_opt-EE` | `xtb_opt-oc` | Independently selected g-xTB screen result |
| `uma_ranked` | `uma_rank_sp-EE` | `xtb_opt-oc` | Independently selected UMA electronic result |
| `full` | `dft_solv_sp-EE` for the default ωB97 plan | `dft_ts_opt-oc` or `dft_opt-oc` | ωB97 stationary points, frequencies, and barrier |

The portable run contains all three tiers, and
`run.compare_barriers()` keeps their source and units separate. The full
result remains labelled DFT/ωB97; `ranking_method` in `run.summary()` and
the manifest show that UMA made the earlier selection. Reference identity
includes the UMA ranking fingerprint and cutoff, so gas and ALPB references
cannot be interchanged. Low-cost references can still be reused when their
g-xTB protocol matches. Resume signatures include the ranking and cutoff.
Existing g-xTB-only, UMA-screened, and ωB97 runs retain their defaults and
need no bundle migration.

Mocked local and submitted tests cover gas and ALPB plans and results, both
selection limits, ties, missing UMA energies, stage order and input geometry,
reference identity, saved tiers, restart, and incompatible options. The fast
UMA suite passed: 402 tests, 13 slow tests deselected. Live ORCA/UMA checks
remain Task 05.
