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
   `include_dft_rank_sp` behavior compatible. Specify the intended gas/ALPB
   and hybrid workflow calls; Tasks 02–04 make those calls executable.
3. Give UMA ranking its own result tier, such as `uma_ranked`, with a clear
   energy and geometry source. Preserve `dft_ranked` for an actual DFT
   ranking SP. Define `low_cost`, `uma_ranked`, and method-specific `full`
   status and selection semantics. A UMA `full` result must say UMA, including
   in filenames, stage columns, `ft.show_steps`, manifests, and analysis.
4. Specify separate candidate limits before and after UMA SP reranking.
   Define how several geometrically diverse candidates reach the focused TS
   refinement path and how their identities survive filtering and collection.
5. Add the minimal calculator-plan and result-contract code needed by later
   tasks. Keep chemistry target expansion unchanged and `targets()`
   inexpensive. Record the chosen names and options below, with one runnable
   plan-construction example using `import frust as ft`.

## Acceptance

- The three paths have unambiguous stage sequences and saved tiers. No UMA
  calculation is written into a `dft_*` column or presented as independent DFT
  validation.
- Existing UMA screening and ωB97 calls retain their behavior and can read
  existing result bundles. The new ranking calculator and full-method family
  serialize with stable fingerprints for later submit/resume integration.
- Focused tests check the semantic stage labels and tier columns without a
  chemistry calculation. Tasks 02–04 test executable ordering, both candidate
  limits, reference identity, and local/cluster parity when those stages are
  connected to the workflows.

## Completion record

Completed 2026-09-30 on `feature/uma-screening`. The executable factory API
still accepts the existing `screening`, `method`, `level`, and
`include_dft_rank_sp` arguments. The following names are reserved for the
remaining tasks:

| Choice | Public spelling | Meaning |
| --- | --- | --- |
| Screening | `screening="gxtb-default"`, `"uma-gas"`, or `"uma-alpb-chloroform"` | Existing low-cost method and environment. |
| Optional intermediate ranking | `ranking=None`, `"uma-gas"`, or `"uma-alpb-chloroform"` | New `uma_rank_sp` after g-xTB constrained optimization; Task 04 connects it. |
| Final method | `method="wb97xd3-631g"`, `"uma-gas"`, or `"uma-alpb-chloroform"` | Existing ωB97 validation or new UMA characterization; Tasks 02–03 add UMA presets and stages. |
| Requested depth | `level="low_cost"`, `"uma_ranked"`, `"dft_ranked"`, or `"full"` | `uma_ranked` is a distinct electronic-energy tier; `full` follows the selected final method. |
| Selection limits | `top_n`, `uma_rank_top_n`, `ts_refine_n` | Existing low-cost retention, candidates passed from UMA SP to ωB97, and UMA TS candidates retained after constrained optimization. The latter two are added with their executable paths. |

For the first implementation, UMA ranking and `include_dft_rank_sp=True`
should be mutually exclusive in a full ωB97 run. Existing `dft_ranked` keeps
its DFT SP meaning. A full UMA run uses `include_dft_rank_sp=False`.

The calculator plan is already constructible:

```python
import frust as ft

ranking = ft.workflows.methods.ranking_preset("uma-alpb-chloroform")
assert ranking.stage_id == "uma_rank_sp"
assert ranking.calculator.solvent == "chloroform"
```

The intended future workflow calls are
`ft.workflows.catalyst_screen(..., screening="uma-gas", method="uma-gas",
level="full", ts_refine_n=3)` and
`ft.workflows.catalyst_screen(..., screening="gxtb-default",
ranking="uma-alpb-chloroform", method="wb97xd3-631g", level="full",
top_n=20, uma_rank_top_n=3)`. These are API targets for Tasks 02–04, not yet
executable calls.

| Path/tier | Input geometry | Energy and selection | Output geometry | Frequency |
| --- | --- | --- | --- | --- |
| UMA `low_cost` | GFN-FF constrained optimization | `uma_sp-EE` filter, then `uma_opt-EE` winner | `uma_opt-oc` | None |
| UMA `full` TS | Retained `uma_opt-oc` candidates | `uma_hessian`/`uma_ts_opt`, analysis `uma_freq-EE` | `uma_ts_opt-oc` | `uma_freq-GE` and `uma_freq-frequencies_cm1` |
| UMA `full` minimum | Screened reference geometry | `uma_min_opt`, analysis `uma_freq-EE` | `uma_min_opt-oc` | `uma_freq-GE` and `uma_freq-frequencies_cm1` |
| Hybrid `low_cost` | GFN-FF constrained optimization | g-xTB SP filter and `xtb_opt-EE` winner | `xtb_opt-oc` | None |
| Hybrid `uma_ranked` | `xtb_opt-oc` candidates | `uma_rank_sp-EE` winner | `xtb_opt-oc` | None |
| Hybrid `full` | UMA-selected g-xTB geometry | ωB97 refinement, normally `dft_solv_sp-EE` | `dft_ts_opt-oc` or `dft_opt-oc` | `dft_freq` |

Path C is the multi-candidate TS branch of UMA `full`, with no separate
analysis tier. Each candidate keeps its `structure_id`/`cid` and parent
guess provenance through filtering; Task 02 implements that handoff.

`frust.results.result_contract` now resolves `uma_ranked` and UMA `full`
columns without using `dft_*` names. `RankingPlan` represents an independent
gas or ALPB UMA SP calculator and has its own fingerprint. `MethodPlan` now
has a `result_family` field; existing DFT plans omit this field from their
serialized form, preserving their fingerprints and cached-reference identity.
`tier_uma_ranked.parquet` is reserved as the intermediate snapshot filename.
Tasks 02–04 will connect these contracts to stage execution, manifests,
reference protocols, and collection.

Checks: `conda run -n UMA python -m pytest tests/test_uma_variant_contracts.py
tests/test_uma_screening_workflow.py tests/test_workflow_methods.py -q` passed
(31 tests). `conda run -n UMA python -m pytest -q` passed (374 tests, 13 slow
tests deselected). No new chemistry calculation was run.
