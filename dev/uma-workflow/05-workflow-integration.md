# 05 — Integrate UMA screening with ωB97 validation

> **Planning update, 2026-09-30:** A method-specific UMA `tsguess2` profile is
> not a prerequisite for screening. The existing ωB97 gas profile supplies the
> initial TS geometry and row-level constraints, just as it does for g-xTB
> screening in a ωB97 workflow. The reviewed UMA gas profile is also available
> as an explicit choice. UMA then evaluates and optimizes those rows. Keep the
> chosen guess/constraint profile explicit in result provenance. The UMA ALPB
> profile remains optional later work; do not imply an ALPB-optimized reference
> when using gas UMA or ωB97 constraints. The exact
> placement of any ALPB single point after gas UMA optimization was resolved
> by using the same ALPB-corrected UMA potential for both screening stages.

## Goal

Make UMA a first-class screening choice in the existing dataframe-first
catalyst workflow, with clear stage names and results. The full calculation
uses the established ωB97 validation method plan after UMA selection.

## Expected behavior

| Requested level for an UMA screen | Stages after constrained UMA optimization |
| --- | --- |
| `low_cost` | Stop and report UMA screening results. |
| `dft_ranked` | Run the ωB97 ranking single point. |
| `full`, ranking SP enabled | Run the ωB97 ranking SP, then full validation. |
| `full`, ranking SP disabled | Proceed directly to ωB97 full validation. |

The ranking-SP switch for `full` is `include_dft_rank_sp`. It defaults to
`False` for UMA and `True` for g-xTB. `dft_ranked` always runs the ranking SP.
Existing g-xTB/r2SCAN-3c/ωB97 workflows retain their current semantics.

## Work

1. Add UMA screening choices for gas phase and ALPB(chloroform): GFN-FF
   preoptimization, UMA single points for selection, and constrained UMA
   optimization. Use the existing ωB97 profile for guesses and constraints
   unless a reviewed UMA profile is explicitly selected. Keep the chosen
   potential and geometry profile separately visible in provenance.
2. Use `uma_sp` and `uma_opt` or similarly unambiguous stage/result labels.
   Carry these names through snapshots, manifests, `ft.show_steps`, and
   analysis rather than writing UMA data into `xtb_*` columns.
3. Select candidates per system using the UMA results while retaining the
   current workflow's configurable conformer limits and diversity handling.
4. Connect the UMA screening plan to the ωB97 validation plan for local and
   submitted runs. Provide an explicit way to include or skip `dft_rank_sp`
   in a full UMA run.
5. Test stage ordering, environment/profile matching, method provenance,
   filtering, resume/collection, and expected result columns. Use small
   inputs and mocks where these are enough to test orchestration; do not
   perform the scientific benchmark here.

## Acceptance

- A user can choose gas-phase or ALPB(chloroform) UMA screening through a
  documented public workflow API, with an explicit, available guess/constraint
  profile; selecting the UMA potential never implies an unreviewed UMA profile.
- The requested levels and both `full` ranking-SP settings follow the table
  above; `full` uses the same target chemistry locally and on the cluster.
- UMA output names and provenance identify the model and selected environment.
- Existing screening and validation presets continue to pass their relevant
  tests.

## Completion record

Implemented on `feature/uma-screening` in FRUST revision `e228b05`. The public
API accepts `screening="uma-gas"` or
`screening="uma-alpb-chloroform"` in `ft.workflows.catalyst_screen(...)` and
the individual molecule, TS, and INT3 factories. For example:

```python
wf = ft.workflows.catalyst_screen(
    dataframe=components,
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,
)
```

The default TS/INT3 guess and constraint profile follows the ωB97 validation
method. The reviewed UMA gas profile can be selected explicitly with
`spec_profile="omol-uma-s-1p2p1/gas"`; it is never inferred from the screening
potential. The selected profile is recorded as `guess_profile`, while the UMA
model and ALPB environment remain in the calculator specification.

| Mode | Stage sequence after GFN-FF preoptimization | Saved lower tiers |
| --- | --- | --- |
| UMA `low_cost` | `uma_sp` → `uma_sp_filter` → `uma_opt` | Final result uses `uma_opt-EE` and `uma_opt-oc`. |
| UMA `dft_ranked` | UMA stages → `dft_rank_sp` → final filter | `tier_low_cost.parquet`; final result uses `dft_rank_sp-EE`. |
| UMA `full`, default | UMA stages → ωB97 refinement | `tier_low_cost.parquet`; no DFT-ranked tier. |
| UMA `full`, ranking enabled | UMA stages → `dft_rank_sp` → `dft_rank_filter` → ωB97 refinement | Low-cost and DFT-ranked tiers. |

The small mocked TS run in `tests/test_uma_screening_workflow.py` confirmed
that an input with three conformers selected the lowest UMA single-point row
before constrained UMA optimization. The same test module checks tier files,
semantic columns, profile resolution, reference fingerprints, and the full
coordinator stage graph including molecule references and INT3.

Checks: `conda run -n UMA python -m pytest tests/test_uma_screening_workflow.py -q`
(7 passed); `conda run -n UMA python -m pytest -q` (363 passed, 13 deselected);
`conda run -n UMA mkdocs build --strict` (passed). A real compute-node
end-to-end run and final user guide review remain in Task 06. The scientific
screening benchmark remains outside this task sequence.
