# 05 — Integrate UMA screening with ωB97 validation

> **Planning update, 2026-09-30:** A method-specific UMA `tsguess2` profile is
> not a prerequisite for screening. The existing ωB97 gas profile supplies the
> initial TS geometry and row-level constraints, just as it does for g-xTB
> screening in a ωB97 workflow. The reviewed UMA gas profile is also available
> as an explicit choice. UMA then evaluates and optimizes those rows. Keep the
> chosen guess/constraint profile explicit in result provenance. The UMA ALPB
> profile remains optional later work; do not imply an ALPB-optimized reference
> when using gas UMA or ωB97 constraints. The exact
> placement of any ALPB single point after gas UMA optimization remains to be
> settled during this task.

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

The ranking-SP switch for `full` is independent of the requested level. Decide
its public name and UMA default before implementation; record the choice in
the index. Existing g-xTB/r2SCAN-3c/ωB97 workflows retain their current
semantics.

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

Pending. Add the public API example, FRUST revision, stage table from a small
run, and focused test results here.
