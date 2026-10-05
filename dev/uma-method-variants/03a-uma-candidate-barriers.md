# 03a — Report a UMA barrier for each TS candidate

## Goal

Complete workflow choice 4. Each retained, refined UMA TS candidate should
have its own auditable barrier and mode quality, while the existing selected
`run.barriers()` result remains one row per system and TS type.

## Intended result

For a full UMA run with `ts_refine_n=3`, expose
`run.candidate_barriers()` with a table such as:

| ts_type | ts_result_id | cid | selected | delta_g_kcal_mol | quality_status |
| --- | --- | ---: | --- | ---: | --- |
| TS1 | `result_a...` | 0 | True | 18.2 | ready |
| TS1 | `result_b...` | 1 | False | 20.1 | review |
| TS1 | `result_c...` | 2 | False | — | invalid |

The example values are illustrative. The table must also retain ΔE‡,
reference IDs, method/model/solvent fingerprint, guess profile, and quality
issues. `selected` identifies the candidate represented by `run.barriers()`.

## Work

1. Add a public candidate-barrier inspection helper and portable table with
   one row per retained UMA TS result. Keep `run.barriers()` and its existing
   one-row selection stable for callers and tier comparisons.
2. Apply the same balanced equations, exact reference selection, energy
   protocol checks, missing-data rules, and kcal/mol units to each candidate.
   Include invalid and incomplete candidates with explicit quality and empty
   energies where the equation cannot be evaluated. A `review` candidate
   remains reviewable until its imaginary mode is approved.
3. Persist each candidate's result ID and parent `cid` so a manual mode review
   updates the right candidate after refresh or restart. Make the selected
   candidate rule explicit and deterministic.
4. Test multiple candidates, approval, bad frequencies, mixed methods,
   missing Gibbs/reference data, portable restart, and compatibility with
   existing DFT bundles using mocked results.

## Acceptance

- A full UMA run can inspect ΔE‡, ΔG‡, quality, and provenance separately for
  every refined candidate. Acceptable candidates can be compared directly.
- `run.barriers()` retains one selected row, and existing DFT analysis and
  `compare_barriers()` keep their established shape.

## Completion record

Completed 2026-09-30 on `feature/uma-screening`. A full UMA run writes
`analysis/candidate_barriers.parquet` alongside the existing selected
`analysis/barriers.parquet`. For example:

```python
import frust as ft

run = ft.screen.open_run("uma_screen")
candidates = run.candidate_barriers()
candidates[[
    "ts_type", "ts_result_id", "cid", "selected",
    "delta_e_kcal_mol", "delta_g_kcal_mol", "quality_status",
]]
```

Each candidate uses the same balanced equation and selected reference rows
as the main barrier. `reference_result_ids` records the exact reference
results; `ts_energy_protocol_fingerprint` identifies the TS calculator even
when a mixed protocol invalidates the barrier. `energy_protocol_fingerprint`
is present only when every term matches. The table also carries the model,
solvent, guess profile, mode count, mode review status, and candidate-specific
quality issues. Missing references or Gibbs data, bad frequencies, mixed
protocols, and unbalanced formulas retain explicit quality rather than
silently hiding a candidate.

`run.barriers()` still contains one row per target. It selects the best
quality candidate, then the lowest available Gibbs or electronic energy,
then result ID; the candidate table marks that exact result with `selected`.
Approving an imaginary mode can change which candidate is selected. Old DFT
bundles return an empty candidate table, including when they lack the new
artifact, and `compare_barriers()` keeps its one-row-per-target shape.

Mocked end-to-end tests cover distinct candidate energies, approval,
candidate-specific bad modes and missing Gibbs, missing references, mixed
environments, balance, relocation, and DFT compatibility. The fast UMA suite
passed: 394 tests, 13 slow tests deselected. No live chemistry ran for this
task; bounded compute checks remain Task 05.
