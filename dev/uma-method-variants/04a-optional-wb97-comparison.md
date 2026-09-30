# 04a — Optionally compare UMA and ωB97 final barriers

## Goal

Let a user request ωB97 final characterization for selected candidates from
an UMA-final workflow, including the required reference minima. Report UMA
and ωB97 barriers side by side without mixing their energies.

## Intended result

| TS candidate | UMA ΔG‡ (kcal/mol) | ωB97 ΔG‡ (kcal/mol) | UMA status | ωB97 status |
| --- | ---: | ---: | --- | --- |
| `result_a...` | 18.2 | 20.5 | ready | review |

The example values are illustrative. The comparison should identify both
method protocols, the candidate mapping, solvent choices, and the reference
results used by each barrier.

## Work

1. Choose a small opt-in public option for the comparison and define whether
   it starts from one selected UMA TS candidate or an explicit set. It must
   leave the four base workflow choices and their default results unchanged.
2. Run the established ωB97 constrained preoptimization, Hessian, OptTS,
   frequency, and chloroform SP path for the selected TS candidates. Compute
   the corresponding ωB97 references with the existing identity and reuse
   rules. Preserve a link to the parent UMA candidate through collection and
   restart, even if ωB97 optimization changes the geometry.
3. Calculate ωB97 barriers solely from ωB97 TS/reference results, and UMA
   barriers solely from UMA results. Compare matched system, position,
   TS type, candidate, stoichiometry, and reference definitions. Report a
   missing or invalid side honestly rather than borrowing the other method's
   energy or quality.
4. Test local and submitted plans, one- and several-candidate selection,
   reference reuse, missing results, method provenance, and portable restart
   with mocked calculations.

## Acceptance

- The comparison is optional and produces independently labelled UMA and
  ωB97 barriers for matched candidates.
- Existing UMA-only and ωB97-only runs keep their current behavior and
  portable bundles remain readable.

## Completion record

Implemented as an explicit follow-on from a completed full-UMA run:

```python
import frust as ft

uma_run = ft.screen.open_run("runs/uma")
wf = ft.workflows.wb97_comparison(uma_run, candidates="selected")
comparison = wf.run(out_dir="runs/uma_wb97")
table = comparison.method_comparison()
print(table[[
    "parent_uma_result_id", "uma_delta_g_kcal_mol",
    "wb97_delta_g_kcal_mol", "uma_quality_status",
    "wb97_quality_status", "match_status",
]])
```

Pass `candidates=[result_id_1, result_id_2]` to characterize several exact
rows from `uma_run.candidate_barriers()`. `wf.submit(...)` uses the same
snapshotted UMA seeds and stage graph as `wf.run(...)`. Each seed starts at its
UMA `uma_ts_opt-oc` geometry and keeps the selected reactive-core constraints
for ωB97 preoptimization. The DFT branch then runs Hessian, released `OptTS`,
frequency, and chloroform single point. ωB97 reference minima use the existing
DFT reference identity, publication, snapshot, and reuse path.

The comparison bundle saves the parent UMA candidate barriers, formulas,
method settings, and exact seed geometries. The `parent_uma_result_id` persists
through the ωB97 result and analysis, even if geometry and result ID change.
`method_comparison()` joins by parent candidate and target identity; UMA and
ωB97 energies, statuses, solvents, protocols, and reference IDs are separate.
It reports missing ωB97 TS or reference results and flags changed atom formulas.
Moving or reopening the comparison bundle does not require the original UMA
run for analysis.

Validation: five mocked comparison tests cover one and multiple candidates,
local and staged submission, ωB97 reference reuse, missing TS/reference
results, method provenance, compact artifacts, and reopening after relocation.
The UMA fast suite passed with `407 passed, 13 deselected` on 2026-09-30.
No live chemistry was launched;
bounded compute-node checks remain Task 05.
