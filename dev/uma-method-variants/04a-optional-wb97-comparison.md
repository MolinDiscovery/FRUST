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

Pending.
