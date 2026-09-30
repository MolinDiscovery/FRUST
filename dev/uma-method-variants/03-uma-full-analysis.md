# 03 — Complete UMA references and barrier analysis

## Goal

Make UMA the final method for an end-to-end result: characterize the TS and
the required reference minima at one potential, then calculate a clearly
labelled UMA electronic and free-energy barrier.

## Work

1. Add UMA unconstrained optimization and numerical frequencies for every
   reference species required by the existing barrier analysis. Preserve
   current reference stoichiometry, multiplicity, charge, and reuse rules.
   Include other states required by the supported full-cycle scope, or make
   an unsupported scope fail clearly rather than silently dropping states.
2. Extend energy and thermochemistry selectors to use the UMA TS and UMA
   reference results chosen in Task 01. Compute ΔE‡ and ΔG‡ only from a
   balanced, single-environment set of states. Do not combine UMA TS values
   with ωB97 reference values or reuse a reference across distinct UMA
   model/solvent fingerprints.
3. Verify the ORCA parser supplies the thermal quantities needed for ΔG‡
   after UMA NumFreq. If it does not, fix the parsing or report that ΔG‡ is
   unavailable; never fill it from electronic energies.
4. Apply the existing minimum and TS frequency checks to UMA results. Define
   `ready`, `review`, and `invalid` in terms of the actual UMA stationary
   points and mode review. Save calculator provenance and quality in the
   portable run bundle while keeping the main dataframe compact.
5. Test local run, submitted plan, collection, restart, missing reference,
   bad frequency, and mixed-environment rejection with small or saved inputs.

## Acceptance

- A UMA `full` request can produce UMA TS/reference geometries, NumFreq data,
  ΔE‡, and ΔG‡ in gas or ALPB when the required calculations succeed.
- The barrier source, units, frequency status, model, solvent correction, and
  selected guess profile are inspectable without mislabeling the result as
  ωB97/DFT. An incomplete or chemically invalid result is reported honestly.
- Existing ωB97 full analysis and old result bundles still work. Relevant
  focused tests pass in the `UMA` conda environment.

## Completion record

Completed 2026-09-30 on `feature/uma-screening`. A full UMA catalyst screen
uses the same pinned model and gas or ALPB(chloroform) correction for TS
refinement and every reference minimum:

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    dataframe=components,
    ts_types=["TS1"],
    screening="uma-gas",
    method="uma-gas",
    level="full",
    scope="barriers",
    ts_refine_n=3,
)
run = wf.run(out_dir="uma_screen", n_cores=8, mem_gb=32)
run.barriers()[[
    "ts_type", "delta_e_kcal_mol", "delta_g_kcal_mol", "quality_status"
]]
```

| State | Final geometry | Energy and thermochemistry | Frequency check |
| --- | --- | --- | --- |
| TS | `uma_ts_opt-oc` | `uma_freq-EE`, `uma_freq-GE`, `uma_freq-vibs` | Exactly one imaginary mode; its displacement needs review |
| Reference minimum | `uma_min_opt-oc` | `uma_freq-EE`, `uma_freq-GE`, `uma_freq-vibs` | No significant imaginary modes |

The portable `states` and `barriers` tables identify the UMA method family,
`omol@uma-s-1p2p1` model, solvent correction, selected guess profile, energy
protocol fingerprint, and quality. ΔE‡ and ΔG‡ use the existing balanced TS
equations and report kcal/mol. The analysis rejects mixed potential/solvent
fingerprints and unbalanced compositions. Missing reference or Gibbs data
leaves the corresponding result incomplete; Gibbs is never filled from
electronic energy. Several UMA TS candidates are ranked by quality and then
energy; approving the intended imaginary mode can promote a reviewed
candidate to `ready`. Old DFT result identities and reference reuse keys stay
unchanged.

`scope="full_cycle"` fails clearly for full UMA because the extra cycle
states have not been added. The installed Tooltoad ORCA parser reads the
`Final Gibbs free energy` line after `NumFreq`; a focused test also verifies
that Stepper carries this value into `uma_freq-GE`. Gas and ALPB runs, portable
restart, submitted stage plans, reference publication, missing data, bad
frequencies, mixed protocols, and balance checks passed with mocked
calculations. The fast UMA suite passed: 392 tests; 13 slow tests were
deselected. Live ORCA/UMA compatibility remains for Task 05.
