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

Pending. Record result columns, scope support, tests, and limitations here
before starting Task 04.
