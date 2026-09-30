# 04 — Build reviewed UMA TS-guess profiles

> **Scope update, 2026-09-30:** ALPB profile construction is deferred. The
> screening workflow can use an existing ωB97 `tsguess2` profile for its initial
> structures and constrained screening stages, even when UMA performs the
> screening calculations. A UMA-specific profile is useful for calibration,
> but is not required to run screening. This task now completes the **gas**
> profile only. The ALPB profile is a later scientific investigation.

## Goal

Provide a calculated and reviewed gas-phase `tsguess2` geometry profile for
the selected OMol UMA model. A separately calculated ALPB profile remains
deferred:

| Profile | Potential used for reference calculations |
| --- | --- |
| UMA gas | UMA alone |
| UMA ALPB(chloroform) | UMA + GFN2-xTB ALPB(chloroform) minus gas-phase GFN2-xTB |

The ALPB profile serves the proposed default workflow. The gas profile allows
gas-phase UMA screening if the correction proves unsuitable. Do not relabel a
DFT reference or reuse gas-phase geometries as ALPB references.

## Existing ωB97 frequency context

The saved gas-phase ωB97X-D3/6-31G** final-frequency data in
[r001_full_font_v2](/Users/jacobmolinnielsen/Developer/FrustActivationProject/fruits/results-2026/r001_full_font_v2/)
give the following descriptive values. These rows terminated and have exactly one
negative frequency; their mode character still needs to be considered when
choosing a *matched* reference for an UMA profile.

| State | First-order rows | Median imaginary frequency (cm⁻¹) | Middle 50% (cm⁻¹) |
| --- | ---: | ---: | ---: |
| TS1 | 32 | −1037 | −1120 to −900 |
| TS2 | 28 | −338 | −358 to −300 |
| TS3 | 29 | −68 | −78 to −60 |
| TS4 | 31 | −165 | −220 to −112 |

For 1-methylpyrrole TS1 specifically, the saved values are −1000.23 cm⁻¹ at
`rpos=2` and −1117.45 cm⁻¹ at `rpos=3`. These figures are sanity context,
not hard UMA acceptance thresholds; compare the actual matching chemistry and
animated mode before accepting a reference.

## Work

1. Start by inspecting the original ωB97 profile-source structures in
   `structures/ts1.xyz`, `structures/ts2.xyz`, `structures/ts3_TMP.xyz`,
   `structures/ts4_TMP.xyz`, and `structures/int3_TMP.xyz`. Their reactive-role
   coordinates match the corresponding built-in ωB97 profile entries exactly.
   Verify each complete structure's chemical identity, atom/role mapping, and
   optimization provenance before treating it as an optimized ωB97 reference.
   An XYZ that lacks that provenance may still seed a new UMA optimization,
   but it cannot by itself supply a matched ωB97 frequency comparison.
   In particular, the TS3/TS4/INT3 TMP XYZ atom inventories are incompatible
   with the profile's migrated `1-methylpyrrole` provenance label; resolve that
   discrepancy before treating their frequencies as matched references. The
   later r012 NMe structures are a different chemical system and may only be
   used as separately identified alternative seeds, not mixed into the same
   reference set without review.
2. From the selected, identified starting structures, run gas-phase UMA
   reference calculations for TS1–TS4 and required intermediates. These are
   new UMA optimizations, not copied ωB97 geometry values. Keep calculation
   inputs, method/version metadata, and structure identifiers. Repeat this
   review independently if ALPB profile work resumes later.
3. Run the appropriate frequency calculation on every candidate reference.
   Require exactly one imaginary frequency for each TS and
   none for a minimum. Record the full negative-frequency list, the selected
   reactive mode index, and its frequency; do not assume the reactive mode is
   always mode zero.
4. Animate and inspect the selected mode for every TS using `ft.plot_vibs`,
   following [the vibration guide](../../docs/visualization/vibrations.md).
   Confirm that the motion follows the intended bond formation/breaking or
   transfer coordinate. Also check connectivity, reactive distances and
   angles, hydride placement, and orientation.
5. Compare each UMA imaginary frequency with the corresponding actual ωB97
   reference frequency for the same TS chemistry. Treat a major magnitude
   difference as a reason to investigate the structure and mode, not as an
   automatic rejection based on a fixed cutoff. In particular, check the
   expected large TS1 imaginary mode against its recorded ωB97 value rather
   than relying on an approximate remembered number.
6. Prepare a compact review table and saved mode viewers for the gas
   candidates. If the reactive-mode assignment or a large frequency
   difference remains ambiguous, present the evidence to the user for
   chemical review before accepting that reference. Quarantine a rejected
   reference rather than quietly filling it from another method family.
7. Adapt `scripts/extract_tsguess2_profile.py` to accept the actual UMA result
   columns (or create an explicit, audited conversion) before extracting
   candidate JSON. Review the calculated role coordinates and distances/angles
   against the final UMA geometries; do not copy source-row constraints.
8. Create and register the UMA gas profile using the model revision pinned in
   task 02. Add focused regression checks for profile resolution, required-state
   coverage, and generated-guess geometry. Do not register an unreviewed ALPB
   profile.
9. In task 05, make workflow profile selection explicit. An ALPB calculation
   may use a separately chosen initial guess profile, but must not describe a
   gas geometry as an ALPB reference.

## Acceptance

- Every required gas state has a reviewed reference. ALPB is explicitly
  deferred and has no registered UMA profile.
- Every accepted TS has exactly one imaginary frequency and a visually
  confirmed reactive mode. Every accepted minimum has no imaginary modes.
  Review records contain the mode index, frequency, viewer path, and matched
  ωB97 comparison where available. Unmatched chemistry and unresolved cases
  are identified explicitly and receive user review.
- A registered gas profile records the UMA model and gas-phase source method.
- Focused geometry and profile tests pass in the UMA environment.

## Completion record

Complete for the agreed gas-only scope. The initial gas TS3 result had two
shallow imaginary frequencies (−27.81 and −13.02 cm⁻¹). The final direct
gas-phase OptTS/NumFreq job `65685852` converged with one imaginary mode,
mode 0 at −94.46 cm⁻¹. Its reactive distances and mode resemble gas TS4,
as expected for these closely related structures. After reviewing the
[executed notebook](evidence/task04/ts3_gas_review.ipynb) and rotating the
py3Dmol viewers, the user accepted its TS3 assignment. The
[review record](evidence/task04/ts3_gas_final_review.md) preserves the numeric
comparison and the limitation that a matched ωB97 frequency for this exact
TMP/thiophene system is unavailable.

The reviewed [candidate JSON](evidence/task04/uma_gas_reviewed_candidates.json)
provides five gas states: TS1 −944.65, TS2 −278.92, TS3 −94.46, and TS4
−65.68 cm⁻¹, each with exactly one imaginary mode, plus INT3 with none. The
registered `omol-uma-s-1p2p1/gas` profile contains those five references.
The profile tests cover exact resolution, constraints against role coordinates,
and a generated TS3 guess. ALPB profile construction remains deferred; the
workflow task must handle this choice explicitly. A bounded follow-up is
described in [Post-task 07](07-optional-alpb-profile.md).

Verification on the local development checkout in the `UMA` environment:

```text
conda run -n UMA python -m pytest tests/test_method_aware_ts_specs.py tests/test_workflow_methods.py tests/test_tsguess2_role_mapping.py -q
25 passed
conda run -n UMA mkdocs build --strict
Documentation built successfully
git diff --check
No whitespace errors
```
