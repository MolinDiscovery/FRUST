# 04 — Build gas-phase and ALPB UMA TS-guess profiles

## Goal

Provide two separately calculated and reviewed `tsguess2` geometry profiles
for the selected OMol UMA model:

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

1. Start by inspecting the original optimized ωB97 structures in
   `structures/ts1.xyz`, `structures/ts2.xyz`, `structures/ts3_TMP.xyz`,
   `structures/ts4_TMP.xyz`, and `structures/int3_TMP.xyz`. Their reactive-role
   coordinates match the corresponding built-in ωB97 profile entries exactly.
   Verify each complete structure's chemical identity, atom/role mapping, and
   associated ωB97 calculation before selecting it as a UMA starting geometry.
   In particular, the TS3/TS4/INT3 TMP XYZ atom inventories are incompatible
   with the profile's migrated `1-methylpyrrole` provenance label; resolve that
   discrepancy before treating their frequencies as matched references. The
   later r012 NMe structures are a different chemical system and may only be
   used as separately identified alternative seeds, not mixed into the same
   reference set without review.
2. From the selected, identified starting structures, run separate gas-phase
   and ALPB-corrected UMA reference calculations for TS1–TS4 and required
   intermediates. These are new UMA optimizations, not copied ωB97 geometry
   values. Keep calculation inputs, method/version metadata, and structure
   identifiers for both sets.
3. Run the appropriate frequency calculation on every candidate reference in
   both environments. Require exactly one imaginary frequency for each TS and
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
6. Prepare a compact review table and saved mode viewers for gas and ALPB
   candidates. If the reactive-mode assignment or a large frequency
   difference remains ambiguous, present the evidence to the user for
   chemical review before accepting that reference. Quarantine a rejected
   reference rather than quietly filling it from another method family.
7. Adapt `scripts/extract_tsguess2_profile.py` to accept the actual UMA result
   columns (or create an explicit, audited conversion) before extracting
   candidate JSON. Review the calculated role coordinates and distances/angles
   against the final UMA geometries; do not copy source-row constraints.
8. Create separate UMA gas and ALPB profile modules in
   `frust/tsguess2/profiles/` and register both in that package's
   `__init__.py`, using the existing profile conventions. Choose filenames
   for the UMA model revision pinned in task 02. Add focused regression
   checks for profile resolution, required-state coverage, and generated-guess
   geometry for each environment.
9. Make workflow profile selection follow its selected UMA environment. If a
   state cannot be covered, report it explicitly before the workflow starts;
   do not silently substitute gas for ALPB or vice versa.

## Acceptance

- Every required state in each environment has a reviewed reference or an
  explicit documented limitation; no silent cross-method or cross-environment
  fallback occurs.
- Every accepted TS has exactly one imaginary frequency and a visually
  confirmed reactive mode. Every accepted minimum has no imaginary modes.
  Review records contain the mode index, frequency, viewer path, and matched
  ωB97 comparison for each TS; unresolved cases receive user review.
- Two registered profile files exist in `frust/tsguess2/profiles/`. Each
  records the UMA model and whether its source geometries used gas-phase UMA
  or the xTB ALPB(chloroform) correction.
- Focused geometry and profile tests pass in the UMA environment.

## Completion record

Pending. Add reference source paths for both environments, the frequency and
mode-review table with viewer paths, matched ωB97 comparisons, any user
review or quarantines, profile revisions, and test results here.
