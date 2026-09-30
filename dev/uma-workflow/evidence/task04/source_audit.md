# Task 04 reference-source audit

The original ωB97 `tsguess2` role coordinates match atoms in these XYZ files
exactly (within the displayed six-decimal coordinate precision). The files are
useful UMA optimization seeds. Matching a profile entry does not, by itself,
prove that the entire XYZ was optimized with ωB97 or that a saved frequency
calculation belongs to it.

| State | XYZ | Atoms | Element inventory | Source chemistry | Saved ωB97 match |
| --- | --- | ---: | --- | --- | --- |
| TS1 | `structures/ts1.xyz` | 54 | C20 H31 B1 N2 | 1-methylpyrrole system; profile labels catalyst NMe | `r001_full_font_v2` TS1, 1-methylpyrrole `rpos=2` |
| TS2 | `structures/ts2.xyz` | 54 | C20 H31 B1 N2 | 1-methylpyrrole system; profile labels catalyst NMe; XYZ comment says `TS2 Guess` | `r001_full_font_v2` TS2, 1-methylpyrrole `rpos=2` |
| TS3 | `structures/ts3_TMP.xyz` | 70 | C25 H39 B2 N1 O2 S1 | Thiophene-derived five-membered C4S ring; TMP template | No matching `r001_full_font_v2` row found |
| TS4 | `structures/ts4_TMP.xyz` | 70 | C25 H39 B2 N1 O2 S1 | Thiophene-derived five-membered C4S ring; TMP template | No matching `r001_full_font_v2` row found |
| INT3 | `structures/int3_TMP.xyz` | 70 | C25 H39 B2 N1 O2 S1 | Thiophene-derived five-membered C4S ring; TMP template | No matching `r001_full_font_v2` row found |

For TS1 and TS2, the element inventory matches both 1-methylpyrrole `rpos=2`
and `rpos=3` saved ωB97 rows. Comparing sorted all-atom pairwise distances
between the source XYZ and the saved final ωB97 geometry gives an RMS
difference of **0.0085 Å vs 0.2284 Å** for TS1 (`rpos=2` vs `3`) and
**0.0212 Å vs 0.1950 Å** for TS2. The `rpos=2` rows are the appropriate
saved frequency comparisons: **−1000.23 cm⁻¹** (TS1) and **−326.41 cm⁻¹**
(TS2). Their final geometries are close, not byte-identical, to the source
XYZ structures. The full XYZ atom ordering differs from the saved rows.

The TMP source files contain a five-membered ring consisting of four carbon
atoms and sulfur. They cannot represent 1-methylpyrrole, despite the migrated
`ReferenceRecord.substrate_name` currently attached to the ωB97 profile.
Do not carry that label into the UMA profile. A matched ωB97 frequency for
these exact TMP/thiophene references remains to be located or calculated.

## Exact XYZ checksums

| State | SHA-256 |
| --- | --- |
| TS1 | `28ac40bdab8da14fc8defc033adbbbaf328260adddfcd15c83f2c70fa2dcc6f5` |
| TS2 | `4c7a017fcfddc84f536f795c19fca0509e5e0f947ac91518a9527ed0ecf1a2aa` |
| TS3 | `2802aba9512483ea466f45b1efc88260379186204f5e90d7b3fdd2d2c71d86e7` |
| TS4 | `64b3cb675902cb892a296f15d98c0f90ed9e7fdd623adc53a6d4e6ba2708f79b` |
| INT3 | `7750e9f074293e49d27bfb48817abef73fc0fcec52003e1063855f5525943e4b` |
