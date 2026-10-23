# Catalyst configurations and racemic dimers

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    systems="selected_systems.csv",
    catalyst_stereoisomers="all",
    dimer_pairing="racemic",
    dimer_reference="lowest",
    method="uma-gas-opt-alpb-chloroform",
    validation_top_n=3,
)
wf.plan()
```

For `Pip_cat_F_atom_0` (`Bc1ccccc1N1CCCCC1F`), this plans two explicit
carbon configurations. Each receives its own TS1–TS4 series. The racemic
reference pool has the following exact isomeric graph counts:

| Topology | Pair graphs | Pair-order equivalence |
| --- | ---: | --- |
| Reciprocal B–H bridges (`dimer`) | 3 | RS and SR are identical |
| Asymmetric B–H bridge (`dimer_bh_bridged`) | 4 | RS and SR differ |
| Reciprocal N–B bonds (`dimer_eight_membered`) | 3 | RS and SR are identical |

RR and SS remain distinct. Mirror symmetry is not used to reuse calculations.
All required pair graphs must have qualified results before selecting a
parent reference. The existing half-dimer barrier equation and
lowest-qualified-dimer rule are retained. Candidate energies are not dimer
populations; concentration corrections and kinetic modeling are outside this
feature.

| Control | Meaning |
| --- | --- |
| `catalyst_stereoisomers="input"` | Preserve input identities; default |
| `catalyst_stereoisomers="all"` | Resolve unspecified tetrahedral carbon configurations |
| `catalyst_stereoisomers="representative"` | One per mirror pair; retain distinct diastereomers and explicitly assigned inputs |
| `dimer_pairing="input"` | Duplicate each input monomer; default |
| `dimer_pairing="racemic"` | Include same-handed and mixed-handed parent pairs, independently of TS coverage |

Neither calculator choice nor calculation level changes these policies.
A cheap screen can explicitly choose `representative` and `input`; this is
an approximation to the reference coverage, not a full racemic search.
Other unspecified stereochemical features, including imine double bonds,
remain unresolved and appear in the plan's stereochemistry metadata.

Expanded CSVs retain `catalyst_parent_name` and `catalyst_parent_smiles`.
Resolving such an input again is idempotent. Selection scores inherited from
an unspecified parent retain their original meaning; they are not newly
calculated enantiomer-specific scores.

The typed builder records the hydrogen-explicit graph and persistent carbon
indices in dataframe attrs. Geometry checking compares mapped tetrahedral
orientation rather than CIP labels, since priorities can change during
assembly. Portable analysis reports `preserved`, `inverted`, `unresolved`
(planar), `unavailable` (missing or inconsistent evidence), or `not_applicable`
(no specified persistent centres). Changed or unavailable evidence cannot
qualify as the requested configuration. Stereo checks neither approve an
imaginary mode nor verify that successive TSs connect along a pathway.

R and S results remain separate. Differences between independently sampled
searches are sampling diagnostics. Choosing the lower configuration separately
at each TS does not establish a single verified pathway.
