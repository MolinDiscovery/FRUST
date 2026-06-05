# Screen Input Tables

Start with a compact component table. Each row is either a substrate or a
catalyst.

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,,pyrrole
substrate,COC1=CC=CO1,methoxyfuran,"3,5",furan
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

```python
import frust as ft

components = ft.screen.read("screen.csv")
components
```

Output:

| role | smiles | compound_name | rpos | series |
| --- | --- | --- | --- | --- |
| substrate | `CN1C=CC=C1` | `n_methyl_pyrrole` |  | `pyrrole` |
| substrate | `COC1=CC=CO1` | `methoxyfuran` | `3,5` | `furan` |
| catalyst | `CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B` | `tmp_bcat` |  | `baseline` |

## Component Columns

| Column | Required | Meaning |
| --- | --- | --- |
| `role` | yes | Component kind. Accepted values are `substrate`, `sub`, `catalyst`, and `cat`; aliases are normalized to `substrate` or `catalyst`. |
| `smiles` | yes | SMILES for the substrate or catalyst. Missing values fail during `ft.screen.read(...)`. |
| `compound_name` | no | Stable readable name used in system names, target tags, and output rows. Missing names become `substrate_000`, `catalyst_000`, and so on. |
| `rpos` | no | Substrate reactive position or positions. Blank means use symmetry-unique aromatic C-H positions. Catalyst `rpos` values are ignored unless `strict=True`. |
| any extra column | no | Metadata. Extra substrate columns become `substrate_<name>` after expansion; extra catalyst columns become `catalyst_<name>`. |

!!! note "Catalyst rows do not use `rpos`"

    In non-strict mode, `ft.screen.read(...)` warns and clears catalyst `rpos`
    values. Use `ft.screen.read("screen.csv", strict=True)` when an accidental
    catalyst `rpos` should fail immediately.

## Reactive Positions

`rpos` is the RDKit atom index of an aromatic substrate C-H atom. It can be
blank, one integer, a comma-separated string, a semicolon-separated string, or
an in-memory sequence of integers.

| `rpos` value | Meaning |
| --- | --- |
| blank | Use `find_unique_ch(...)` to generate symmetry-unique aromatic C-H positions |
| `3` | Use only atom index `3` |
| `"3,5"` | Use atom indices `3` and `5` |
| `"3;5"` | Same as `"3,5"` |
| `[3, 5]` | Same selection when constructing a dataframe in Python |

For the example input:

| substrate | `rpos` input | generated `rpos` values |
| --- | --- | --- |
| `n_methyl_pyrrole` | blank | `2`, `3` |
| `methoxyfuran` | `3,5` | `3`, `5` |

Use a labeled drawing before choosing manual `rpos` values:

```python
ft.DrawUniqueChGrid(
    ["CN1C=CC=C1", "COC1=CC=CO1"],
)
```

!!! warning "Validate `rpos` from the exact SMILES"

    Atom indices come from RDKit after parsing the exact SMILES string in the
    screen table. A position that looks right in a hand drawing can still be
    invalid if the SMILES atom order differs.

If an invalid position is requested, FRUST reports both the bad values and the
valid aromatic C-H positions:

```text
ValueError: Invalid rpos values [2] for SMILES 'COC1=CC=CO1'.
Valid aromatic C-H positions: (3, 4, 5)
```

## Expand Components Into Systems

`ft.screen.expand(...)` crosses every substrate with every catalyst.

```python
systems = ft.screen.expand(components)
systems[
    [
        "system_name",
        "substrate_name",
        "catalyst_name",
        "rpos",
        "substrate_series",
        "catalyst_series",
    ]
]
```

Output:

| system_name | substrate_name | catalyst_name | rpos | substrate_series | catalyst_series |
| --- | --- | --- | --- | --- | --- |
| `n_methyl_pyrrole__tmp_bcat` | `n_methyl_pyrrole` | `tmp_bcat` |  | `pyrrole` | `baseline` |
| `methoxyfuran__tmp_bcat` | `methoxyfuran` | `tmp_bcat` | `3,5` | `furan` | `baseline` |

The system table stores the explicit pair:

| Column | Meaning |
| --- | --- |
| `system_name` | `<substrate_name>__<catalyst_name>` |
| `substrate_name`, `catalyst_name` | Component names copied from the input |
| `substrate_smiles`, `catalyst_smiles` | Component SMILES copied from the input |
| `smiles`, `rpos` | Substrate SMILES and substrate reactive-position selection |
| `substrate_*`, `catalyst_*` | Extra input metadata with component prefixes |

The expansion is deliberately explicit. If a screen has 11 substrates and 3
catalysts, this step gives 33 substrate-catalyst systems before `rpos`, TS
type, and conformer expansion.

## Count The Work Before Generating Structures

The approximate number of generated TS guess rows is:

```text
rows = TS types x systems x reactive positions x conformers
```

For the example:

| Quantity | Count |
| --- | ---: |
| TS types | 4 |
| substrate-catalyst systems | 2 |
| total reactive positions | 4 |
| conformers per generated TS guess | 1 |
| rows across all TS dataframes | 16 |

With `n_confs=None`, FRUST chooses the conformer count from the assembled
molecule's rotatable-bond count. Use `n_confs=1` for wiring checks and geometry
inspection before increasing coverage.

## Common Input Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `screen input is missing required columns` | Missing `role` or `smiles` | Add both columns to the CSV |
| `Unrecognized screen role values` | Role is not one of the accepted values | Use `substrate`, `sub`, `catalyst`, or `cat` |
| `Invalid rpos values` | Requested atom is not an aromatic C-H in the parsed SMILES | Draw labels from the exact SMILES and update `rpos` |
| Catalyst `rpos` warning | Catalyst row has an unused `rpos` value | Clear the value, or use `strict=True` to catch this in validation |
| No substrate or no catalyst rows | The component table cannot form pairs | Include at least one row of each role |
