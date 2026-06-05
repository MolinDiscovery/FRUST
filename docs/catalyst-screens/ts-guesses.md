# TS Guess Dataframes

Generate TS guesses from expanded systems:

```python
import frust as ft

components = ft.screen.read("screen.csv")
systems = ft.screen.expand(components)

ts_guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
)

ts_guesses.keys()
```

Output:

```text
dict_keys(['TS1', 'TS2', 'TS3', 'TS4'])
```

Each key contains one FRUST initial dataframe for that TS family.

```python
ts_guesses["TS1"][
    [
        "custom_name",
        "system_name",
        "structure_type",
        "rpos",
        "cid",
        "ts_spec_id",
    ]
].head()
```

Representative output:

| custom_name | system_name | structure_type | rpos | cid | ts_spec_id |
| --- | --- | --- | ---: | ---: | --- |
| `TS1(n_methyl_pyrrole__tmp_bcat_rpos(2))` | `n_methyl_pyrrole__tmp_bcat` | `TS1` | 2 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `TS1(n_methyl_pyrrole__tmp_bcat_rpos(3))` | `n_methyl_pyrrole__tmp_bcat` | `TS1` | 3 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `TS1(methoxyfuran__tmp_bcat_rpos(3))` | `methoxyfuran__tmp_bcat` | `TS1` | 3 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `TS1(methoxyfuran__tmp_bcat_rpos(5))` | `methoxyfuran__tmp_bcat` | `TS1` | 5 | 0 | `TS1::builtin::methylpyrrole_v1` |

## What A Generated Row Contains

| Column | Meaning |
| --- | --- |
| `custom_name` | Human-readable structure name used in plots and outputs |
| `structure_id` | Stable identifier with TS type, system, and `rpos` |
| `structure_type` | TS family: `TS1`, `TS2`, `TS3`, or `TS4` |
| `molecule_role` | `ts` for generated transition-state guesses |
| `system_name` | Substrate-catalyst pair name |
| `substrate_name`, `catalyst_name` | Component names |
| `rpos` | Substrate reactive atom index used for this row |
| `cid` | Embedded conformer id |
| `atoms` | Element symbols for the assembled molecule |
| `coords_embedded` | Starting coordinates for calculation stages |
| `connectivity_bonds` | Bonds used to reconstruct and draw the assembled molecule |
| `constraint_roles` | Mapping from chemical role names to atom indices |
| `constraint_spec` | Role-based distance and angle constraints |
| `constraint_atoms` | Legacy atom-order projection for older TS code paths |
| `ts_spec_id` | Versioned built-in TS specification identifier |
| `ts_core_metrics` | Measured distances/angles and deltas against the template constraints |

The dataframe also carries conformer-generation provenance in
`df.attrs["frust_conformers"]`. Inspect it through `ft.show_steps(df)` instead
of adding many sparse provenance columns to the main table.

## TS1-TS4 At A Glance

The built-in TS specs are role-based. The template defines where named roles
should sit in space; assembly finds those roles in the actual catalyst,
substrate, and built-in fragments.

| TS type | Extra fragment | Main motif | Main roles |
| --- | --- | --- | --- |
| `TS1` | H | C-H activation motif from the methylpyrrole template | `cat_B`, `cat_N`, `substrate_C`, `transfer_H` |
| `TS2` | H2 | H-H motif involving catalyst B/N and one retained catalyst B-H | `cat_B`, `cat_N`, `cat_H`, `transfer_H`, `n_transfer_H`, `substrate_C` |
| `TS3` | HBpin | HBpin-associated TMP motif where the HBpin B-H becomes `transfer_H` | `cat_B`, `cat_N`, `cat_H`, `pin_B`, `transfer_H`, `substrate_C` |
| `TS4` | HBpin | HBpin-associated TMP motif where a catalyst B-H becomes `transfer_H` | `cat_B`, `cat_N`, `cat_H`, `transfer_H`, `pin_B`, `substrate_C` |

Role meanings:

| Role | Meaning |
| --- | --- |
| `cat_B` | Boron atom in the catalyst B-aryl-N scaffold |
| `cat_N` | Nitrogen atom in the same catalyst scaffold |
| `substrate_C` | Aromatic substrate carbon selected by `rpos` |
| `transfer_H` | Hydrogen involved in the main TS hydrogen-transfer motif |
| `cat_H` | Retained catalyst B-H hydride used in `TS2`, `TS3`, or `TS4` |
| `n_transfer_H` | Hydrogen placed toward catalyst N in the `TS2` H-H motif |
| `pin_B` | Boron atom in the built-in HBpin fragment |

!!! note "TS3 and TS4 differ in hydride bookkeeping"

    In `TS3`, `transfer_H` is the B-H hydrogen on the HBpin fragment, and one
    catalyst B-H is retained as `cat_H`.

    In `TS4`, the HBpin B-H is removed, two catalyst B-H atoms are retained,
    and one of those catalyst hydrides becomes `transfer_H`.

## How A TS Guess Is Constructed

For every system, `rpos`, TS type, and conformer, FRUST:

1. Parses catalyst and substrate SMILES and adds explicit hydrogens.
2. Finds exactly one catalyst B-aryl-N scaffold and assigns `cat_B` and `cat_N`.
3. Finds the substrate atom selected by `rpos`, removes its attached H, and
   assigns `substrate_C`.
4. Adds the TS-specific fragment: H for `TS1`, H2 for `TS2`, and HBpin for
   `TS3` and `TS4`.
5. Assigns hydride roles such as `cat_H`, `transfer_H`, `n_transfer_H`, and
   `pin_B`.
6. Places role atoms on built-in template coordinates.
7. Adds softer frame anchors for the neighboring substrate atoms and, for
   `TS3`/`TS4`, HBpin oxygen atoms.
8. Embeds and rigidly places the disconnected fragments.
9. Stores atoms, coordinates, role maps, constraints, diagnostics, and plotting
   connectivity in the dataframe row.

This is why related substrates and catalysts can change while the reactive
core remains template-like.

## Hard Anchors And Soft Frame Anchors

Hard anchors are role atoms in the reactive core. Examples are `cat_B`,
`cat_N`, `transfer_H`, `pin_B`, and `substrate_C` when they belong to the TS
type.

Soft frame anchors guide larger fragments without pretending every atom belongs
to the reactive core:

| Frame | What it anchors | Why it matters |
| --- | --- | --- |
| substrate frame | `substrate_C` plus its two heavy neighbors | Keeps the aromatic ring face and bond directions aligned with the template |
| HBpin frame | `pin_B` plus its two oxygen neighbors | Prevents arbitrary HBpin rotation in `TS3` and `TS4` |

Too few anchors let fragments rotate into chemically wrong orientations. Too
many force-snapped atoms over-constrain substrates and catalysts that are only
similar to the template. The current design fixes the reactive role atoms and
gently controls the surrounding fragment frame.

## Row-Level Constraints

Every generated row carries the information needed to run constrained
optimization.

```python
row = ts_guesses["TS3"].iloc[0]
row["constraint_roles"]
```

Representative output:

```python
{
    "cat_B": 16,
    "cat_N": 9,
    "substrate_C": 42,
    "cat_H": 39,
    "pin_B": 56,
    "transfer_H": 67,
}
```

The actual constraints are stored separately:

```python
row["constraint_spec"]
```

Representative output for `TS3`:

| kind | roles | value |
| --- | --- | ---: |
| distance | `['transfer_H', 'cat_B']` | 1.376 |
| distance | `['transfer_H', 'pin_B']` | 1.264 |
| distance | `['transfer_H', 'substrate_C']` | 2.477 |
| distance | `['cat_B', 'substrate_C']` | 1.616 |
| distance | `['pin_B', 'substrate_C']` | 2.180 |
| distance | `['pin_B', 'cat_B']` | 2.007 |
| angle | `['cat_B', 'transfer_H', 'pin_B']` | 98.89 |
| angle | `['cat_B', 'substrate_C', 'pin_B']` | 61.75 |

`constraint_roles` maps chemistry to atom indices for this specific molecule.
`constraint_spec` defines the distances and angles by role. This separation is
what makes variable-catalyst screens possible.

!!! warning "Display bonds are not constraints"

    `connectivity_bonds` is for reconstructing and drawing the assembled
    molecule. Optimizers should use `constraint_spec`.

    For example, `TS3` and `TS4` can keep a `cat_B` to `pin_B` distance
    constraint even when no visual B-B bond is stored.

## Inspect Geometry Before Calculations

Always inspect a few rows before launching a large screen:

```python
ft.plot_row(ts_guesses["TS1"], 0)
ft.plot_row(ts_guesses["TS2"], 0)
ft.plot_row(ts_guesses["TS3"], 0)
ft.plot_row(ts_guesses["TS4"], 0)
```

For a compact panel:

```python
ft.plot_mols(ts_guesses["TS4"], range(0, min(6, len(ts_guesses["TS4"]))))
```

Look for these features:

| TS type | Geometry check |
| --- | --- |
| `TS1` | Transferred H sits between substrate C and catalyst N; catalyst B is close to substrate C |
| `TS2` | H-H motif is outside the catalyst/substrate framework, not buried in a fragment |
| `TS3` | HBpin B-H, catalyst B, and substrate C form the expected compact TMP-like core |
| `TS4` | HBpin is placed against substrate C; B-B contact is a constraint, not necessarily a visual bond |

## Core Metrics

`ts_core_metrics` measures each generated row against the built-in constraints.

```python
row = ts_guesses["TS3"].iloc[0]
row["ts_core_metrics"][:4]
```

Representative output:

| kind | roles | reference | measured | delta |
| --- | --- | ---: | ---: | ---: |
| distance | `['transfer_H', 'cat_B']` | 1.376 | 1.382 | 0.006 |
| distance | `['transfer_H', 'pin_B']` | 1.264 | 1.265 | 0.001 |
| distance | `['transfer_H', 'substrate_C']` | 2.477 | 2.482 | 0.005 |
| distance | `['cat_B', 'substrate_C']` | 1.616 | 1.616 | 0.000 |

Use these metrics immediately after embedding to catch poor placement. After
DFT TS optimization, the same distances and angles help you judge whether the
optimized TS stayed close to the template.

!!! note "Template quality is chemical, not just numerical"

    A small core-distance delta is reassuring only when the visual geometry and
    imaginary mode also match the intended reaction. Always inspect final TS
    structures before using barriers.

## Common TS Guess Failures

| Failure | Meaning | Next check |
| --- | --- | --- |
| No unique B-aryl-N scaffold | Catalyst is outside the current v1 matcher or has ambiguous scaffold matches | Inspect catalyst SMILES and decide whether the scaffold model should be extended |
| Missing B-H hydrogens | The TS type needs catalyst hydrides that the catalyst cannot provide | Check catalyst valence, charge, and whether the motif applies |
| Invalid `rpos` | Requested substrate atom is not an aromatic C-H | Draw atom labels from the exact SMILES |
| Strange `TS2` H-H placement | Underconstrained fragment orientation or a clash | Inspect `TS2` rows and add regression coverage if recurring |
| Wrong HBpin orientation | Frame anchors or hydride bookkeeping are not matching the intended motif | Compare `TS3`/`TS4` rows with TMP-like reference structures and `ts_core_metrics` |
