# Structure Comparison

Use `compare_rmsd` when two structures should be checked as the same geometry
up to alignment, atom mapping, and small coordinate differences.

```python
import frust as ft

result = ft.vis.compare_rmsd(
    "structures/gxtb.xyz",
    "structures/orca.xyz",
    top_n=3,
)

result["rmsd"]
result["df_dev"].head()
```

Representative output:

<iframe
  src="../../assets/structure-comparison-example.html"
  title="Structure RMSD comparison overlay"
  width="100%"
  height="410"
  loading="lazy"
  style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
></iframe>

The mental model is:

```text
two structure-like inputs -> atom mapping -> aligned probe
-> RMSD + per-atom deviations -> scene overlay + compact table
```

The probe structure is aligned to the reference. The largest mapped atom
deviations are drawn as distance overlays, and the full deviation table stays in
`result["df_dev"]`.

!!! note "Current atom scope"

    Structure comparison currently uses `atom_scope="heavy"`. Hydrogens are
    ignored during RMSD and atom mapping, although they can still be present in
    the input structures and in the displayed overlay.

## Mix Structure Sources

Each side is resolved independently, so the probe and reference do not need to
come from the same dataframe or from two files. This is useful when one
structure lives on disk and the other is already in a notebook dataframe:

```python
result = ft.vis.compare_rmsd(
    {"path": "/path/to/font2017.xyz", "label": "Font 2017"},
    {
        "df": df,
        "row_index": 0,
        "coords_col": "wb97-oc",
        "label": "FRUST wb97",
    },
    show="deviations",
    top_n=5,
)
```

The returned labels appear in the scene title:

```text
Font 2017 -> FRUST wb97
RMSD 0.043 A | 9 mapped | max 0.081 A
```

## Accepted Inputs

`compare_rmsd(probe, ref, ...)` accepts the same structure input forms on both
sides:

| Input | Example | Meaning |
| --- | --- | --- |
| XYZ path | `"gxtb.xyz"` | Read atoms and coordinates from an XYZ file |
| Explicit XYZ path | `{"path": "gxtb.xyz", "label": "g-xTB"}` | Same as a path, with a display label |
| XYZ block | `{"xyz": xyz_block, "label": "guess"}` | Parse raw XYZ text |
| Atoms and coordinates | `(atoms, coords)` | Tooltoad-style `atoms`, `coords` pair |
| Explicit atoms and coordinates | `{"atoms": atoms, "coords": coords}` | Same pair, with optional `label` |
| Dataframe row | `{"df": df, "row_index": 0, "coords_col": "orca-oc"}` | Use `df.iloc[0]["atoms"]` and `df.iloc[0]["orca-oc"]` |
| Selected row | `{"row": df.iloc[0], "coords_col": "orca-oc"}` | Use an already selected row |

!!! warning "Pass XYZ blocks explicitly"

    Bare strings are treated as file paths. If you already have XYZ text in
    memory, pass it as `{"xyz": xyz_block}` so FRUST can tell paths and text
    apart.

## Compare Columns In A FRUST Row

For workflow outputs, pass each coordinate column as a dataframe structure
input:

| substrate_name | rpos | gxtb-oc | orca-oc |
| --- | ---: | --- | --- |
| ethanol | 1 | probe coordinates | reference coordinates |

```python
result = ft.vis.compare_rmsd(
    {"df": df, "row_index": 0, "coords_col": "gxtb-oc"},
    {"df": df, "row_index": 0, "coords_col": "orca-oc"},
    top_n=3,
)
```

This uses `df.iloc[0]["atoms"]` for both structures, aligns `gxtb-oc` to
`orca-oc`, and labels the scene with the row's `substrate_name` and `rpos` when
present.

## Compare XYZ Blocks Or Atoms And Coordinates

Use an XYZ block when a structure is already in memory:

```python
result = ft.vis.compare_rmsd(
    {"xyz": xyz_block, "label": "external TS guess"},
    {"df": df, "row_index": 0, "coords_col": "coords_embedded"},
)
```

Use a tooltoad-style atoms/coordinates pair when you already have those arrays:

```python
result = ft.vis.compare_rmsd(
    (probe_atoms, probe_coords),
    {"atoms": ref_atoms, "coords": ref_coords, "label": "reference"},
)
```

## Choose The Mapping Mode

| `mapping` value | Behavior | Use when |
| --- | --- | --- |
| `"topology"` | Infer bonds with RDKit, match the heavy-atom graph, and choose the lowest-RMSD match | Atom order may differ and RDKit can perceive the molecule |
| `"index"` | Pair heavy atoms in input order and skip bond perception/topology matching | Atom order already matches, or bond perception fails for a TS-like geometry |

The default is `mapping="topology"` because it is robust to atom-order changes:

```python
result = ft.vis.compare_rmsd("gxtb.xyz", "orca.xyz")
```

Use index mapping for structures with known matching atom order:

```python
result = ft.vis.compare_rmsd(
    {"xyz": ts_guess_xyz, "label": "embedded guess"},
    {"df": df, "row_index": 0, "coords_col": "OptTS-oc"},
    mapping="index",
)
```

If the atom correspondence is known but not index-identical, pass explicit atom
pairs:

```python
result = ft.vis.compare_rmsd(
    {"atoms": probe_atoms, "coords": probe_coords},
    {"atoms": ref_atoms, "coords": ref_coords},
    atom_map=[(0, 3), (1, 4), (2, 5)],
)
```

The atom map uses original atom indices as `(probe_idx, ref_idx)` pairs.

## Choose The Display Mode

| `show` value | Behavior |
| --- | --- |
| `"deviations"` | Show the aligned overlay and draw the largest mapped atom deviations |
| `"overlay"` | Show the aligned overlay without deviation lines |
| `"none"` | Compute RMSD and the deviation table without creating a scene |

Use `render=False` when you want the data and scene object without immediately
displaying the viewer:

```python
result = ft.vis.compare_rmsd(
    "structures/gxtb.xyz",
    "structures/orca.xyz",
    render=False,
)

ft.vis.show_scene(result["scene"])
```

The returned dictionary contains both the scalar RMSD and the mapped atom table:

| key | Meaning |
| --- | --- |
| `rmsd` | Heavy-atom RMSD after aligning the probe to the reference |
| `mapping` | Mapping mode actually used: `topology`, `index`, or `explicit` |
| `atom_map` | Mapped atom-index pairs as `(probe_idx, ref_idx)` |
| `df_dev` | Per-atom mapped deviations sorted from largest to smallest |
| `scene` | Reusable `GridScene` object |
| `viewer` | py3Dmol viewer when `render=True` or `export_HTML` is set |

## Match The Molecule Grid Style

Structure comparison accepts the same practical display options used in
molecule grids:

```python
result = ft.vis.compare_rmsd(
    "structures/gxtb.xyz",
    "structures/orca.xyz",
    background_color=("blue", 0.1),
    show_labels=False,
    show_charges=True,
    kekulize=True,
    cell_size=(520, 480),
)
```

The reference structure keeps normal element-colored styling. The aligned probe
is drawn as a thinner orange overlay, so the two geometries stay visually
separable.
