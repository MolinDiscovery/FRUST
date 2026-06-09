# Structure Comparison

Use `compare_xyz_rmsd` when two optimized structures should be checked as the
same geometry up to alignment, atom mapping, and small coordinate differences.

```python
import frust as ft

result = ft.vis.compare_xyz_rmsd(
    "gxtb.xyz",
    "orca.xyz",
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
XYZ files or FRUST row coordinates -> atom mapping -> aligned probe
-> RMSD + per-atom deviations -> scene overlay + compact table
```

The probe structure is aligned to the reference. The largest mapped atom
deviations are drawn as distance overlays, and the full deviation table stays in
`result["df_dev"]`.

!!! note "Current atom scope"

    Structure comparison currently uses `atom_scope="heavy"`. Hydrogens are
    ignored during RMSD and atom mapping, although they can still be present in
    the input structures.

## Compare XYZ Files

Start with two coordinate files:

```text
structures/
├── gxtb.xyz
└── orca.xyz
```

Then compare them:

```python
result = ft.vis.compare_xyz_rmsd(
    "structures/gxtb.xyz",
    "structures/orca.xyz",
    show="deviations",
    top_n=3,
    show_table=True,
)
```

The returned dictionary contains both the scalar RMSD and the mapped atom table:

| key | Meaning |
| --- | --- |
| `rmsd` | Heavy-atom RMSD after aligning the probe to the reference |
| `atom_map` | Mapped atom-index pairs as `(probe_idx, ref_idx)` |
| `df_dev` | Per-atom mapped deviations sorted from largest to smallest |
| `scene` | Reusable `GridScene` object |
| `viewer` | py3Dmol viewer when `render=True` |

Use `render=False` when you want the data and scene object without immediately
displaying the viewer:

```python
result = ft.vis.compare_xyz_rmsd(
    "structures/gxtb.xyz",
    "structures/orca.xyz",
    render=False,
)

ft.vis.show_scene(result["scene"])
```

## Compare Columns In A FRUST Row

For workflow outputs, compare two coordinate columns directly:

| substrate_name | rpos | gxtb-oc | orca-oc |
| --- | ---: | --- | --- |
| ethanol | 1 | probe coordinates | reference coordinates |

```python
result = ft.vis.compare_structure_rmsd(
    df,
    row_index=0,
    probe_col="gxtb-oc",
    ref_col="orca-oc",
    top_n=3,
)
```

This uses `df.iloc[row_index]["atoms"]` for both structures, aligns
`probe_col` to `ref_col`, and labels the scene with the row's
`substrate_name` and `rpos` when present.

## Choose The Display Mode

| `show` value | Behavior |
| --- | --- |
| `"deviations"` | Show the aligned overlay and draw the largest mapped atom deviations |
| `"overlay"` | Show the aligned overlay without deviation lines |
| `"none"` | Compute RMSD and the deviation table without creating a scene |

## Match The Molecule Grid Style

Structure comparison accepts the same practical display options used in
molecule grids:

```python
result = ft.vis.compare_xyz_rmsd(
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

Build a scene explicitly when you want to compose or export it:

```python
scene = ft.vis.structure_comparison_scene_from_dataframe(
    df,
    row_index=0,
    probe_col="gxtb-oc",
    ref_col="orca-oc",
    top_n=3,
)

ft.vis.show_scene(scene)
```
