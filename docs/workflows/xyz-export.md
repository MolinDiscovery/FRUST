# Exporting XYZ Structures

FRUST dataframes often contain atom lists and one or more optimized coordinate
columns. Use `write_xyz` when you want to export those structures to `.xyz`
files for figures, external programs, or manual inspection.

```python
from frust.utils.io import write_xyz
```

## Start from a structure dataframe

`write_xyz` expects one structure per dataframe row. Think of the dataframe as
the source table for the XYZ files:

```text
one dataframe row + one coordinate column -> one .xyz file
```

A typical filtered FRUST dataframe might look like this:

| substrate_name | moltype | atoms | xtb_opt-oc | orca_opt-oc |
| --- | --- | --- | --- | --- |
| acetanilide | product(anin) | `["C", "O", "N", "H", ...]` | xTB coordinates | final DFT coordinates |
| benzamide | product(anin) | `["C", "O", "N", "H", ...]` | xTB coordinates | final DFT coordinates |

The important columns are:

| Dataframe field | Used for |
| --- | --- |
| `substrate_name` | output filename |
| `atoms` | XYZ atom symbols |
| `xtb_opt-oc`, `orca_opt-oc`, or another coordinate column | XYZ coordinates |

The export process is:

```text
row 0
├── filename from substrate_name: acetanilide
├── atom symbols from atoms: ["C", "O", "N", "H", ...]
└── coordinates from orca_opt-oc: [(x, y, z), ...]
    ↓
acetanilide.xyz
```

Conceptually, the first row becomes an XYZ file with an atom count, a comment
line, and one line per atom:

```text
4

C 0.02000000 0.01000000 0.00000000
O 1.22000000 0.02000000 0.00000000
N 0.01000000 1.31000000 0.01000000
H 0.00000000 1.91000000 0.81000000
```

!!! note "Coordinate arrays"
    The number of atom symbols must match the number of coordinate rows. A
    row with four atoms should have a coordinate array with shape `(4, 3)`.

!!! info "Automatic coordinate selection"
    If `coords_col` is omitted, FRUST uses the last coordinate-like column in
    dataframe order. Coordinate-like columns contain `coords` or end in `-oc`
    / `-opt_coords`. In a table with `xtb_opt-oc` followed by `orca_opt-oc`,
    the default export uses `orca_opt-oc`.

!!! tip "Older dataframe column names"
    Older FRUST parquet files may use longer coordinate names such as
    `xtb-gfn-opt-opt_coords` or
    `DFT-wB97X-D3-6-31G**-OptTS-opt_coords`. `write_xyz` recognizes those too.

## Export the final geometry from a dataframe

For the common case, pass a dataframe and an output directory. If you do not
specify `coords_col`, FRUST exports the latest coordinate-like column.

```python
df_write = df[df["moltype"] == "product(anin)"]

paths = write_xyz(
    df_write,
    "structures/products",
)

paths
```

Example output:

```text
[
    PosixPath("structures/products/acetanilide.xyz"),
    PosixPath("structures/products/benzamide.xyz"),
]
```

This writes one file per row:

```text
structures/products/
├── acetanilide.xyz
└── benzamide.xyz
```

## Choose a specific coordinate column

Use `coords_col` when you want a specific geometry instead of the automatic
latest-column choice.

```python
paths = write_xyz(
    df_write,
    "structures/products/dft",
    coords_col="orca_opt-oc",
)
```

Example output:

```text
[
    PosixPath("structures/products/dft/acetanilide.xyz"),
    PosixPath("structures/products/dft/benzamide.xyz"),
]
```

!!! tip "Use explicit columns for manuscript exports"
    The automatic default is convenient in notebooks. For final figures or SI
    exports, passing `coords_col="..."` makes it clear exactly which geometry
    was written.

## Export DFT and xTB structures into separate folders

Pass a mapping when you want to export multiple coordinate columns from the
same rows. The mapping keys become subfolder names and filename suffixes.

```python
paths = write_xyz(
    df_write,
    "structures/products",
    coords_col={
        "DFT": "orca_opt-oc",
        "xTB": "xtb_opt-oc",
    },
)
```

Example output:

```text
[
    PosixPath("structures/products/DFT/acetanilide_DFT.xyz"),
    PosixPath("structures/products/xTB/acetanilide_xTB.xyz"),
    PosixPath("structures/products/DFT/benzamide_DFT.xyz"),
    PosixPath("structures/products/xTB/benzamide_xTB.xyz"),
]
```

Directory layout:

```text
structures/products/
├── DFT/
│   ├── acetanilide_DFT.xyz
│   └── benzamide_DFT.xyz
└── xTB/
    ├── acetanilide_xTB.xyz
    └── benzamide_xTB.xyz
```

## Preview structures while exporting

Set `show_mols=True` to display the written structures after export. FRUST
opens the same interactive 3D grid used by `MolTo3DGrid`, so you can inspect
the saved geometries immediately.

```python
paths = write_xyz(
    df_write,
    "structures/products",
    show_mols=True,
    columns=3,
)
```

<iframe
  src="../../assets/xyz-export-structures.html"
  title="Interactive 3D preview of the acetanilide and benzamide XYZ export examples"
  width="100%"
  height="310"
  loading="lazy"
  style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
></iframe>

This is the same two-structure export shown above.

!!! info "Preview layout"
    Keyword arguments such as `columns` are forwarded to `MolTo3DGrid` and
    only affect the preview grid, not the written `.xyz` files.

The returned `paths` are still available even when the molecule grid is shown:

```text
[
    PosixPath("structures/products/acetanilide.xyz"),
    PosixPath("structures/products/benzamide.xyz"),
]
```

## Avoid overwriting files

By default, `write_xyz` overwrites existing files. Use `overwrite=False` when
you want FRUST to stop instead.

```python
paths = write_xyz(
    df_write,
    "structures/products",
    overwrite=False,
)
```

If a target file already exists, FRUST raises:

```text
FileExistsError: XYZ file already exists: structures/products/acetanilide.xyz
```

## Filename behavior

`write_xyz` uses safe filename stems. For example, `N-aryl/amides product`
becomes:

```text
N-aryl_amides_product.xyz
```

If two rows produce the same filename stem, FRUST appends a stable numeric
suffix:

```text
acetanilide.xyz
acetanilide_2.xyz
acetanilide_3.xyz
```

This prevents multiple rows in the same export from silently overwriting each
other.
