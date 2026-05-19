# Exporting XYZ Structures

FRUST dataframes often contain atom lists and one or more optimized coordinate
columns. Use `write_xyz` when you want to export those structures to `.xyz`
files for figures, external programs, or manual inspection.

```python
from frust.utils.io import write_xyz
```

## Export the final geometry from a dataframe

For the common case, pass a dataframe and an output directory. If you do not
specify `coords_col`, FRUST exports the latest coordinate-like column.

```python
df_write = df[df["moltype"] == "product(anin)"]

paths = write_xyz(
    df_write,
    "structures/products",
    name_col="ligand_name",
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

!!! info "Default coordinate selection"
    If `coords_col` is omitted, FRUST uses the last coordinate-like column in
    dataframe order. Coordinate-like columns are columns containing `coords` or
    ending in `-oc` / `-opt_coords`. This usually corresponds to the latest
    optimized geometry.

## Choose a specific coordinate column

Use `coords_col` when you want a specific geometry instead of the automatic
latest-column choice.

```python
paths = write_xyz(
    df_write,
    "structures/products/dft",
    coords_col="DFT-wB97X-D3-6-31G**-OptTS-opt_coords",
    name_col="ligand_name",
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
        "DFT": "DFT-wB97X-D3-6-31G**-OptTS-opt_coords",
        "xTB": "xtb-gfn-opt-opt_coords",
    },
    name_col="ligand_name",
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

Set `show_mols=True` to display the written structures after export.
Additional keyword arguments are passed to `MolTo3DGrid`.

```python
paths = write_xyz(
    df_write,
    "structures/products",
    name_col="ligand_name",
    show_mols=True,
    columns=3,
)
```

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
    name_col="ligand_name",
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
ligand.xyz
ligand_2.xyz
ligand_3.xyz
```

This prevents multiple rows in the same export from silently overwriting each
other.
