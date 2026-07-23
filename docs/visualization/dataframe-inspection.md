# DataFrame Inspection

FRUST workflow outputs are usually inspected as pandas DataFrames. The
DataFrame plotting helpers convert rows and coordinate columns into molecular
views.

## Read FRUST Output

```python
import pandas as pd

df = pd.read_parquet("runs/example/init.hess.optts.freq.solv.parquet")
```

!!! warning "Filter failures before ranking"

    A failed row may still contain metadata or partial coordinates. Filter on
    the final `*-NT` column before comparing energies or selecting examples to
    visualize.

```python
nt_cols = [col for col in df.columns if col.endswith("-NT")]
df_ok = df[df[nt_cols[-1]]]
```

## Plot Molecules From Rows

```python
import frust as ft

ft.plot_mols(df_ok)
```

Representative output:

<iframe
  src="../../assets/plot-mols-dataframe-example.html"
  title="plot_mols-style FRUST DataFrame molecule grid"
  width="100%"
  height="400"
  loading="lazy"
  style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
></iframe>

The helper searches coordinate-like columns such as `coords`, `*-oc`, and
`*-opt_coords`, converts atoms and coordinates to molecules, and shows them in a
grid.

Pass one legend for each displayed cell when the generated dataframe titles
are not the clearest labels:

```python
preview = df_ok.iloc[:2]

ft.plot_mols(
    preview,
    legends=["Reactant", "Optimized product"],
    columns=2,
)
```

With the default coordinate selection, each row contributes its last
coordinate column, so this two-row preview needs two legends.

!!! example "Inspect one row"

    ```python
    ft.plot_row(df_ok, row_index=0)
    ```

    Representative output:

    <iframe
      src="../../assets/plot-row-example.html"
      title="plot_row-style row inspection grid"
      width="100%"
      height="400"
      loading="lazy"
      style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
    ></iframe>

!!! example "Inspect a substrate or reactive position"

    ```python
    ft.plot_lig(df_ok, "anisole")
    ft.plot_rpos(df_ok, 4)
    ```

    Representative output:

    <iframe
      src="../../assets/plot-lig-rpos-example.html"
      title="plot_lig and plot_rpos-style substrate grid"
      width="100%"
      height="400"
      loading="lazy"
      style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
    ></iframe>

## Select Coordinate Columns

Use `include_coords`, `exclude_coords`, or `coord_indices` when a workflow has
many optimization stages.

```python
ft.plot_mols(
    df_ok,
    row_indices=[0],
    include_coords=["OptTS", "Freq"],
    coord_indices=None,
    legends=["Optimized TS", "Frequency geometry"],
    cell_size=(450, 450),
)
```

Representative output is still an interactive molecule grid like the examples
above, but restricted to the selected coordinate stages. Cells—and therefore
legends—are ordered by dataframe row first and coordinate column second.
Coordinates that are missing for a row do not create a cell. FRUST validates
the legend count against the cells that remain.

!!! tip "Start broad, then narrow"

    First call `ft.plot_mols(df_ok.head())` to see which coordinate columns are
    present. Then narrow to the stages that matter for your inspection.

    `coord_indices` takes precedence over `include_coords` and
    `exclude_coords`. Set `coord_indices=None` when selecting coordinate
    columns by name.
