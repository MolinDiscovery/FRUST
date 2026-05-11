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
from frust.vis import plot_mols

plot_mols(df_ok)
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

!!! example "Inspect one row"

    ```python
    from frust.vis import plot_row

    plot_row(df_ok, row_index=0)
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
    from frust.vis import plot_lig, plot_rpos

    plot_lig(df_ok, "anisole")
    plot_rpos(df_ok, 4)
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
plot_mols(
    df_ok,
    include_coords=["OptTS", "Freq"],
    cell_size=(450, 450),
)
```

Representative output is still an interactive molecule grid like the examples
above, but restricted to the selected coordinate stages.

!!! tip "Start broad, then narrow"

    First call `plot_mols(df_ok.head())` to see which coordinate columns are
    present. Then narrow to the stages that matter for your inspection.
