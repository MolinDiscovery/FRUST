# Vibrations

Use `plot_vibs` to inspect normal modes from FRUST frequency calculations. For
transition states, the imaginary mode should match the intended reaction
coordinate.

```python
from frust.vis import plot_vibs

plot_vibs(df_ok, vId=0)
```

`plot_vibs` uses the same scene renderer and default visual style as
`plot_mols`, so static molecule grids and animated vibration grids have matching
cell sizes, labels, background, and atom/stick styling.

<iframe
  src="../../assets/3methoxyphenol-ts1-imaginary-mode.html"
  title="3-methoxyphenol TS1 imaginary-mode animation"
  width="100%"
  height="480"
  loading="lazy"
  style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
></iframe>

!!! warning "One imaginary frequency is not enough"

    A first-order saddle point should have one imaginary frequency, but the mode
    must also describe the intended bond formation, bond breaking, proton
    transfer, hydride transfer, or other reaction coordinate.

## Multiple Rows

By default, `plot_vibs(df_ok)` displays every row in the dataframe, matching
`plot_mols(df_ok)`. This makes filtered dataframes convenient:

```python
plot_vibs(
    df_ok[df_ok["substrate_name"] == "1-benzylpyrrole"],
    columns=2,
)
```

For explicit subsets, pass row positions:

```python
plot_vibs(
    df_ok,
    row_indices=[0, 1, 2, 3],
    columns=2,
    linked=True,
)
```

Use `max_rows` for large screens:

```python
plot_vibs(
    df_ok,
    max_rows=12,
    columns=3,
    vId=0,
)
```

!!! note "Single-row views"

    Pass `row_index=0` when you want only one row. Omitting row selectors shows
    all rows.

FRUST automatically chooses the latest non-empty vibration column and the best
matching optimized coordinate column. This lets the same call work for
conventional columns such as `DFT-wB97X-D3-6-31G**-OptTS-vibs` and compact
screen-chain columns such as `Freq-vibs`.

## Custom Coordinate Column

Use `custom_coords_col_name` when you want to inspect vibrations against a
specific coordinate stage.

```python
plot_vibs(
    df_ok,
    row_index=0,
    vId=0,
    custom_coords_col_name="UMA-OptTS-oc",
)
```

## Export HTML

Use `export_HTML` to save an interactive viewer that can be embedded in the
documentation or shared with collaborators.

!!! example "Export an imaginary-mode viewer"

    ```python
    plot_vibs(
        df_ok,
        row_index=0,
        vId=0,
        export_HTML="docs/assets/my-imaginary-mode.html",
    )
    ```

    Then embed it in a Markdown page:

    ```html
    <iframe
      src="../../assets/my-imaginary-mode.html"
      title="Imaginary-mode animation"
      width="100%"
      height="480"
      loading="lazy"
      style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
    ></iframe>
    ```

!!! tip "Label comparison grids"

    When comparing rows, pass `legends` so each viewer cell is identified:

    ```python
    plot_vibs(
        df_ok,
        row_indices=[0, 1],
        legends=["lowest", "second-lowest"],
    )
    ```

## Scene-Based Comparison

The visualization layer can also build a reusable scene before rendering. This
is useful for mixed static/animated comparison views.

```python
import frust as ft

scene = ft.vis.vibration_scene_from_dataframe(
    df_ok,
    row_indices=[0, 1, 2, 3],
    vId=0,
    columns=2,
)

ft.vis.show_scene(scene)
```
