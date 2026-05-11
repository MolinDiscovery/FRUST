# Vibrations

Use `plot_vibs` to inspect normal modes from FRUST frequency calculations. For
transition states, the imaginary mode should match the intended reaction
coordinate.

```python
from frust.vis import plot_vibs

plot_vibs(df_ok, row_index=0, vId=0)
```

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

```python
plot_vibs(
    df_ok,
    row_indices=[0, 1, 2, 3],
    viewergrid=(2, 2),
    linked=True,
)
```

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
