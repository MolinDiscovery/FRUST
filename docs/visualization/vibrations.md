# Vibrations

Use `ft.plot_vibs` to inspect normal modes from FRUST frequency calculations. For
transition states, the imaginary mode should match the intended reaction
coordinate.

```python
import frust as ft

ft.plot_vibs(df_ok, vId=0)
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

For a portable catalyst-screen run, inspect its full-tier review queue and the
corresponding animated mode before approving a barrier:

```python
run = ft.screen.open_run("runs/uma-ts1-full")
queue = run.review_queue()
queue[["result_id", "state_id", "n_imag", "imaginary_frequencies_cm1"]]

result_id = queue.iloc[0]["result_id"]
run.plot_vibration(result_id, mode=0)
```

Rotate the molecule by dragging in the 3D viewer; watch whether the moving
atoms follow the expected bond changes. A single imaginary frequency alone
leaves the TS at `review`. After checking its geometry and mode, record an
explicit decision:

```python
run.set_review(
    result_id,
    "approved",
    note="Transfer H moves between catalyst N and substrate C.",
)
```

Minimum references need zero imaginary modes. If a ligand has an imaginary
methyl torsion, reoptimize it from a displaced geometry and repeat the
frequency calculation before using its free energy in a barrier.

For a full UMA run that retains several candidates, inspect each mode before
using its candidate-specific barrier:

```python
run.candidate_barriers()[[
    "ts_cid", "selected", "ts_review_status", "n_imag", "quality_status",
]]
```

| Status | Meaning for a candidate barrier |
| --- | --- |
| `ready` | Its TS mode is approved and all reference minima are ready. |
| `review` | The numerical barrier exists but a mode or other flagged result still needs inspection. |
| `invalid` | A TS or reference fails a quality check; the displayed energy is diagnostic only. |
| `incomplete` | A required calculation or thermal quantity is missing. |

One TS1 check had a clear N–H to substrate-C transfer mode, but its **gas
UMA ligand reference** retained an imaginary frequency at −66.53 cm⁻¹. The
gas barrier remained `invalid` after the TS mode was approved. A TS review
cannot override a bad minimum reference.

## Multiple Rows

By default, `plot_vibs(df_ok)` displays every row in the dataframe, matching
`plot_mols(df_ok)`. This makes filtered dataframes convenient:

```python
ft.plot_vibs(
    df_ok[df_ok["substrate_name"] == "1-benzylpyrrole"],
    columns=2,
)
```

For explicit subsets, pass row positions:

```python
ft.plot_vibs(
    df_ok,
    row_indices=[0, 1, 2, 3],
    columns=2,
    linked=True,
)
```

Use `max_rows` for large screens:

```python
ft.plot_vibs(
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
ft.plot_vibs(
    df_ok,
    row_index=0,
    vId=0,
    custom_coords_col_name="uma_ts_opt-oc",
)
```

## Export HTML

Use `export_HTML` to save an interactive viewer that can be embedded in the
documentation or shared with collaborators.

!!! example "Export an imaginary-mode viewer"

    ```python
    ft.plot_vibs(
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
    ft.plot_vibs(
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
