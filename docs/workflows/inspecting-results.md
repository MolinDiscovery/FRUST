# Inspecting Results

FRUST automates workflow plumbing. It does not remove the need for chemical
inspection. Before using a barrier in a substrate-scope prediction, check that
the calculation succeeded and that the optimized structure still represents
the intended chemistry.

!!! warning "Do not rank failed rows"

    Filter on normal-termination columns before sorting energies. A failed row
    can still contain metadata or partial output that should not enter a
    barrier estimate.

## First Pass: Calculation Health

Start from the final parquet file:

```python
import pandas as pd

df = pd.read_parquet("runs/example/init.hess.optts.freq.solv.parquet")
```

Find status columns and keep successful rows:

```python
nt_cols = [col for col in df.columns if col.endswith("-NT")]
print(nt_cols)

final_nt = nt_cols[-1]
df_ok = df[df[final_nt]]
```

Inspect failures before ignoring them:

```python
error_cols = [col for col in df.columns if col.endswith("-error")]
df[[*nt_cols, *error_cols]].head()
```

## What To Inspect Before Trusting A Result

- The optimized TS has one imaginary frequency.
- The imaginary mode corresponds to the intended bond formation or breaking.
- The lowest conformer is chemically sensible.
- The reactive position was mapped correctly.
- Failed ORCA, xTB, g-xTB, or UMA jobs were not silently included.
- The final energy column matches the stage you intend to compare.

!!! example "Summarize TS vibrations"

    FRUST includes a helper for quickly counting true and non-true TS rows:

    ```python
    from frust.utils.analytics import summarize_ts_vibrations

    summarize_ts_vibrations(
        df,
        col="Freq-vibs",
        max_rows=10,
    )
    ```

    If your frequency stage has a different name, use the matching `*-vibs`
    column from `df.columns`.

## Inspect The Imaginary Mode

Use `plot_vibs(...)` to view the normal mode. For a true TS, the single
imaginary mode should move along the intended reaction coordinate.

```python
from frust.vis import plot_vibs

plot_vibs(df_ok, row_index=0, vId=0)
```

!!! tip "Check the mode, not only the count"

    One imaginary frequency is necessary for a first-order saddle point, but it
    is not enough. The mode should describe the intended bond formation,
    breaking, proton transfer, hydride transfer, or other reaction coordinate.

## Rank Only After Filtering

Once failed rows are removed, rank by the energy column for the stage you mean
to compare:

```python
energy_cols = [col for col in df_ok.columns if col.endswith("-EE")]
print(energy_cols)

final_energy = energy_cols[-1]
df_ranked = df_ok.sort_values(final_energy)

df_ranked[
    ["substrate_name", "structure_type", "rpos", "cid", final_energy]
].head()
```

## Common Red Flags

| Symptom | What to check |
| --- | --- |
| no imaginary frequencies | optimized to a minimum instead of a TS |
| two or more imaginary frequencies | higher-order saddle point or bad geometry |
| imaginary mode is unrelated | wrong reactive position, wrong template, or conformer drift |
| best row looks distorted | low energy may come from an unphysical structure |
| many `*-NT=False` rows | backend setup, memory, geometry, or scheduler issue |

For symptom-first debugging, see
[Transition States](../troubleshooting/transition-states.md) and
[Failed Calculations](../troubleshooting/failed-calculations.md).
