# Transition States

Transition-state workflows need chemical validation after the calculation
finishes. A normally terminated ORCA job is not automatically a chemically
useful TS.

## Fast Checks

- The frequency stage terminated normally.
- The TS has exactly one imaginary frequency.
- The imaginary mode follows the intended reaction coordinate.
- The optimized geometry still matches the intended substrate, template, and
  reactive position.
- The chosen conformer is chemically sensible.

!!! warning "One imaginary frequency is not the whole test"

    A first-order saddle point can still correspond to the wrong motion. Always
    inspect the imaginary mode before using the barrier.

## Count Imaginary Frequencies

Use the vibration column from the frequency stage:

```python
vibs_col = [col for col in df.columns if col.endswith("-vibs")][-1]

def n_imag(vibs):
    return sum(entry["frequency"] < 0 for entry in vibs)

df["n_imag"] = df[vibs_col].map(n_imag)
df[["substrate_name", "rpos", "cid", "n_imag"]].head()
```

Or use the FRUST helper:

```python
from frust.utils.analytics import summarize_ts_vibrations

summarize_ts_vibrations(df, col=vibs_col, max_rows=20)
```

## Inspect The Mode

```python
from frust.vis import plot_vibs

plot_vibs(df, row_index=0, vId=0)
```

The imaginary mode is usually `vId=0`, because negative frequencies are listed
first in typical parsed vibration output. If in doubt, inspect the frequencies
in the `*-vibs` entry for that row.

## Symptoms And Likely Causes

| Symptom | Likely cause | Next check |
| --- | --- | --- |
| zero imaginary frequencies | optimization found a minimum | inspect geometry and rerun from a better TS guess |
| more than one imaginary frequency | higher-order saddle point | inspect all imaginary modes |
| imaginary mode is unrelated | wrong template or reactive position | check `rpos` and template mapping |
| TS collapses during optimization | starting conformer too far from TS | use fewer bad conformers or improve template |
| g-xTB ORCA frequency fails | analytic `Freq` used with external g-xTB | use `NumFreq` |

!!! example "Filter to true-TS candidates"

    ```python
    vibs_col = [col for col in df.columns if col.endswith("-vibs")][-1]
    df_true_ts = df[df[vibs_col].map(lambda vibs: sum(v["frequency"] < 0 for v in vibs) == 1)]
    ```

    This is a useful first filter, but still inspect the mode animation before
    trusting the row.
