# Directory And Outputs

Workflow objects write one directory per lightweight chemistry target. Local
and submitted runs use the same target tags and stage names.

```text
runs/screen_ts/
├── TS1__substrate__catalyst__r2/
│   ├── structure_guess.parquet
│   ├── init.parquet
│   ├── dft_ts_opt.parquet
│   └── timing.json
├── TS4__substrate__catalyst__r2/
│   └── ...
├── merged.parquet
└── collection_report.json
```

`structure_guess.parquet` is the method-aware, constraint-bearing input. Its
rows include `constraint_roles`, `constraint_spec`, and `ts_spec_id`.

## Inspect Outputs

```python
result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
    collect=True,
)

print(result.save_dirs)
print(result.collection_output)
```

Read and summarize the merged dataframe:

```python
import pandas as pd
import frust as ft

df = pd.read_parquet(result.collection_output)
ft.show_steps(df)
```

## Manual Collection

If automatic collection was disabled or a run needs recovery:

```python
merged = wf.collect(
    "runs/screen_ts",
    output="runs/screen_ts/recovered.parquet",
)
```

FRUST reads the deepest completed parquet for each known target and merges
dataframe provenance from `df.attrs`.

## Calculation Directories

When `save_output_dir=True`, Stepper also preserves engine-specific files below
the target directory. These can include ORCA output, Hessian files, and xTB
scratch results. Set it to `False` for small wiring tests.
