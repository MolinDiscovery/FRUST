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

## Artifact Retention

Use the same `artifact_policy` on `wf.run(...)` and `wf.submit(...)`.

| Policy | Saved outputs |
| --- | --- |
| `standard` (default) | Rich results, normal calculator outputs, and the final and required tier checkpoints for successful targets. |
| `compact` | Final geometries, energies, frequencies, and final TS modes. Routine calculator output stays in scratch. Failed ORCA logs are retained when available. |
| `diagnostic` | Standard results and every checkpoint produced by the chosen execution mode. |

For example, use compact retention for a staged run.

```python
result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
    artifact_policy="compact",
)
```

After automatic collection verifies the merged output, successful targets can
release obsolete checkpoints even in a partial run. Failed targets keep their
checkpoints. Manual collection preserves its source files.

!!! note "Updating workflow calls"
    Replace `artifact_policy="screening"` with `artifact_policy="compact"`.
    Remove workflow-level `target_retention` and `save_output_dir` arguments.
    Use `artifact_policy="diagnostic"` to keep staged checkpoint history.
    Use direct `ft.Stepper(...)` calls for custom calculator file retention.
    Resume runs with the FRUST version that created their submission records.
