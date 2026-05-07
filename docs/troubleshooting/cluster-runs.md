# Cluster Runs

Cluster failures usually come from one of four places: input validation,
environment setup, scheduler configuration, or a backend chemistry job.

## Local Submission Test

Before using Slurm, test the submission wiring locally:

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_mols",
    out_dir="runs/local_test",
    cluster=ClusterConfig(backend="local", log_dir="logs/local_test"),
    resources=Resources(cpus=1, mem_gb=2, timeout_min=10),
    debug=True,
    production=False,
    n_confs=1,
    save_output_dir=False,
)
```

!!! tip "Use local mode for wiring, not chemistry"

    Local submitit mode is best for checking CSV paths, template paths,
    pipeline names, and basic Python imports. Use Slurm for real ORCA and xTB
    workloads.

## Common Errors

??? question "Missing `smiles` column"

    FRUST pipeline submissions expect a CSV with a `smiles` column:

    ```csv
    smiles,substrate_name
    COc1ccccc1,anisole
    ```

??? question "Missing `ts_xyz`"

    TS workflows need a template path:

    ```python
    submit_jobs(
        csv_path="datasets/example.csv",
        pipeline="run_ts_per_rpos",
        ts_xyz="structures/ts2.xyz",
        out_dir="runs/ts_example",
        cluster=cluster,
        resources=resources,
    )
    ```

??? question "Unsupported pipeline name"

    `submit_jobs(...)` only accepts the high-level pipelines wired into the
    cluster interface, such as `run_mols`, `run_ts_per_lig`, and
    `run_ts_per_rpos`.

??? question "Chain stage did not produce the expected parquet"

    Check the previous stage first. In a dependent chain, later stages depend
    on files from earlier stages:

    ```text
    init.parquet
    init.hess.parquet
    init.hess.optts.parquet
    init.hess.optts.freq.parquet
    init.hess.optts.freq.solv.parquet
    ```

## Reading The Submission Result

Both `submit_jobs(...)` and `submit_chain(...)` return a `JobSubmissionResult`:

```python
print(result.job_ids)
print(result.tags)
print(result.save_dirs)
```

Use these fields to connect scheduler jobs, output directories, and generated
tags.

## After Jobs Finish

Merge many parquet outputs before analysis:

```bash
merge_parquet --input-dir runs/ts_example --output merged.parquet --recursive
```

Then inspect status columns before ranking:

```python
import pandas as pd

df = pd.read_parquet("merged.parquet")
nt_cols = [col for col in df.columns if col.endswith("-NT")]
df[nt_cols].mean()
```

For chemistry-level failures inside completed jobs, continue with
[Failed Calculations](failed-calculations.md) and
[Transition States](transition-states.md).
