# Cluster Runs

Cluster failures usually come from one of four places: input validation,
environment setup, scheduler configuration, or a backend chemistry job.

## Local Submission Test

Before using Slurm, test the submission wiring locally:

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="docs/examples/substrates.csv",
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

    Local submitit mode is best for checking CSV paths, workflow targets,
    method plans, and basic Python imports. Use Slurm for real ORCA and xTB
    workloads.

## Reading Stepper Log Names

Stepper loggers include the workflow label and the job context:

```text
frust.stepper.GENERIC.local
frust.stepper.TS1.job123456
```

The final part tells you where the calculation is running:

| Context | Logger suffix |
| --- | --- |
| Local Python or local submitit run | `.local` |
| Explicit `Stepper(job_id=42)` | `.job42` |
| Slurm allocation with `SLURM_JOB_ID=123456` | `.job123456` |

Seeing `.local` in a log is expected outside Slurm. It means FRUST did not find
a scheduler job id and did not invent a fake one.

## Common Errors

??? question "Missing `smiles` column"

    FRUST pipeline submissions expect a CSV with a `smiles` column:

    ```csv
    smiles,substrate_name
    COc1ccccc1,anisole
    ```

??? question "Unsupported pipeline name"

    `submit_jobs(...)` only accepts the high-level pipelines wired into the
    cluster interface, such as `run_mols` and `run_mols_per_rpos`. Submit TS
    work through `ft.workflows.screen_ts(...).submit(...)`.

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

`submit_jobs(...)`, `submit_screen_chain(...)`, and workflow submission return
a `JobSubmissionResult`:

```python
print(result.job_ids)
print(result.tags)
print(result.save_dirs)
```

Use these fields to connect scheduler jobs, output directories, and generated
tags.

For arrays and batches, use `result.records`: `job_ids` and `tags` need not
have the same length. Each record identifies a target, group, worker job, and
actual Slurm array index when one exists. Inspect the separate
`collection_job_id`; complete screens also have a `finalization_job_id`.

## Array Failures And Safe Retries

```python
import json
from pathlib import Path

report = json.loads(Path(result.collection_report).read_text())
report["retry_targets"]
report["target_results"]
```

`retry_targets` lists targets needing another attempt. A `blocked` staged
target could not continue after upstream failure; `interrupted` means the
worker ended without recording a final outcome. Check Slurm accounting and
logs to determine the cause. An absent parquet alone does not establish a
timeout. Collection uses completion dependencies, so it can report failures
after failed workers and cancelled descendants terminate.

If a retry reports an active or unverified earlier attempt, inspect its jobs
with `squeue`, `sacct`, and `scontrol show job`. Wait for the earlier collector
or screen finalizer. Use explicit `retry=True` with a compatible tracked run;
do not overwrite it through a new ordinary submission. See
[Slurm Arrays](../cluster/arrays.md#results-and-retries) for runnable selections.

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
