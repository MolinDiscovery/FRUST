# Slurm Arrays, One Example At A Time

Suppose your workflow has ten targets, named A through J. Each target is one
independently tracked piece of chemistry, such as a molecule or a TS guess
family. It can contain several conformers.

| Submission | Targets | Arrays | Worker jobs | Maximum running worker jobs |
| --- | --- | --- | --- | --- |
| Individual, `single_job` | 10 | 0 | 10 | Determined by the scheduler |
| Array, one target per element, limit 2 | 10 | 1 | 10 | 2 |
| Array, five sequential targets per element, limit 2 | 10 | 1 | 2 | 2 |
| Two staged arrays, limits 2 and 1 | 10 | 2 | 20 | Up to 3 across both groups |

Collectors and screen finalizers are additional jobs, outside these counts.

## The Words In The Example

```text
Target A: molecule A
    prepare → optimize → frequencies
              stages of the same target

single_job array:
    element 0 / job: all stages for target A
    element 1 / job: all stages for target B
    ...
```

| Word | Meaning |
| --- | --- |
| Target | The piece of chemistry FRUST tracks, with its own output directory |
| Stage | A calculation or processing step within that target |
| Stage group | Stages FRUST runs together in one job, for example `init` |
| Job | One allocation of CPUs, memory, and time from the scheduler |
| Array | Jobs submitted together with common initial settings |
| Element or task | One worker job in that array |
| Parallelism | Maximum running elements within one array |

**A target is not a stage.** Target A passes through several stages. The
submission options decide how those stages and targets are placed into jobs.

## One Target Per Element: Complete Validation

For a raw molecule workflow, each row in `molecules.csv` becomes one target:

| smiles | substrate_name |
| --- | --- |
| `COc1ccccc1` | anisole |
| `Cc1ccccc1` | toluene |

```python
import frust as ft

cluster = ft.cluster.ClusterConfig(
    backend="slurm", partition="kemi1", log_dir="logs/validation",
)
wf = ft.workflows.raw_mols(csv_path="molecules.csv", dft=True)

submitted = wf.submit(
    out_dir="runs/validation", cluster=cluster,
    execution="single_job", array=True, array_parallelism=2,
    targets_per_task=1,
    stage_resources={
        "single_job": ft.cluster.Resources(cpus=8, mem_gb=32, timeout_min=720),
    },
)
```

With ten CSV rows, this submits ten elements. Each runs its molecule's entire
workflow, including its conformers. At most two elements run together. Each
running element requests **8 CPUs and 32 GB**, so two elements can request
16 CPUs and 64 GB together. Every element has a 720-minute time limit.

!!! info "Existing calls keep individual submission"

    `array=False` is the default. Arrays require an explicit positive
    `array_parallelism`. A selection containing only one element becomes an
    ordinary Submitit job; its result has no Slurm array parent or index.

## Several Targets Per Element: Shared UMA Server

```text
Ten targets, targets_per_task=5:

element 0: start server → A → B → C → D → E → stop server
element 1: start server → F → G → H → I → J → stop server
```

For UMA SP reranking of g-xTB geometries, select the TS child workflow:

```python
screen = ft.workflows.catalyst_screen(
    csv_path="screen.csv", ts_types=["TS1", "TS2"],
    screening="gxtb-default", ranking="uma-gas", level="uma_ranked",
)
sp_wf = screen.children()["transition_states"]
sp_wf.show_stages(execution="single_job")[["group", "stage", "engine"]]

submitted = sp_wf.submit(
    out_dir="runs/uma-rerank", cluster=cluster,
    execution="single_job", array=True, array_parallelism=2,
    targets_per_task=5,
    stage_resources={
        "single_job": ft.cluster.Resources(cpus=8, mem_gb=32, timeout_min=720),
    },
)
```

This workflow prepares and optimizes its guesses before UMA SP reranking. If
it has ten TS targets, it submits two worker jobs. The five targets within a
job run **sequentially** and share one lazily started UMA server. A new element
gets its own server. Model, environment, and server settings must be compatible
across the batch; incompatible settings fail validation. The server is closed
when the element exits, including after a target failure.

!!! warning "The timeout covers the whole batch"

    Five targets share one allocation and one time limit. They do not each
    receive 720 minutes. Larger batches also place more work at risk if a
    worker is interrupted. Start with small batches and inspect timings.

## Staged Arrays: Different Resources For Different Steps

For the DFT molecule workflow above:

```python
wf.show_stages(execution="dft_staged")[["group", "stage", "engine"]]
```

Its groups are `init`, `dft_opt`, `dft_freq`, and `dft_solv_sp`.
The simpler two-group example below shows how the dependencies behave:

```text
                  initialization array        frequency array
target A:         element 0: init A ──────────→ element 0: freq A
target B:         element 1: init B ──────────→ element 1: freq B
target C:         element 2: init C ──────────→ element 2: freq C
```

Frequency A can start when initialization A succeeds; it does not wait for B
or C. An upstream failure blocks that target's descendants. Slurm cancels the
invalid dependency, and the collector still runs after all workers terminate.

```python
groups = wf.show_stages(execution="dft_staged")["group"].unique()
limits = {group: 2 for group in groups}
limits["dft_freq"] = 1

submitted = wf.submit(
    out_dir="runs/staged", cluster=cluster,
    execution="dft_staged", array=True, array_parallelism=limits,
    targets_per_task=1,
    stage_resources={
        "init": ft.cluster.Resources(cpus=4, mem_gb=16, timeout_min=120),
        "dft_freq": ft.cluster.Resources(cpus=12, mem_gb=64, timeout_min=1440),
    },
)
```

Ten targets and four groups produce four arrays and forty worker jobs. Missing
resource overrides use the workflow defaults. A scalar `array_parallelism=2`
sets the same limit for every group; a mapping must cover every group.
Staged arrays require `targets_per_task=1`. They use matching-element Slurm
dependencies (`aftercorr`), or `afterok` for a single ordinary job.

!!! info "Limits apply separately to each array"

    Two running initialization jobs and one running frequency job can overlap.
    Likewise, a screen's TS array and reference array each have their own
    limit. `array_parallelism=2` does not cap the complete screen at two jobs.

## Results And Retries

Use the returned records to map targets to jobs:

```python
[(r.target, r.group, r.job_id, r.array_index) for r in submitted.records]
```

Example for a two-target single-job array:

```text
[("anisole", "single_job", "12345_0", 0),
 ("toluene", "single_job", "12345_1", 1)]
```

`submitted.job_ids` contains unique worker IDs. Several records can share a
job ID in a batch; staged targets have a record for each group. Use
`submitted.array_job_ids` for array parents and `submitted.collection_job_id`
for the separate collector. Collection normally writes `merged.parquet` and
`collection_report.json`. Wait for the collector before reading them.

| Report value | Meaning |
| --- | --- |
| `success` | The target completed its worker calculations |
| `failed` | A calculation raised an error |
| `non_normal` | A returned result failed its normal-termination check |
| `blocked` | A staged descendant could not proceed after upstream failure |
| `interrupted` | A worker ended without a final target outcome; its cause is unknown |
| `unattempted` | A batch could not reach this target, for example after server failure |
| `missing` | The expected result or stage outcome is absent |
| `unreadable` | A result parquet could not be read |
| `stale` | A result belongs to a different submission attempt |

Inspect `target_results`, `retry_targets`, and `failure_summary`; a successful
scheduler job alone does not establish scientific success. Retries are
explicit:

```python
import json
from pathlib import Path

report = json.loads(Path(submitted.collection_report).read_text())
failed = set(report["retry_targets"])
selected = [target for target in wf.targets() if target.tag in failed]
if selected:
    retried = wf.submit(
        out_dir="runs/staged", cluster=cluster, execution="dft_staged",
        array=True, array_parallelism=2, retry=True, targets=selected,
    )
```

Earlier workers and collection must have ended. FRUST preserves successful
targets, archives selected targets' older files, and rejects incompatible
chemistry or overlapping writes. A staged retry reruns the selected target's
complete chain. Its automatic collection contains the retry selection; after
it finishes, `wf.collect("runs/staged")` collects the whole workflow again.
If an attempt's completion cannot be verified, inspect scheduler jobs before
trying to resubmit. Older untracked output directories require a new run.

## Complete Screens And The CSV Interface

```python
screen = ft.workflows.catalyst_screen(csv_path="screen.csv", level="full")
submitted = screen.submit(
    out_dir="runs/screen", cluster=cluster,
    array=True, array_parallelism=2,
)
```

Only references marked `calculate` receive workers; cached references marked
`reuse` are snapshotted. Wait for `submitted.finalization_job_id`, which builds
the portable analysis bundle. A screen report's `success` means finalization
completed; `partial` means results are incomplete; `failed` can indicate a
missing required collection report. Review scientific quality separately.

After a partial screen finishes, `screen.submit(..., retry=True, array=True,
array_parallelism=2)` selects failed targets from branch reports and recollects
complete branches, including earlier successful targets. Keep the same output
directory and scientific settings. Screen retries also support
`artifact_policy="screening"`; cleanup waits for successful final validation.
The original reference reuse plan and submission records are retained.
A successfully finalized screen rejects further retries, including after its
per-target files have been cleaned up. Use a new output directory for more work.

The CSV facade accepts the same single-job array settings:

```python
submitted = ft.cluster.submit_jobs(
    csv_path="molecules.csv", pipeline="run_mols_per_rpos",
    select_mols=["ligand"], out_dir="runs/csv", cluster=cluster,
    resources=ft.cluster.Resources(cpus=4, mem_gb=8, timeout_min=120),
    array=True, array_parallelism=2, targets_per_task=3,
)
```

Each prepared structure is a target. `pipeline="run_mols"` treats the entire
CSV as one target. Tracked array outputs use `<target>/final.parquet`; default
individual facade calls keep their existing flat filenames. These two facade
pipelines use xTB/ordinary ORCA and do not provide shared UMA servers. Legacy
`submit_screen_chain` calls reject array options; use a workflow factory's
`submit` for staged arrays.

## Local Checks And Cluster Size Limits

Change the backend to exercise the same targets, stages, and collectors:

```python
local = ft.cluster.ClusterConfig(backend="local", log_dir="logs/local")
submitted = wf.submit(
    out_dir="runs/local", cluster=local, execution="single_job", targets=[0, 1],
    array=True, array_parallelism=2,
)
```

Local arrays use bounded ordinary workers and return no Slurm array parent or
index. Submission may wait for worker slots; staged local submission waits for
its graph to finish. A managed local screen waits for collectors before
queueing its finalizer. Direct `wf.run(...)` uses the same chemistry graph and
runs locally without submitting scheduler jobs.

If your site has an array-size limit, set
`ft.cluster.ClusterConfig(..., max_array_size=1000)`. FRUST rejects a larger
array before submission; it does not split it automatically. Choose a smaller
target selection or a suitable single-job batch size. FRUST owns the array and
dependency scheduler flags, so conflicting `extra_slurm_parameters` are
rejected.
Slurm jobs explicitly request one node and one Python worker; additional CPUs
belong to that worker's calculations. Conflicting task/node overrides are
rejected so scheduler CPU rounding cannot duplicate workers or finalizers.

See [Slurm's array documentation](https://slurm.schedmd.com/job_array.html)
for scheduler IDs and per-array limits, and
[Cluster Runs](../troubleshooting/cluster-runs.md) for diagnosing failures.
