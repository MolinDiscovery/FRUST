# 05 — Add staged arrays with matching-target dependencies

## Example outcome

```text
Optimization array              Frequency array
0: target A (16 CPUs, 32 GB) --> 0: target A (8 CPUs, 64 GB)
1: target B (16 CPUs, 32 GB) --> 1: target B (8 CPUs, 64 GB)
2: target C (16 CPUs, 32 GB) --> 2: target C (8 CPUs, 64 GB)
```

There are three targets, two arrays, and six jobs. A's frequency job becomes
eligible when A's optimization succeeds; it does not wait for B or C.

## Work

1. Extend the planner/adapter to `dft_staged` and `fully_staged`, submitting one
   array per stage group. Use the existing group definitions and resource lookup;
   do not hard-code chemistry stages. A group may contain multiple stages.
2. Preserve identical target ordering and element indices across groups. Use
   `aftercorr:<upstream_array_id>` for corresponding-element success dependency.
   Handle single-target ordinary jobs with normal `afterok` dependencies.
3. Apply each group's `stage_resources` and concurrency limit independently.
   Accept scalar limits or the complete group mapping defined in Task 01.
   Explain that two stages can run simultaneously, each with its own limit.
4. Reject `targets_per_task>1` before creating jobs. Do not change target order
   or regroup between stages. Different batch sizes across stages are deferred.
5. Make failure terminal: a failed upstream target must block its downstream
   work without leaving pending jobs that prevent collection forever. Verify a
   supported invalid-dependency cancellation mechanism on the target Slurm
   version/configuration (for example `kill_on_invalid_dep`) and configure it
   explicitly, or implement a bounded cancellation mechanism. Fail clearly when
   the chosen guarantee cannot be provided.
6. Collect after all relevant worker arrays have terminated, including failed
   upstream arrays and cancelled downstream work. Parent `afterany` dependencies
   are suitable only once blocked descendants become terminal. Distinguish
   upstream failure from downstream work blocked by that failure in the report.
7. Implement equivalent local ordering and failure propagation explicitly.
   Current local executor configuration ignores Slurm dependencies; verify that
   no worker reads a checkpoint before its predecessor finishes. Avoid waiting
   downstream workers consuming all local execution slots and causing deadlock.
8. Specify explicit staged retries. Reuse only validated compatible checkpoints;
   otherwise rerun the failed target's complete chain. If retrying from a group
   boundary is unsupported, document that limitation instead of implying resume.

## Acceptance

- Tests cover both execution modes, mapping/index stability, distinct resources,
  per-group limits, single-target fallback, rejected batching, and dependency IDs.
- Local staged integration proves checkpoint order, independent target progress,
  blocked descendants, and a collection report after upstream failure.
- Fake scheduler tests inspect cancellation/dependency settings. Task 07 must
  build on a live check completed here: two tiny staged arrays with different
  resource requests, matching-target dependencies, one injected upstream failure,
  terminal blocked descendants, and a finished collector. Record job IDs, states,
  timestamps, and reports. Follow local -> GitHub -> HPC; leave this gate pending
  if access/configuration is unavailable.

## Completion record

Completed 2026-10-05: implementation `5effe5a` and cancellation-evidence fix
`c521995` on `feature/slurm-arrays`.

### Concrete API

For a molecule workflow whose `show_stages(execution="dft_staged")` lists
`init`, `dft_opt`, `dft_freq`, and `dft_solv_sp`:

```python
import frust as ft

submitted = wf.submit(
    out_dir="runs/staged",
    cluster=cluster,
    execution="dft_staged",
    array=True,
    targets_per_task=1,
    array_parallelism={
        "init": 4,
        "dft_opt": 2,
        "dft_freq": 1,
        "dft_solv_sp": 2,
    },
    stage_resources={
        "dft_opt": ft.cluster.Resources(cpus=16, mem_gb=32, timeout_min=720),
        "dft_freq": ft.cluster.Resources(cpus=8, mem_gb=64, timeout_min=720),
    },
)
```

This submits four arrays. Element 0 always represents the first selected target,
element 1 the second, and so on. Each optimization element waits for its own
initialization element; each frequency element waits for its own optimization.
Target A can progress while C is still initializing. The limits apply separately:
two optimization jobs and one frequency job can run at the same time. Resource
overrides apply per element; omitted groups use the existing resource defaults.
Use the actual groups from `wf.show_stages(...)`, not this example's names for
every workflow. `fully_staged` uses the same submission mechanism with that mode's
group definitions. A scalar `array_parallelism=2` gives every group a limit of 2.

### Failure and collection

```text
Target   Initialization   Frequency
A        success          success
B        failed           blocked (Slurm cancels this element)
C        success          success

Collector: A and C included; retry_targets = ["B"]
```

Slurm workers request `kill-on-invalid-dep=yes`. Downstream arrays use
`aftercorr:<upstream-array>`; one-target submissions use ordinary jobs with
`afterok:<upstream-job>`. These settings follow the
[Slurm dependency and cancellation contract](https://slurm.schedmd.com/sbatch.html).
Unsupported scheduler submission is an error with accepted-job inspection
guidance; FRUST does not fall back to a dependency that can leave blocked work
pending indefinitely. Collection uses `afterany` on **all** worker arrays/jobs,
including failed upstream and cancelled downstream work.

Each group writes a durable outcome under `.frust/stages/<attempt>/<group>/`.
Its output carries target/group/attempt attribution in dataframe attrs. A
downstream worker validates that attribution before running calculations.
Scientific `*-NT` failure also produces a failed scheduler job, so descendants
are blocked even when the Python calculation returned a dataframe.

Never-started invalid-dependency cancellations may be absent from `sacct` on
this cluster even while `scontrol` reports `CANCELLED`. Terminal checks therefore
fall back to `scontrol` when Submitit returns `UNKNOWN`. The automatic collector
also saves durable `afterany_collection` completion receipts for such elements;
later retries do not depend on cancelled jobs remaining in the scheduler's cache.
If neither scheduler view nor durable evidence establishes completion, FRUST
keeps the attempt unverified and blocks overlapping writes.

`target_results` includes `stage_results`, with each group's job ID, status,
output, and available timestamps/error. `blocked` means the target's final work
could not proceed after upstream failure; `failed_group` and `upstream_status`
identify the first problem. An interrupted worker without a final outcome has
an unknown cause; a missing file alone is not described as a timeout.

For `backend="local"`, an in-process dispatcher starts only ready groups and
applies separate group limits. Failed descendants are recorded as `blocked`
without allocating a waiting job. This avoids checkpoint races and deadlocks.
The submission call waits until its local staged graph is terminal, including
when `collect=False`. Slurm submission returns after queueing work.

### Explicit retries

After the first collector finishes, select the tags in its `retry_targets`:

```python
selected = [t for t in wf.targets() if t.tag in report["retry_targets"]]
retried = wf.submit(
    out_dir="runs/staged",
    cluster=cluster,
    execution="dft_staged",  # same groups as the original submission
    targets=selected,
    array=True,
    array_parallelism=1,
    retry=True,
)
# After the retry collector finishes:
complete = wf.collect("runs/staged", require_normal_termination=True)
```

This retries B from initialization through frequency. It archives B's earlier
directory and keeps A/C unchanged. Checkpoint resume from a group boundary is
unsupported. Chemistry, target identity, and execution groups must match;
resources and per-group limits may change. The retry collector writes its
selected subset separately; final recollection merges preserved successes
with the latest attempts. Staged retries require `array=True`; legacy staged
individual submission retains its existing behavior. Screen wrappers and the
cluster facade receive public array/retry options in Task 06.

### Verification

- Full fast suite: final gate **484 passed**, **13 slow tests deselected**.
  Final focused staged/array/retry gate: 51 passed, including 11 staged-specific
  tests for fully-staged local retry, checkpoint attribution, terminal evidence,
  and interrupted/cancelled descendant regressions. Both staged modes, stable
  indices, different resources/limits, singleton fallback, blocked local work,
  scientific failure, and full-chain retries are covered.
- Full slow suite: **13 passed**.
- Live Slurm gate at `c521995`, Submitit 1.5.2, Slurm 26.05.4, partition `kemi1`:
  `/lustre/hpc/kemi/jmni/results/slurm-arrays-task05-20261005-v2`.
  Initialization array **65847236** requested 1 CPU/2 GB/5 minutes with limit 3;
  frequency array **65847237** requested 2 CPUs/4 GB/5 minutes with limit 1 and
  corresponding-element dependencies. A's frequency started at 19:03:55 while
  C's initialization continued until 19:04:27 (cluster accounting time).
  B's initialization failed; `scontrol` confirmed **65847237_1 CANCELLED**, reason
  `DependencyNeverSatisfied`, with `KillOnInvalidDependent=Yes`. Collector
  **65847239** completed, collected A/C, and reported B blocked by its failed
  initialization. Its durable cancellation receipt was saved despite the element's
  absence from `sacct`.
- Selected full-chain retry of B used ordinary jobs **65847277 → 65847278** with
  `afterok`, followed by collector **65847279**. All completed. Final recollection
  returned A/B/C exactly once; A/C retained their original attempt attribution.
  [Portable scheduler/report evidence](evidence/task05/slurm-verified.json),
  [cancelled-element completion receipt](evidence/task05/blocked-completion.json),
  and [runnable fixture](evidence/task05/smoke.py) are committed. An earlier pilot
  run at `5effe5a` exposed the missing-accounting edge case; the completed gate
  above repeats the entire cycle with the fix.

To reproduce, use a fresh root and the cluster's UMA Python from this checkout:

```bash
PYTHONPATH=. python dev/slurm-arrays/evidence/task05/smoke.py \
    --phase initial --output /path/to/new/run --partition kemi1
# Wait for its collector to finish.
PYTHONPATH=. python dev/slurm-arrays/evidence/task05/smoke.py \
    --phase retry --output /path/to/new/run --partition kemi1
# Wait for the retry collector to finish.
PYTHONPATH=. python dev/slurm-arrays/evidence/task05/smoke.py \
    --phase verify --output /path/to/new/run --partition kemi1
```
