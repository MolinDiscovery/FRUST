# Native Slurm arrays — sequential implementation plan

Start with [Task 01](01-api-and-submission-plan.md), then complete the tasks
below in order. Before starting a task, read this page and the previous task's
completion record. After finishing, record the implementation, checks, and
remaining limitations in that task and update this table. Do not implement all
tasks at once.

## What we are building

For this example, assume each molecule is one FRUST target:

```text
10 molecules = 10 targets

single_job, individual submission:
    10 jobs; each runs its molecule's complete workflow

single_job, array, targets_per_task=1:
    1 array containing 10 elements (jobs)
    each element runs its molecule's complete workflow

single_job, array, targets_per_task=5:
    1 array containing 2 elements (jobs)
    each element processes 5 molecules sequentially
    UMA calculations share one server within that element

staged, array, targets_per_task=1, two stage groups:
    optimization array: 10 elements
    frequency array:    10 elements
    frequency A waits for optimization A; B waits for B; etc.
    2 arrays, 20 jobs, still 10 targets
```

| Term | Meaning |
| --- | --- |
| Target | One independently tracked piece of chemistry, with its own output directory; it can contain several conformers |
| Stage | One workflow step, such as optimization |
| Stage group | One or more stages executed together in one job, as defined by the execution mode |
| Job | One scheduler allocation of CPUs, memory, and time |
| Array | Jobs submitted together with common initial scheduler settings |
| Element/task | One job within an array |
| Parallelism | Maximum running elements within one array |

Tasks 02 and 03 implement this `single_job` array API:

```python
import frust as ft

submission = wf.submit(
    out_dir="runs/validation",
    cluster=cluster,
    execution="single_job",
    array=True,
    array_parallelism=4,
    targets_per_task=1,
    stage_resources={
        "single_job": ft.cluster.Resources(cpus=8, mem_gb=32, timeout_min=720),
    },
)
```

Changing `targets_per_task=5` means five sequential targets in each job, not
five concurrent calculations. Each job's time limit covers the entire batch.

## Task order

| Task | Status | Depends on | Deliverable |
| --- | --- | --- | --- |
| [01 — API and submission plan](01-api-and-submission-plan.md) | Complete | None | Validated options, lightweight grouping, durable target/job records |
| [02 — Single-job arrays](02-single-job-arrays.md) | Complete | 01 | One target per element, Submitit integration, bounded local execution, tiny live array check |
| [03 — Batched targets and UMA](03-target-batches-and-uma.md) | Complete | 02 | Sequential target batches sharing one job-local server |
| [04 — Collection and retries](04-collection-and-retries.md) | Complete | 03 | Failure-aware collection and safe explicit target retries |
| [05 — Staged arrays](05-staged-arrays.md) | Pending | 04 | Matching-target dependencies, different stage resources/limits, live dependency/failure check |
| [06 — Public submission integration](06-public-submission-integration.md) | Pending | 05 | Screen and cluster entry points use consistent submission semantics |
| [07 — Verification and documentation](07-verification-and-documentation.md) | Pending | 06 | Integrated checks, bounded Slurm evidence, beginner-facing examples |

## Decisions and boundaries

- Preserve individual submission by default: `array=False`,
  `array_parallelism=None`, `targets_per_task=1`.
- Array mode requires an explicit positive concurrency limit. In staged mode,
  accept either one limit for every stage group or a complete mapping keyed by
  the names from `wf.show_stages(execution=...)`. Task 01 defines validation.
- `execution` continues to choose stage grouping; array settings choose how
  jobs are submitted. Neither changes chemistry, conformer selection, or the
  method plan.
- Batching is initially supported only in `single_job`. Staged arrays require
  one target per element and a stable mapping across stage groups.
- Resource requests apply per element. Never multiply CPU or memory requests
  automatically by the number of sequential targets.
- A limit applies per array, not across a screen's branches or all its stages.
  No promise of a global run-wide concurrency cap.
- Do not silently split submissions into multiple arrays. Oversized arrays need
  an actionable error and guidance on explicit target selection. Automatic
  splitting, global limits, concurrent targets inside an element, automatic
  retries, and mixed batching between stages are future work.
- Successful outputs remain per target. Scheduler records and attempt history
  belong in submission metadata; do not add scheduler-only dataframe columns.
- Preserve ordinary target failures and keep going within a batch when safe.
  Allocation termination can leave later targets unattempted. Record both.
- Retry selected failed/missing targets explicitly; do not blindly rerun
  successful targets from the same element.

## Working rules

1. Make library edits in the local repository. For cluster verification, commit
   and push before updating an HPC checkout from GitHub. Do not edit mounted
   HPC library source. Preserve material run artifacts before overwriting them.
2. Use the `UMA` environment and focused tests at each task. Public docstrings
   use NumPy style. Add expensive service checks under the `slow` marker.
3. Keep `targets()` and submission planning calculation-free. Reuse the same
   target objects, stage definitions, and worker behavior locally and on Slurm.
4. Inspect existing submission, screening manifest, retention, and retry/resume
   behavior before adding metadata. Preserve compatible run signatures and
   older result readability.
5. A task is complete only when its acceptance checks pass. Record real
   limitations rather than marking an unverified scheduler claim complete.

## Cluster checks during development

Develop and test locally, then use small live cluster checks at milestones:

| Milestone | Live check |
| --- | --- |
| Task 02 | A tiny array of Python workers; actual IDs, concurrency, and collection |
| Task 03 | A small multi-target UMA element; one server reused and cleaned up |
| Task 04 | Mixed failures, selected retries with changed resources, all-target recollection |
| Task 05 | Two staged arrays; matching dependencies, resources, failure and cancellation |
| Task 07 | Final integrated verification, including explicit retry and documentation examples |

Use brief timestamp/sleep workers for scheduler checks, and small molecules
only for chemistry/server checks. Discover the existing cluster connection,
partition, and runtime settings rather than inventing them. When they are
unavailable, record the missing prerequisite and keep the live gate pending.
Cluster access does not change the local edit -> GitHub -> HPC library rule.

## References

- [Submitit arrays, map_array, and batch](https://github.com/facebookincubator/submitit/blob/main/docs/examples.md)
- [Slurm array elements, limits, and dependencies](https://slurm.schedmd.com/job_array.html)
- Relevant implementation: `frust/cluster/config.py`, `executor.py`,
  `facade.py`, `chains.py`; `frust/workflows/core.py`, `screening.py`;
  `frust/utils/uma.py`.
