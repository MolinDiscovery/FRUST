# 02 — Submit single-job arrays with one target per element

## Example outcome

```text
array=True, array_parallelism=2, targets_per_task=1
Targets A, B, C -> elements 0, 1, 2
At most two elements run simultaneously.
Each writes <target>/final.parquet and its usual timing information.
```

## Work

1. Connect the plan to `BaseWorkflow.submit(execution="single_job")` using a
   shared cluster submission adapter. Prefer `map_array` with compact worker
   payloads. Configure resources once per array and forward
   `slurm_array_parallelism` through the correct Submitit interface.
2. Use a shared array-level job name; keep target names in logs/output metadata.
   Do not change executor settings within a batch context or read unresolved
   job IDs. Preserve the existing lean serialized workflow and compact worker
   results rather than returning full dataframes through Submitit.
3. Persist actual parent/element IDs after submission and handle a one-target
   selection according to installed Submitit behavior. Do not invent an array
   ID if Submitit creates an ordinary job. Reject raw scheduler `array` or
   dependency options that conflict with FRUST-managed submission semantics.
4. Provide a local implementation that executes the same workers and enforces
   the requested concurrency bound. Do not assume LocalExecutor applies Slurm
   throttling. Keep direct `wf.run(...)` chemistry behavior compatible; local
   submission is the initial scheduler-parity test path.
5. Wire the existing collector to wait for every element, using parent IDs on
   Slurm where appropriate and actual completion waits locally. Keep collection
   resources separate from array worker settings and clear stale dependencies.
6. Validate size against a configured/obtainable cluster limit when practical;
   otherwise surface scheduler rejection with actionable guidance. No automatic
   splitting and no silent fallback to individual submission on Slurm.
7. After local tests pass, commit/push and update the cluster checkout. Run a
   reproducible tiny live array (for example four brief timestamp/sleep workers
   with parallelism two), a single-target submission, and a collector. Record
   versions, partition/resources, actual parent/element IDs, timestamp evidence
   for the concurrency bound, and collection output. Include one failing worker
   to confirm collection still runs. This is a scheduler check, not a chemistry
   benchmark. If cluster configuration/access is unavailable, preserve the
   script and mark the live gate pending rather than claiming full completion.

## Acceptance

- Fake-executor tests verify array settings, resource requests, IDs, one/zero
  target behavior, and exactly one scientific execution per selected target.
- A lightweight local integration check verifies the concurrency bound and
  that collection waits until all workers finish, including a failed worker.
- Existing individual submissions and output layouts remain compatible.
- The small live gate confirms actual array IDs, concurrency, single-target
  fallback, and collection after failure. Submission acceptance alone is not
  sufficient evidence.

## Completion record

Implementation completed 2026-10-05 at `1b7ad19` on
`feature/slurm-arrays`. Live acceptance passed on 2026-10-05.

- `BaseWorkflow.submit(..., execution="single_job", array=True,
  array_parallelism=2, targets_per_task=1)` uses Submitit `map_array` with the
  existing compact target worker. Resources are configured once per array.
  Actual element and parent IDs are saved atomically in one operation for a
  Slurm array; individual/local IDs are saved as each worker is accepted.
- Single-target submission reports the ordinary job returned by Submitit;
  empty selection creates no executor or collection job. Raw array/dependency
  overrides are rejected before submission. `ClusterConfig.max_array_size`
  supports a known limit; unknown limits are left to Slurm with actionable
  failure context. Arrays are never silently split or submitted individually.
- Local array submission uses the same worker, throttles actual active jobs,
  and may block while waiting for slots. Before launching collection it waits
  for workers in the submitting process. Submitit's serialized LocalJob loses
  its process handle, so passing those handles to a separate collector was
  insufficient; the integration test exposed and verified the fix. Worker
  exceptions do not suppress successful targets or prevent collection.
- Collection uses array-parent `afterany` dependencies on Slurm. A separate
  executor prevents worker array settings from leaking into the collector.
  Its resource override and existing retention behavior remain independent.
- Later features are explicitly guarded: batched targets (Task 03), staged
  arrays (Task 05), and other public entry points (Task 06). Retry safety remains
  Task 04; no automatic retry or resume support is claimed here.
- Checks: final focused command from Task 01 passed **92 tests**. The full fast
  suite passed **454 tests, 13 slow tests deselected** before the final bulk
  metadata-write optimization; the focused suite passed again afterward.
  `git diff --check` and Python compilation passed.
- Reproducible smoke script: [evidence/task02/smoke.py](evidence/task02/smoke.py).
  Local output `/tmp/frust-array-smoke-20261005-task02/verified.json` confirmed
  maximum worker concurrency 2, collection of three successes after one injected
  failure, and a successful one-target job and collector.
- HPC3 SSH timed out, but HPC5 (`fend05.cluster`) was reachable. The production
  checkout was clean and left on its existing branch. After local commit/push,
  a detached HPC worktree was fetched from GitHub at
  `/lustre/hpc/kemi/jmni/dev/FRUST-slurm-arrays-task02`.
- Cluster versions: Submitit **1.5.2**, Slurm **26.05.4**; local Submitit **1.5.3**.
  Partition `kemi1`; smoke worker and collector resources: 1 CPU, 2 GB, 5 minutes.
  Run: `/lustre/hpc/kemi/jmni/results/slurm-arrays-task02-20261005`.
  Array `65842984` (elements 0–3), collector `65842986`; one-target ordinary job
  `65842987`, collector `65842988`.
- Live verification passed: actual array indices 0–3, maximum observed worker
  concurrency **2**, successful collection of **3 targets** with the deliberate
  failed target missing, and successful one-target ordinary-job collection.
  Slurm accounting confirmed element `65842984_1` failed with exit code 1, the
  other elements completed, and both collectors completed with exit code 0.
  Jobs ran on `node066`. Requests were 1 CPU/2 GB; Slurm accounting reported
  2 allocated CPUs/2 GB per job on this cluster.
- Portable evidence: [slurm-verified.json](evidence/task02/slurm-verified.json),
  including submission/result mappings, versions, timestamp-derived concurrency,
  collection reports, and Slurm accounting. Run files and logs remain in the HPC
  run directory above; the dedicated HPC worktree remains available for the next
  tasks. The production checkout was not switched or edited.


