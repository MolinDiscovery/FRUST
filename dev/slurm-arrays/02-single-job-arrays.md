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

Pending. Record revision, Submitit version, test commands/results, and limits.
