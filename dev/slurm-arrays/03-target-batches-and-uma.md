# 03 — Batch targets and share the job-local UMA server

## Example outcome

```text
Element 0: start UMA -> A -> B -> C -> stop UMA
Element 1: start UMA -> D -> E     -> stop UMA
```

Five targets, two jobs, one array. Each target retains its own output directory.

## Work

1. Add a sequential batch worker for `single_job`. Reuse existing target/stage
   execution and write each target's final parquet/timing immediately. Include
   target identity in progress logs. Return compact per-target records.
2. Move UMA scope ownership to the outer batch when any stage needs UMA. Inner
   target execution must reuse a compatible existing scope without closing it.
   Preserve standalone target scope ownership and reject incompatible runtime
   reuse. The current context manager rejects nesting; do not simply wrap the
   existing worker in another scope.
3. Keep server startup lazy and job-local. Reuse one server across targets and
   stages; ensure cleanup on normal exit, Python errors, and handled termination.
   Do not inherit the scope across process forks or share it across allocations.
4. Continue after recoverable target exceptions, recording them durably. After
   all targets are attempted, fail the element if any target raised an exception
   so the scheduler reports failure. Keep scientific non-normal termination
   distinct from Python exceptions. Never catch cancellation/SystemExit merely
   to continue the batch.
5. Define recovery after a broken UMA server: restart with verified cleanup
   before proceeding, or stop the batch and record remaining targets as
   unattempted. Do not claim safe continuation without checking server health.
6. Reuse each element's CPU/memory allocation for sequential targets. Explain
   that timeout covers the whole batch, and distinguish target elapsed time from
   shared server startup and element elapsed time in existing timing metadata.
7. Use the same batch worker for local submission. If needed for direct local
   batch smoke tests, add a narrowly scoped `wf.run(..., targets_per_task=...)`
   option; preserve the default and avoid adding Slurm settings to `run()`.

## Acceptance

- Tests verify one scope/server start and close for multiple UMA targets,
  zero starts for non-UMA work, isolated target outputs, and uneven batches.
- Injected target failure preserves earlier successes and allows later targets
  to run; termination preserves successes and leaves truthful pending records.
- Existing UMA scope tests pass. Include a focused slow service check where
  feasible and a small live Slurm batch after local tests pass. Record server
  startup/cleanup evidence and per-target outputs, following the local -> GitHub
  -> HPC path. Task 07 repeats this only as needed for integrated verification.

## Completion record

Pending. Record revision, lifecycle/failure decisions, checks, and limitations.
