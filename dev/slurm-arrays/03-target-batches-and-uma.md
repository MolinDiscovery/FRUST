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

Completed 2026-10-05. Implementation and live-tested revision: `a879304`
on `feature/slurm-arrays`.

```python
submission = wf.submit(
    out_dir="runs/uma-sp",
    cluster=cluster,
    execution="single_job",
    array=True,
    array_parallelism=2,
    targets_per_task=3,
    uma_oet_tools="/path/to/pinned/oet-runtime",
)
```

- The new worker runs each planned batch sequentially using the existing target
  runner. Outputs and target timing are saved immediately, and progress logs
  identify each target. CPU/memory requests remain per element; its timeout
  covers all targets. Direct `wf.run()` is unchanged; use `backend="local"`
  submission to exercise exactly the same batch worker locally.
- One lazy outer UMA scope owns the server for the whole batch. Inner target
  scopes borrow it with runtime compatibility checks and never close it.
  Standalone target ownership and default rejection of nested ownership remain
  intact. Process ownership checks prevent reuse after a fork.
- Recoverable target exceptions are recorded and later targets continue. At the
  end, an aggregate exception marks the element failed in Submitit/Slurm without
  discarding successful target outputs. Scientific non-normal termination is a
  separate outcome and does not itself raise an aggregate Python error.
- Chosen server recovery policy: stop the batch if startup fails or a health
  check fails, preserving completed outputs and leaving later targets
  unattempted. There is no automatic server restart. Interruption/SystemExit
  propagates after recording the active target and cleaning up the outer scope.
- Atomic batch records live at
  `<out_dir>/.frust/batches/<attempt_id>/<batch_index>.json`, separate from
  submission and scientific manifests. Target states mean: `unattempted` has
  not started, `running` has started without a final outcome, `success` produced
  a normally terminated output, `non_normal` produced an output with a failed
  normal-termination check, `failed` raised a Python exception, and `interrupted`
  received a handled process interruption. An abrupt kill can leave `running`
  records; scheduler reconciliation belongs to Task 04.
- Batch elapsed time includes server cleanup. `server_startup_s` measures service
  readiness startup and overlaps the first target's elapsed time; it is not an
  extra duration to add to target totals. Model loading on first inference remains
  in that target's elapsed time. Existing target timings and dataframe provenance
  remain intact; batch records include the shared server PID, host, and bind.
- Local checks: full fast suite **463 passed, 13 slow deselected**; focused
  batches/arrays/scope tests **43 passed**; focused tests including installed OET
  slow service checks **58 passed**. After an additional startup-failure
  regression test, the batch file passed **9 tests**. Compilation and whitespace
  checks passed. Tests cover uneven batches, shared Stepper SP/Opt/NumFreq calls,
  non-UMA work, runtime incompatibility, failure continuation, interruption,
  unhealthy/startup-failed server shutdown, and real local Submitit collection.
- Live script: [evidence/task03/smoke.py](evidence/task03/smoke.py). Local changes
  were committed/pushed before fetching a separate detached HPC worktree at
  `/lustre/hpc/kemi/jmni/dev/FRUST-slurm-arrays-task03` through HPC5.
  Run: `/lustre/hpc/kemi/jmni/results/slurm-arrays-task03-20261005`.
  Slurm array `65843239` has two elements, concurrency 1, batch size 3 for four
  water targets. Target 1 fails deliberately during preparation. Workers request
  4 CPUs/32 GB/20 minutes; collector `65843240` requests 1 CPU/2 GB/5 minutes.
  Runtime: `/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu`, cached
  `omol@uma-s-1p2p1`, gas-phase SP only. No large chemistry benchmark was run.

- Live verification passed on Slurm 26.05.4 / Submitit 1.5.2: batch 0 states
  `success`, `failed`, `success`; batch 1 state `success`. The three successful
  SP outputs have normal termination. Targets 0 and 2 share the same server PID,
  host, and bind. Exactly two server logs have one start and one stop each,
  confirming cleanup for the failed element as well as the successful element.
  Collection completed and included the three successful targets. Accounting
  records the intended failed element and completed second element/collector.
- Portable evidence: [slurm-verified.json](evidence/task03/slurm-verified.json).
  Run outputs/logs remain at the HPC path above. Task 04 will connect batch
  outcome records to richer collection diagnostics and safe explicit retries;
  current collection still reports the deliberately failed target as missing.
