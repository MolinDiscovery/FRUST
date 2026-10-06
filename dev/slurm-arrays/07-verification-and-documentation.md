# 07 — Verify the complete feature and document it with examples

## Example outcome

User-facing documentation starts with the counts, then shows the call:

| Example | Targets | Arrays | Worker jobs | Maximum simultaneous worker jobs |
| --- | --- | --- | --- | --- |
| Individual single-job submission | 10 | 0 | 10 | Determined by scheduler |
| Array, one target/element, limit 2 | 10 | 1 | 10 | 2 |
| Array, five sequential targets/element, limit 2 | 10 | 1 | 2 | 2 |
| Two staged arrays, limits 2 and 1 | 10 | 2 | 20 | Up to 3 across both stages |

Collectors/finalizers are additional jobs and are excluded from these counts.

## Work

1. Run integrated tests for individual submission, one-target arrays, sequential
   batches, staged arrays, failures, explicit retries, and collection. Use small
   workers for scheduler behavior; do not launch a large chemistry benchmark.
2. Run the appropriate project suites in UMA. Because this crosses execution
   and UMA lifecycle paths, run both fast and slow tests before declaring the
   entire sequence complete:

   ```bash
   conda run -n UMA python -m pytest -m "slow or not slow"
   ```

3. Review live evidence from Tasks 02, 03, and 05, then prepare reproducible
   bounded integrated Slurm smoke checks using the user's configured
   cluster. Record FRUST/Submitit/Slurm versions, resources, IDs, and run paths.
   Commit/push local library changes before updating the HPC checkout. If no
   cluster is available, preserve runnable checks and leave Slurm verification
   pending; do not claim mocks prove scheduler behavior.
   Include the public CSV facade and a managed screen from Task 06: verify
   branch collection, reference reuse, a partial finalizer, and a targeted
   retry that retains earlier successful targets in the finalized bundle.
4. Reuse earlier evidence when the implementation remains unchanged; rerun
   affected checks after relevant changes. Verify one target per element and observed concurrency with a tiny array.
   Verify a multi-target UMA element starts one server and cleans it up. Verify
   two staged arrays use matching-target dependencies, differing resources, and
   overlapping healthy progress. Inject an upstream failure and confirm blocked
   descendants terminate and collection runs. Exercise a targeted retry.
5. Update cluster/workflow guides and NumPy public docstrings. Define target,
   stage group, job, array, and element beside concrete examples. Show current
   defaults, per-element resources, whole-batch timeout, independent per-array
   limits, actual return records, collection, and safe retry calls.
6. Include beginner-friendly diagrams like those in this plan. Show runnable
   examples using `import frust as ft` and stage names obtained from
   `wf.show_stages(...)`. Define report statuses near their examples. Explain
   staged batching restrictions and array-size limits without exposing internal
   implementation details in user-facing instructions.
7. Run `conda run -n UMA mkdocs build --strict` and check examples against the
   actual implemented signatures and outputs. Update this plan's status table
   only after evidence for each acceptance gate is recorded.

## Acceptance

- Fast/slow tests and strict documentation build pass, or concrete environmental
  blockers remain visibly pending with commands and diagnostics recorded.
- Bounded real Slurm evidence confirms concurrency, server reuse/cleanup,
  dependencies, failure termination, and collection. A successful submission
  alone is not a completed smoke check.
- A reader can reproduce both p28 patterns and explain how many targets, jobs,
  arrays, and simultaneous calculations each example produces.
- No unimplemented API, automatic retry guarantee, or run-wide concurrency cap
  is presented as available.

## Completion record

Completed 2026-10-05 on `feature/slurm-arrays`: implementation/documentation
`3613bf7` and live-discovered fixes `b475acf`. Final tests, strict documentation
build, and affected live Slurm assertions all passed.

### Local verification

- Final full suite in UMA, including both fast and slow tests: **519 passed**.
  Only existing dependency deprecation/future warnings were reported.
- Strict documentation build passed. Output was directed to
  `/tmp/frust-task07-docs` to preserve the existing untracked `site/` directory.
- The new [public API smoke](evidence/task07/smoke.py) completed all four phases
  with native local Submitit workers. Its assertions verified CSV facade
  batching and legacy individual outputs, a deliberately partial screen,
  targeted retry, complete recollection, reference reuse, scientific manifest
  stability, successful cleanup, and preserved attempt/completion records.
  Local evidence: `/tmp/frust-arrays-task07-local-20261005/verified.json`.
- Guide examples were inspected through the real factories without running
  chemistry: raw molecule target labels, DFT stage-group keys, and the
  g-xTB → UMA SP ranking stage matched the examples.

```bash
conda run -n UMA python -m pytest -m "slow or not slow" -q
conda run -n UMA mkdocs build --strict --site-dir /tmp/frust-task07-docs
```

### Live gates

| Gate | Evidence | Revision |
| --- | --- | --- |
| One target per element, concurrency 2, singleton fallback, collector after failure | [Repeated element check](evidence/task07/elements-slurm-verified.json), array `65848221`, singleton `65848225` | `b475acf` |
| Real UMA water SP batches, one server per element, continuation after middle-target failure, server shutdown | [Task 03](evidence/task03/slurm-verified.json), array `65843239`, collector `65843240` | `a879304` |
| Staged corresponding-element dependencies, unequal resources, overlapping healthy progress, terminal blocked element, full-chain retry | [Repeated staged check](evidence/task07/staged-slurm-verified.json), arrays `65848218` and `65848222` | `b475acf` |
| Public CSV facade, managed screen, partial finalizer, preserved-success retry, all references reused | [Public API check](evidence/task07/slurm-verified.json), facade array `65848227`, screen arrays `65848232` and `65848234` | `b475acf` |

The existing UMA ownership utility is unchanged since Task 03. Its real water
workers requested four CPUs and ran with one Python process per element;
their server-sharing/shutdown evidence remains applicable. After fixing task
counts, the one-CPU element and staged fixtures were repeated at `b475acf`.
Task 07 also exercises the newly connected public interfaces at that revision.
Synthetic calculation frames in that fixture are explicitly marked
and isolated in its run-local reference store; they test scheduler plumbing
and finalization, not molecular energies or optimized geometries.

All repeated gates ran through HPC5 in the isolated checkout
`/lustre/hpc/kemi/jmni/dev/FRUST-slurm-arrays-task07`, updated from GitHub after
local commits/pushes. FRUST 0.1.0, Submitit 1.5.2, Slurm 26.05.4, partition
`kemi1`. No production checkout or ongoing production jobs were modified.

| Run directory under `/lustre/hpc/kemi/jmni/results/` | Completed behavior |
| --- | --- |
| `slurm-arrays-task07-elements-20261005` | Four elements, observed concurrency 2, three successes plus injected failure; collectors `65848224` and `65848226` completed, singleton `65848225` completed |
| `slurm-arrays-task07-staged-20261005` | Initialization `65848218` uses 1 CPU/2 GB/limit 3; frequency `65848222` uses 2 CPUs/4 GB/limit 1; both have 5-minute timeouts. A's frequency started 7.8 seconds before C's initialization finished. B's descendant was cancelled and collector `65848223` finished; retry `65848258 → 65848259`, collector `65848260`, returned A/B/C with A/C attribution preserved |
| `slurm-arrays-task07-20261005-v2` | Three facade targets in two elements, observed concurrency 2, collector `65848228`; legacy individual jobs `65848229–65848231`; partial screen finalizer `65848236`; TS1-only retry `65848265`, collector `65848266`, successful finalizer `65848267`; all references reused in a fresh screen, TS-only array `65848295`, collector `65848296`, successful finalizer `65848297` |

The public fixture's initial TS failure and first finalizer intentionally exit
nonzero while producing a partial report. Both final bundles later reported
`success`. Assertions verified complete branch collection with earlier
successes, unchanged scientific manifest, preserved attempt records after
cleanup, no second-rank log files, and rejection of a completed-screen retry
before submission. Synthetic shared references remained isolated to the run.

### Live-discovered fixes

The first public smoke at `3613bf7`, under
`/lustre/hpc/kemi/jmni/results/slurm-arrays-task07-20261005`, exposed duplicate
Python processes for one-CPU allocations. Slurm rounded the allocation to two
CPUs and Submitit launched two ranks because the task count was unspecified.
The first finalizer published references and cleaned up; the second found
missing target files and overwrote the report with `partial` despite both
processes exiting successfully. The
[second-rank finalizer log](evidence/task07/pilot-finalizer-rank1.txt) records
`global_rank=1(2)`. No scientific calculation was performed by this fixture.

`b475acf` explicitly requests one node and one Python worker for every FRUST
Slurm worker/control job, leaving `cpus_per_task` for calculator parallelism.
Conflicting task/node overrides are rejected. Tests inspect both parameter
paths and the generated sbatch headers. The complete affected live checks are
repeated with this fix; scheduler acceptance alone is not used as evidence of
successful finalization.

The same revision rejects retries of an already successfully finalized screen,
including after per-target cleanup. Regression checks cover standard and
screening artifact policies; the local fixture verifies rejection before any
new submission. Further calculations require a new output directory.

### Documentation and remaining boundaries

[Slurm Arrays, One Example At A Time](../../docs/cluster/arrays.md) starts with
the ten-target job counts, defines target/stage/job/element, and shows complete
validation, UMA SP batching, staged resource limits, records, collection,
explicit retries, complete screens, CSV submission, and local parity. The
cluster guide, workflow guide/tutorial, troubleshooting, navigation, and NumPy
public docstrings link or explain the same behavior. An obsolete local
`xtb_only` execution example was corrected to `single_job`.

Individual submission remains the default. Arrays require an explicit positive
limit, applied per array. Staged batching, automatic retries, intermediate
checkpoint resume, implicit array splitting, and adoption of old untracked
outputs remain unsupported. Legacy chain callables explicitly reject array
options. Scientific result quality still requires its normal FRUST review;
synthetic scheduler fixtures are not chemistry validation.

To reproduce the new fixture after updating an isolated HPC checkout from
GitHub, use the UMA Python and a fresh run root. Run these phases sequentially,
waiting for each phase's collector/finalizer jobs before the next phase:

```bash
PYTHONPATH=. python dev/slurm-arrays/evidence/task07/smoke.py \
    --partition kemi1 --output /path/to/new/task07-run --phase initial
PYTHONPATH=. python dev/slurm-arrays/evidence/task07/smoke.py \
    --partition kemi1 --output /path/to/new/task07-run --phase retry
PYTHONPATH=. python dev/slurm-arrays/evidence/task07/smoke.py \
    --partition kemi1 --output /path/to/new/task07-run --phase reuse
PYTHONPATH=. python dev/slurm-arrays/evidence/task07/smoke.py \
    --partition kemi1 --output /path/to/new/task07-run --phase verify
```

Use `--backend local` for the corresponding native local check. Each worker
performs seconds of synthetic work; the three screen phases use 1 CPU,
2 GB, and a 5-minute timeout for workers, collectors, and finalizers.
The CSV facade uses those worker resources and its ordinary default collector
resources (2 CPUs, 4 GB, 120-minute timeout).
