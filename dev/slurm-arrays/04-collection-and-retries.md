# 04 — Collect per-target outputs and support explicit retries

## Example outcome

```text
Element 0: A succeeds, B raises, C succeeds
Element 1: D succeeds, allocation ends before E

Collection: A, C, D included; B failed; E missing/unattempted
Retry:      submit only B and E, optionally one target per element
```

## Work

1. Keep collection based on expected outputs for selected targets, using
   `afterany` so failed elements do not prevent a report. Preserve existing
   normal-termination checks, retention rules, and compact scientific outputs.
2. Join submission/attempt metadata with collection diagnostics to distinguish
   a target exception, scientific calculation failure, missing output, and work
   left unattempted by a terminated batch. Do not label an unexplained missing
   file as a known timeout without evidence.
3. Ensure successful targets in a failed element are collected individually.
   Retention must not delete records or evidence needed for failed/missing
   targets or hide scheduler failures. Collect independently of Submitit result
   pickle success.
4. Provide a concrete explicit retry workflow using existing `targets=` and
   suitable existing report/resume helpers. Add a small public selection helper
   only if necessary. Keep target identity stable, allow changed resources or
   batch size, and record previous/current attempt mappings.
5. Implement the safe retry output policy settled in Task 01. Validate that a
   stale final parquet cannot count as success for a new failed attempt; prevent
   overlapping attempts from writing the same target output. Preserve successful
   targets and meaningful earlier artifacts. Integrate existing screen reuse/
   resume contracts instead of bypassing them.
6. Define recollection explicitly: a collector submitted before a retry cannot
   wait for that future retry. A retry submission collects its selected targets;
   a final all-target collection must include preserved successes plus latest
   retry outcomes. Update reports without losing attempt history.

## Acceptance

- Tests cover mixed success within an element, interrupted batches, non-normal
  outputs, stale files, retry resource changes, overlapping attempts, and final
  merged results with no duplicate targets.
- A small local run follows the example above: first collection, explicit retry,
  final collection. Ordinary successful outputs are not recalculated.
- Old output directories remain readable and retention tests still pass.

## Completion record

Completed 2026-10-05 on `feature/slurm-arrays`: implementation `6369430`,
late-failure regression fix `5aae1a0`.

### How to retry

After the first submission's workers **and collector** finish, read its report:

```python
import json
from pathlib import Path
import frust as ft

report = json.loads(Path(first.collection_report).read_text())
print(report["retry_targets"])  # ["B", "D", "E"] in the example above
selected = [t for t in wf.targets() if t.tag in report["retry_targets"]]

retried = wf.submit(
    out_dir="runs/validation",  # same root as the first submission
    cluster=cluster,
    execution="single_job",
    targets=selected,
    retry=True,
    array=True,
    array_parallelism=2,
    targets_per_task=1,
    stage_resources={
        "single_job": ft.cluster.Resources(cpus=2, mem_gb=8, timeout_min=30),
    },
)
```

A and C keep their original files. The previous B/D/E directories move into
`.frust/history/<retry-attempt>/`; the retry writes fresh target directories.
The retry collector includes **only B/D/E** and writes separate files under
`.frust/retries/<retry-attempt>/`. It preserves the original merged result/report.

After the retry's workers and collector finish:

```python
complete = wf.collect("runs/validation", require_normal_termination=True)
# Reads preserved A/C plus the latest B/D/E outputs, once per target.
# Writes merged.parquet and collection_report.json at the run root.
```

`target_results` explains each target; `retry_targets` gives the tags requiring
attention. It does not submit anything automatically.

| Status | Meaning |
| --- | --- |
| `success` | Readable output; all available `*-NT` checks passed |
| `non_normal` | Output exists but at least one scientific normal-termination check failed |
| `failed` | The target raised a Python exception |
| `interrupted` | The target began but did not complete; unexplained termination has unknown cause |
| `unattempted` | A batch record confirms this target never started |
| `missing` | Expected output is absent; no more specific failure is recorded |
| `unreadable` | The output cannot be read as a parquet dataframe |
| `stale` | Output attribution does not match the selected attempt |
| `running` | The target began and completion has not been established |

Normal-termination checks remain scientific checks: a scheduler exit code does
not replace them. A successful target in a failed array element is still collected.
Reports include per-target attempt/job attribution and `attempt_history`; immutable
report snapshots and batch evidence stay under `.frust/` regardless of target
checkpoint retention. No scheduler columns are added to the scientific dataframe.

Retries validate target payload/metadata, workflow configuration, and method plan
before moving files. Resource requests, array limits, and batch size may change.
Successful targets and incompatible/legacy fingerprints are rejected. Active or
unverified earlier workers/collectors block overlapping writes. A filesystem lock
also prevents competing submissions/collections; an abandoned lock requires
inspection of its owner before manual removal.

### Checks and scope

- UMA checks: full fast suite **471 passed**, **13 deselected**; full slow suite
  **13 passed**. After adding missing-output and late-failure regressions, the final
  focused workflow/screen/array/batch/retry gate passed **133 tests**. The retry-only
  gate passed **9 tests**, including a real local Submitit retry/recollection cycle.
- Live Slurm run:
  `/lustre/hpc/kemi/jmni/results/slurm-arrays-task04-20261005`, using an isolated
  Git worktree updated from GitHub. Initial revision `6369430`: array `65844023`
  had two elements, three targets per full batch, concurrency limit 2, and requests
  of 1 CPU/2 GB/5 minutes per element. Both elements failed as injected; collector
  `65844024` completed and preserved A/C, reporting B failed, D interrupted,
  E unattempted. Retry revision `5aae1a0`: array `65844089` retried only B/D/E,
  one target per element, limit 2, with 2 CPU/4 GB/5 minute requests. All three
  elements and collector `65844091` completed. Final all-target collection contained
  A–E exactly once; A/C retained their original attempt attribution. Accounting
  confirms the failed first jobs, successful `afterany` collectors, and 2 GB to
  4 GB memory request change. The interruption is deliberately injected `SystemExit`,
  not a claim of a live timeout/cancellation check.
  [Portable evidence](evidence/task04/slurm-verified.json) and
  [runnable scheduler fixture](evidence/task04/smoke.py) are committed.
- Explicit retries currently support `BaseWorkflow.submit`, `single_job`, with
  standard artifacts. They rerun selected targets from the beginning; they do not
  resume intermediate checkpoints. Staged retries and screen retry/reuse integration
  remain for Tasks 05–06; existing staged screen manifest restart behavior remains.
- Legacy result directories remain collectable without attempt attribution. They
  cannot be adopted into the new safe retry contract; use a fresh output root.
- Submission records with ambiguous/unverified scheduler acceptance stay blocked;
  inspect the scheduler rather than assuming no job was accepted.

Reproduce the live cycle from a checkout containing this revision, setting
`PYTHONPATH=.` so the fixture and workers use that checkout:

```bash
PYTHONPATH=. python dev/slurm-arrays/evidence/task04/smoke.py \
    --phase initial --output /path/to/new/run --partition kemi1
# Wait for the printed collector job to finish.
PYTHONPATH=. python dev/slurm-arrays/evidence/task04/smoke.py \
    --phase retry --output /path/to/new/run --partition kemi1
# Wait for the retry collector job to finish.
PYTHONPATH=. python dev/slurm-arrays/evidence/task04/smoke.py \
    --phase verify --output /path/to/new/run --partition kemi1
```
