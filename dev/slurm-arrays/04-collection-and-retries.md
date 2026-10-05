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

Pending. Record revision, runnable retry example, test results, and limitations.

