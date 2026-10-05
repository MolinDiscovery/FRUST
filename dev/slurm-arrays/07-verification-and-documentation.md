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

Pending. Record revision, test/build results, Slurm evidence paths, and limits.
