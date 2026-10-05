# 01 — Define the API and submission plan

## Example outcome

```text
Selected targets: A, B, C, D, E
targets_per_task: 2
Planned jobs:     [A, B], [C, D], [E]
```

Planning this must not embed molecules, start UMA, or run a calculator.

## Work

1. Audit the current submit signatures and result consumers in workflows,
   catalyst screens, the cluster facade, and chain submission. Check installed
   Submitit behavior/version in UMA, including `map_array`, single-element
   arrays, local execution, and deferred job IDs in `batch()`.
2. Specify and introduce `array=False`, `array_parallelism=None`, and
   `targets_per_task=1`. For arrays, require positive integer limits, rejecting
   booleans as integers. Reject non-default array options when `array=False`.
   In staged mode, accept a scalar limit or a complete mapping from active
   stage-group names to limits; reject unknown/missing groups. Initially reject
   staged batching above one and mapping limits in `single_job` mode.
3. Build a lightweight deterministic submission plan from selected targets,
   execution groups, resource settings, and batch size. Preserve selection
   order, reject duplicate output identities, and support an empty selection.
   Prefer an internal plan; do not add a public planning API without a need.
4. Extend `JobSubmissionResult` compatibly with explicit target-to-job records
   and array parent IDs. Keep `job_ids` as unique submitted worker-job IDs in
   submission order; `tags`/`save_dirs` describe targets. Do not imply these
   lists zip together for batched or staged runs. Specify records for target,
   group, batch membership, job ID, array ID/index, output, and attempt identity.
5. Define atomic, versioned on-disk submission records, reusing suitable
   existing run infrastructure. Keep scheduler metadata separate from the
   scientific screen manifest/signature. Record submitted IDs incrementally so
   a later submission failure does not erase knowledge of already queued work.
   Mark planned, submitted, and submission-failed work distinctly.
6. Document these contracts in the completion record, including supported
   entry points, local behavior, retry output policy, and size-limit handling.
   New options not yet executable must raise a clear error until later tasks
   connect them; never accept an option and silently ignore it.

## Acceptance

- Focused tests cover deterministic batching, uneven final batches, invalid
  values, duplicate targets, empty selections, stage mappings, and serialization
  compatibility. Planning never calls expensive preparation.
- Existing default submission tests pass unchanged in behavior.
- Metadata preserves individual, batched, and staged target/job relationships
  without widening scientific dataframes.

## Completion record

Completed 2026-10-05, based on FRUST `0888f44` (local changes).

- Added the three options to `BaseWorkflow.submit`, an internal lightweight
  planner, `SubmissionRecord`, and compatible result fields. Scalars and complete
  staged mappings validate as specified. Unknown, duplicate, unsafe, and invalid
  values fail before filesystem/executor side effects. Planning is deterministic
  and does not call embedding or calculators.
- Submission records are atomic version-1 JSON files under
  `<out_dir>/.frust/submissions/<attempt_id>.json`. Every submission gets a new
  attempt identity. Accepted job IDs are saved before the next submission;
  scientific manifests/signatures and dataframe columns are unchanged. The file
  tracks submission state, not calculation success. Scheduler errors may leave
  accepted work: inspect the scheduler before retrying an ambiguous failure.
- `job_ids` describes worker jobs; `tags`/`save_dirs` describe selected targets;
  `records` is the explicit mapping. Actual array IDs come from Slurm element
  IDs; ordinary/local jobs have no array parent. Legacy result construction is
  unchanged because the new fields have defaults.
- Audited Submitit 1.5.3 in UMA: `map_array` returns one job per input, Slurm
  uses an ordinary job for one input, and `batch` defers IDs and submissions to
  context exit. Local map submission does not throttle workers. Screen and chain
  result consumers currently use named fields, not tuple unpacking.
- Current support is `BaseWorkflow.submit`. Screen wrappers, facade, and chain
  interfaces retain their old signatures until Task 06; attempts to pass new
  options there fail rather than being ignored. Task 02 enables single-job
  arrays; staged arrays and target batching remain explicitly unsupported.
- Retry policy for Task 04: preserve earlier successful targets and evidence;
  reject overlapping writes, validate checkpoint compatibility, and associate
  collected outputs with the relevant attempt so stale files cannot imply retry
  success. This safety logic is not implemented in Task 01; there is no automatic
  retry/resume guarantee.
- Array-size policy: never auto-split. Task 02 adds an optional known cluster
  limit; otherwise expose scheduler rejection with target-selection guidance.
- Checks: `conda run -n UMA python -m pytest tests/test_cluster_arrays.py
  tests/test_workflows.py tests/test_cluster_screen_chain.py
  tests/test_screening_artifacts.py -q` passed at the Task 01 gate (82 tests).
  Task 02's expanded tests subsequently passed 91 tests.


