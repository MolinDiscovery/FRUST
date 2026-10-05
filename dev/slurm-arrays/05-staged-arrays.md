# 05 — Add staged arrays with matching-target dependencies

## Example outcome

```text
Optimization array              Frequency array
0: target A (16 CPUs, 32 GB) --> 0: target A (8 CPUs, 64 GB)
1: target B (16 CPUs, 32 GB) --> 1: target B (8 CPUs, 64 GB)
2: target C (16 CPUs, 32 GB) --> 2: target C (8 CPUs, 64 GB)
```

There are three targets, two arrays, and six jobs. A's frequency job becomes
eligible when A's optimization succeeds; it does not wait for B or C.

## Work

1. Extend the planner/adapter to `dft_staged` and `fully_staged`, submitting one
   array per stage group. Use the existing group definitions and resource lookup;
   do not hard-code chemistry stages. A group may contain multiple stages.
2. Preserve identical target ordering and element indices across groups. Use
   `aftercorr:<upstream_array_id>` for corresponding-element success dependency.
   Handle single-target ordinary jobs with normal `afterok` dependencies.
3. Apply each group's `stage_resources` and concurrency limit independently.
   Accept scalar limits or the complete group mapping defined in Task 01.
   Explain that two stages can run simultaneously, each with its own limit.
4. Reject `targets_per_task>1` before creating jobs. Do not change target order
   or regroup between stages. Different batch sizes across stages are deferred.
5. Make failure terminal: a failed upstream target must block its downstream
   work without leaving pending jobs that prevent collection forever. Verify a
   supported invalid-dependency cancellation mechanism on the target Slurm
   version/configuration (for example `kill_on_invalid_dep`) and configure it
   explicitly, or implement a bounded cancellation mechanism. Fail clearly when
   the chosen guarantee cannot be provided.
6. Collect after all relevant worker arrays have terminated, including failed
   upstream arrays and cancelled downstream work. Parent `afterany` dependencies
   are suitable only once blocked descendants become terminal. Distinguish
   upstream failure from downstream work blocked by that failure in the report.
7. Implement equivalent local ordering and failure propagation explicitly.
   Current local executor configuration ignores Slurm dependencies; verify that
   no worker reads a checkpoint before its predecessor finishes. Avoid waiting
   downstream workers consuming all local execution slots and causing deadlock.
8. Specify explicit staged retries. Reuse only validated compatible checkpoints;
   otherwise rerun the failed target's complete chain. If retrying from a group
   boundary is unsupported, document that limitation instead of implying resume.

## Acceptance

- Tests cover both execution modes, mapping/index stability, distinct resources,
  per-group limits, single-target fallback, rejected batching, and dependency IDs.
- Local staged integration proves checkpoint order, independent target progress,
  blocked descendants, and a collection report after upstream failure.
- Fake scheduler tests inspect cancellation/dependency settings. Task 07 must
  build on a live check completed here: two tiny staged arrays with different
  resource requests, matching-target dependencies, one injected upstream failure,
  terminal blocked descendants, and a finished collector. Record job IDs, states,
  timestamps, and reports. Follow local -> GitHub -> HPC; leave this gate pending
  if access/configuration is unavailable.

## Completion record

Pending. Record revision, dependency/cancellation policy, checks, and limitations.
