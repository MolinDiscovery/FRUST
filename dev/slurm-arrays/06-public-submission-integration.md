# 06 — Connect the public submission interfaces

## Example outcome

The same supported settings mean the same thing through each entry point:

```python
wf.submit(..., array=True, array_parallelism=4, targets_per_task=1)
ft.cluster.submit_jobs(..., array=True, array_parallelism=4, targets_per_task=1)
```

A catalyst screen may submit separate TS and reference arrays. Four running
elements in each array does not mean four running elements across the screen.

## Work

1. Forward the options through catalyst-screen submission, including relevant
   comparison workflows. Preserve reference reuse, branch-specific collection,
   finalization, portable reports, and screening artifact cleanup.
2. Integrate `ft.cluster.submit_jobs(...)` with the same adapter and sequential
   batch semantics. Audit pipeline-level UMA ownership before enabling shared
   batches there; reuse the scope rules from Task 03. Do not advertise sharing
   if a pipeline restarts its server for every target.
3. Integrate `submit_screen_chain`/`submit_chain_jobs` with staged one-target
   arrays where their contracts permit it. Use the common dependency policy and
   resource handling. If a legacy entry point cannot safely support an option,
   reject it explicitly and document the supported workflow alternative.
4. Preserve existing default calls, scheduler overrides that do not conflict,
   target naming, lazy namespaces, and result consumers. Keep helpers under
   `ft.cluster` rather than adding broad top-level aliases.
5. Account for screen branches with no calculated references and empty target
   selections. Keep collectors/finalizers as separate control jobs, outside the
   arrays; they use their own resources and completion dependencies.
6. Verify screen retry/reuse and retention preserve the submission records
   established in Tasks 01 and 04. Scheduler grouping must not alter scientific
   run signatures, calculated chemistry, or reference fingerprints.

## Acceptance

- Focused screen, facade, chain, artifact, and reference-reuse tests verify option
  forwarding and actual array submission, rather than only matching signatures.
- Existing entry points retain default behavior and old result consumers work.
- The completion record lists support/limitations per public entry point.

## Completion record

Completed 2026-10-05: implementation `c82a70e` on `feature/slurm-arrays`.

### Supported interfaces

| Entry point | Array support | Batching | Retry behavior |
| --- | --- | --- | --- |
| Workflow `wf.submit(...)` | Single-job and staged arrays from Tasks 02–05 | Sequential single-job batches; staged modes require one target per element | Selected failed targets; staged retries rerun the whole chain |
| `ft.workflows.catalyst_screen(...).submit(...)` | Separate arrays per calculated branch and stage group | Same rules as child workflows, including their UMA server scope | Failed targets selected from branch reports, then complete branches recollected with earlier successes |
| `ft.workflows.wb97_comparison(...).submit(...)` | Uses catalyst-screen submission, including seeded TS targets and reference reuse | Same child-workflow rules | Same managed screen retry contract |
| `ft.cluster.submit_jobs(...)` | One single-job array of existing prepared pipeline targets | Sequential pipeline calls within an element | Explicit failed target selection in tracked runs; retry collection contains the selected targets |
| `ft.cluster.submit_screen_chain(...)` and `frust.cluster.chains.submit_chain_jobs(...)` | Explicit rejection of nondefault array/retry options | Legacy defaults only | Use a workflow factory for tracked retries |

### Concrete examples

Submit a complete low-cost screen, then retry its failed targets after the
finalizer has finished:

```python
import frust as ft

screen = ft.workflows.catalyst_screen(
    dataframe=components, level="low_cost", ts_types=["TS1", "TS2"],
)
submitted = screen.submit(
    out_dir="runs/screen", cluster=cluster,
    array=True, array_parallelism=2, targets_per_task=2,
)
# Inspect run_report.json after submitted.finalization_job_id finishes.
# If the branch reports list failed targets, retry those targets:
retried = screen.submit(
    out_dir="runs/screen", cluster=cluster,
    array=True, array_parallelism=2, targets_per_task=2, retry=True,
)
```

For example, if TS1 fails and TS2 succeeds, the retry runs TS1. The new TS branch
parquet contains both TS1 and the preserved TS2 result. Successful reference
targets are not resubmitted. References originally marked `calculate` keep that
decision even if a compatible cached reference appears later.

An explicit retry selection is branch-specific:

```python
screen.submit(
    out_dir="runs/screen", cluster=cluster,
    array=True, array_parallelism=2, retry=True,
    targets={"transition_states": [0]},
)
```

Here `0` is the first target in
`screen.children()["transition_states"].targets()`. Omitted branches select no
retry work. An initial `targets={}` is a no-op; initial partial screens are
rejected because their manifest describes the complete screen. Submit a child
workflow directly for an initial subset.

The CSV facade retains its existing chemistry expansion:

```python
submitted = ft.cluster.submit_jobs(
    csv_path="molecules.csv", pipeline="run_mols_per_rpos",
    out_dir="runs/molecules", cluster=cluster,
    resources=ft.cluster.Resources(cpus=4, mem_gb=8, timeout_min=120),
    array=True, array_parallelism=2, targets_per_task=3,
)
```

If preparation produces six targets, this creates two worker elements, each
running three targets sequentially, plus a collector. The tracked outputs use
`runs/molecules/<target>/final.parquet`. Ordinary facade calls keep their old
flat output filenames. `pipeline="run_mols"` processes the whole CSV as one
target; it does not create an array element per CSV row. Tracked retries cannot
adopt old untracked facade output directories.

### Resources, collection, and retained records

- A staged screen accepts either one positive concurrency limit or a mapping
  containing every group name across all branches. Each child receives only
  its applicable group keys. Inspect `screen.show_stages(execution=...)` before
  constructing the mapping. Limits remain per array, not per complete screen.
- Collectors and the finalizer are separate ordinary jobs with their own
  resources. Dependencies cover all submitted work. Fully reused reference
  branches create no reference workers or collectors.
- Managed local submission waits for actual collector completion before
  starting the finalizer. It does not treat an older report as completion of
  the current attempt. The native local test verifies a partial finalizer
  after a failed target.
- Each managed screen attempt has its own branch report filenames and root
  finalizer record. Canonical reports are updated by the finalizer. Earlier
  run reports and child attempt records remain available. Overlapping writes
  are blocked until the prior finalizer is terminal or durably completed.
- Screen retries work with `artifact_policy="screening"` and deferred cleanup.
  Cleanup happens only after successful final validation; child ledgers and
  root attempt/completion records survive. Standalone workflow screening
  retries retain their existing explicit rejection.
- Scientific manifest signatures and reference fingerprints are unchanged by
  array limits or batching. Legacy default screen restarts retain their
  manifest-validated behavior, including `single_job` screens.

### Pipeline ownership and legacy chains

The two facade pipelines supported by `prepare_pipeline_inputs` are `run_mols`
and `run_mols_per_rpos`. Their current implementations use xTB/ordinary ORCA,
not UMA. The adapter invokes each pipeline sequentially with its own target
directory and forwards the allocated cores, ORCA memory fraction, scratch,
and calculation options. It does not advertise shared UMA ownership there.
Workflow factories remain the supported entry point for the UMA batches from
Task 03.

Legacy chain callables accept arbitrary stage functions and output conventions.
They lack the checkpoint attribution and attempt contract required for safe
arrays and retries. Nondefault options fail before inputs, directories, or
scheduler jobs are created. A supported alternative is:

```python
wf = ft.workflows.screen_ts(dataframe=components)
submitted = wf.submit(
    out_dir="runs/ts", cluster=cluster, execution="dft_staged",
    array=True, array_parallelism=2,
)
```

### Verification

Focused public/screen/chain/artifact/retry/UMA comparison tests: **102 passed**.
The new tests execute fake-scheduler workers and collectors, inspect actual
array calls and dependencies, and include a real Submitit local failure and
finalization run. They also verify facade retries, differing staged resources,
reference reuse, empty selection, invalid scheduler options before manifest
creation, cleanup record retention, and legacy default screen restarts.

Fast suite: **498 passed, 13 slow tests deselected**. Slow suite: **13 passed,
498 fast tests deselected**. Both ran in UMA; only existing dependency warnings
were reported.

```bash
conda run -n UMA python -m pytest -q
conda run -n UMA python -m pytest -m slow -q
```

No new Slurm jobs were submitted for this task. Task 05's live dependency and
cancellation evidence remains recorded in its completion record. Task 07
must check the newly connected public interfaces in the bounded integrated
live smoke run; local and fake-scheduler tests do not establish that gate.
