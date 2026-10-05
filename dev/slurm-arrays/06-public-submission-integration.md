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

Pending. Record revision, supported-interface table, tests, and limitations.

