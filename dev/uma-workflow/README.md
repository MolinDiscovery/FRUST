# UMA screening workflow plan

This plan is for one agent working through the tasks in order. Read this page and
the next pending task before starting. After finishing a task, record the result
in that task's **Completion record** and update the table below. The task files
are the work requests; this page is the only progress record.

## Target workflow

```text
200 conformers -> RMSD filter -> constrained GFN-FF optimization
               -> UMA single-point selection -> constrained UMA optimization
               -> ωB97 full validation
```

The proposed default UMA screen uses the OMol task and an xTB implicit solvent
correction for chloroform. Gas-phase UMA is also a selectable screen, with its
own geometry profile, so the solvent correction can be evaluated separately.
The corrected potential applies the difference to both energies and gradients:

```text
E = E_UMA + E_GFN2-xTB,ALPB(chloroform) - E_GFN2-xTB,gas
```

For a corrected run, the same composite potential must be used for UMA single
points, optimization, and numerical-frequency displacements. The ORCA input
and saved result metadata must identify the UMA model and whether gas phase or
ALPB(chloroform) was used. Select the corresponding geometry profile.

For an UMA screen, `level="dft_ranked"` runs the ωB97 ranking single point.
With `level="full"`, the ranking single point is independently configurable:
it can run before ωB97 validation or be skipped so UMA selection feeds
validation directly. Preserve existing r2SCAN-3c and ωB97 behavior. Task 05
settles the public switch name and its UMA default before implementation.

## Task order and status

| Task | Status | Dependency | Main result |
| --- | --- | --- | --- |
| [01 — Audit](01-audit.md) | Complete | None | Version and behavior baseline, including the OET fork decision |
| [02 — Solvent correction](02-solvent-correction.md) | Complete | 01 | OET UMA energy/gradient correction with an explicit input option |
| [03 — Server lifecycle](03-server-lifecycle.md) | Complete | 01–02 | One reusable UMA server per submitted target job |
| [04 — Geometry profiles](04-geometry-profile.md) | Pending | 02–03 | Separate reviewed UMA gas and ALPB(chloroform) TS-guess profiles |
| [05 — Workflow integration](05-workflow-integration.md) | Pending | 02–04 | Named UMA screening stages and ωB97 validation path |
| [06 — Final smoke test and docs](06-smoke-and-docs.md) | Pending | 01–05 | Small cluster end-to-end run and user-facing guidance |

The scientific benchmark against full ωB97 results is **outside this plan**. A
small calculation used to prove that a feature works is a functional test, not
a screening benchmark.

## Working rules

1. Work on the local FRUST repository. Commit and push FRUST changes to GitHub
   before updating the HPC checkout; do not edit mounted HPC library source.
2. Keep OET fork changes in its local development repository and record the
   exact revision installed for the cluster tests. Do not silently substitute
   a different OET or UMA version midway through the sequence.
3. Use the `UMA` conda environment for project Python and tests. Use focused
   tests while iterating, then run the appropriate broader checks for changed
   paths.
4. At each gate, record commands or artifact paths and the observed result in
   the task's Completion record. If a gate fails, keep that task open and note
   the blocker before moving on.
5. Run cluster tests inside a compute-node allocation. Use node066 on the
   `kemi1` partition for the focused server check when available. Record both
   client and server hostnames and process IDs.

## Decisions already agreed

| Decision | Reason |
| --- | --- |
| UMA is the screening method; ωB97 is the main full-validation method | Earlier r2SCAN-3c results showed problems; its existing workflow remains supported. |
| Use an xTB ALPB chloroform difference on top of UMA | UMA's current OET wrapper has no built-in solvent flag; the correction must supply matching energy and gradient changes. |
| Build separate UMA gas and ALPB(chloroform) profiles | The ALPB result may be unsuitable; each potential needs its own calculated and reviewed geometry references. The proposed workflow uses ALPB by default, with gas phase selectable. |
| One UMA server per submitted target job | Reuse the loaded model across the job's UMA stages and numerical-frequency calls; stop it at job exit. |
| Use explicit UMA stage names and result labels | `xtb_sp` and `xtb_opt` would misdescribe UMA results. |
| Make DFT ranking SP optional in UMA `full` runs | Keep `dft_ranked` available and allow a full run both with and without the ranking SP. The UMA default is still to be decided. |
| Benchmark later | Large-scale accuracy and candidate-recovery work starts only after the workflow is functional. |

## Continuation note

When resuming in a later session, start at the first pending task above. The
Completion record in the previous task should identify the code revision,
checks performed, and any remaining limitation. Update this page only when a
decision or task status changes; no separate running diary is needed.
