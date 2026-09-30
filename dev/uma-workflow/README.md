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
own reviewed geometry profile. An ALPB-specific profile remains deferred; task
05 must choose the initial-guess profile explicitly for corrected screening.
The corrected potential applies the difference to both energies and gradients:

```text
E = E_UMA + E_GFN2-xTB,ALPB(chloroform) - E_GFN2-xTB,gas
```

For a corrected run, the same composite potential must be used for UMA single
points, optimization, and numerical-frequency displacements. The ORCA input
and saved result metadata must identify the UMA model and whether gas phase or
ALPB(chloroform) was used. Record the initial-guess profile separately from
the calculation environment.

For an UMA screen, `level="dft_ranked"` runs the ωB97 ranking single point.
With `level="full"`, the ranking single point is independently configurable:
it can run before ωB97 validation or be skipped so UMA selection feeds
validation directly. Preserve existing r2SCAN-3c and ωB97 behavior. Task 05
uses `include_dft_rank_sp`, which defaults to `False` for UMA and `True` for
g-xTB. `dft_ranked` always runs the ranking single point.

## Task order and status

| Task | Status | Dependency | Main result |
| --- | --- | --- | --- |
| [01 — Audit](01-audit.md) | Complete | None | Version and behavior baseline, including the OET fork decision |
| [02 — Solvent correction](02-solvent-correction.md) | Complete | 01 | OET UMA energy/gradient correction with an explicit input option |
| [03 — Server lifecycle](03-server-lifecycle.md) | Complete | 01–02 | One reusable UMA server per submitted target job |
| [04 — Geometry profiles](04-geometry-profile.md) | Complete for gas; ALPB deferred | 02–03 | Reviewed UMA gas TS-guess profile for TS1–TS4 and INT3 |
| [05 — Workflow integration](05-workflow-integration.md) | Complete | 02–04 | Named UMA screening stages and ωB97 validation path; optional DFT ranking SP |
| [06 — Cluster smoke test](06-cluster-smoke-test.md) | Running; awaiting result review | 01–05 | Small compute-node end-to-end run; launch verified, review results after the user's update |
| [07 — Documentation](07-documentation.md) | Pending | 01–06 | User-facing UMA guide checked against the smoke-run artifacts |

After Task 07, [Post-task 08 — Optional ALPB geometry
profile](08-optional-alpb-profile.md) can revisit the unresolved TS3/TS4
ALPB saddles. It is outside the current completion path and includes no
screening benchmark.

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
| Build the UMA gas profile now; defer ALPB geometry calibration | The ALPB potential may be unsuitable for low-frequency TS3/TS4 references. Corrected screening may start from an explicitly selected gas or ωB97 guess profile without labelling it as an ALPB reference. |
| One UMA server per submitted target job | Reuse the loaded model across the job's UMA stages and numerical-frequency calls; stop it at job exit. |
| Use explicit UMA stage names and result labels | `xtb_sp` and `xtb_opt` would misdescribe UMA results. |
| Make DFT ranking SP optional in UMA `full` runs | `include_dft_rank_sp=False` is the UMA default; g-xTB retains `True`, and `dft_ranked` always runs the ranking SP. |
| Benchmark later | Large-scale accuracy and candidate-recovery work starts only after the workflow is functional. |

## Continuation note

When resuming in a later session, start at the first pending task above. The
Completion record in the previous task should identify the code revision,
checks performed, and any remaining limitation. Update this page only when a
decision or task status changes; no separate running diary is needed.
