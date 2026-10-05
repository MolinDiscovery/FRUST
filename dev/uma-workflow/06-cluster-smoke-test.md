# 06 — Run a small cluster end-to-end check

## Goal

Verify the finished UMA screen and ωB97 validation path on the cluster with a
small representative target. This is a functional integration check, not the
later accuracy benchmark. User-facing documentation is Task 07.

## Work

Work in two passes. The first pass ends once the jobs are demonstrably running;
the second begins when the user reports that they have finished. Do not keep an
agent turn open to poll a long-running calculation.

### Pass 1 — Prepare, submit, and return control

1. Commit and push local FRUST library changes, then update the HPC checkout
   from GitHub. Install the recorded OET fork revision and UMA checkpoint on
   the compute-node-accessible filesystem.
2. Submit a small catalyst-screen target on `kemi1`, preferably node066 when
   available, through ALPB-corrected UMA selection and the ωB97 `full` path.
   Keep enough ORCA inputs and outputs to verify the solvent flag, stages,
   result labels, and server placement. Check gas-phase UMA selection with a
   separate small calculation. Avoid a broad screening campaign.
3. Select the ranking-SP-disabled setting for the small full run. Confirm from
   the scheduler and early logs that each job is running on a compute node,
   its UMA client and server are on that node, and the intended potential and
   guess profile were selected. Record job IDs, artifact paths, and commands
   the user can use to check progress. Then return control to the user without
   waiting for the calculations to finish. Mark Task 06 as **running**, not
   complete, and state which results still need review.

### Pass 2 — Review after the user's update

4. When the user says the jobs have finished, collect their outputs. Verify
   that no DFT ranking SP ran, that the full validation stages completed or
   reported a chemical failure clearly, and that the UMA server exited with
   the job. Cover the ranking-SP-enabled path with the focused workflow test
   from Task 05; no second full cluster run is needed for that switch.
5. Record the observed stage sequence, solvent and profile provenance, result
   labels, server evidence, test results, and any limitations in the Completion
   record. Mark Task 06 complete only after this review.

## Acceptance

- Saved small runs demonstrate both UMA environment choices and their explicitly
  recorded guess profiles. An ALPB-specific geometry profile is deferred; do
  not describe a gas or ωB97 profile as an ALPB reference. The ALPB run also
  demonstrates the full validation path, input settings, stage names, metadata,
  and compute-node-only server communication.
- The server is gone after the job. Numerical-frequency reuse has already
  passed task 03's focused check.
- The relevant functional tests pass in the UMA environment.
- Any chemistry-specific failure in the small target is reported accurately;
  the integration check does not conceal it as a passing validation result.

## Completion record

**Completed 2026-09-30 as a functional integration check. The original
finalization had an invalid full barrier; the run-local ligand repair below
leaves the full barrier at `review` pending TS1 review.** The HPC checkout was updated from
GitHub to FRUST `21d0346` on `feature/uma-screening`. The pinned OET fork is
`1b4fcda`, using
`/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu`. Both runs use the
`wb97xd3-631g/gas` guess/constraint profile explicitly; this is not an ALPB
geometry profile.

| Check | Submitted job(s) | Status at handoff |
| --- | --- | --- |
| ALPB full TS1 target | Initial job `65689162`; later dependent jobs `65689163`, `65689164`, `65689166`, `65689168` | UMA SP and optimization finished; ωB97 preoptimization running on node066. |
| ALPB full references | Initial jobs `65689170`, `65689174`, `65689178`, `65689182`; dependent DFT jobs through `65689185` | Initial UMA stages ran on node066; DFT jobs started. |
| ALPB collectors and finalizer | `65689169`, `65689186`, `65689187` | Pending upstream completion. Wait for finalizer `65689187` before opening the full result bundle. |
| Gas TS1 low-cost check | `65689189`; collector `65689191` | Both completed on node066 with exit code 0. One final row contains `uma_sp-EE`, `uma_opt-EE`, and `uma_opt-oc`. |

The run root is
`/lustre/hpc/kemi/jmni/results/uma-task06-smoke-20260930`. Each mode has
`submission.json`, its input CSV, Submitit logs, client-call logs, saved ORCA
inputs, and persistent server logs. The ALPB full run also has `plan.csv` and
a `workflow/manifest.json` recording `include_dft_rank_sp: false`, the
resolved guess profile, and `uma_xtb_alpb: chloroform`.

The saved ALPB `uma_sp` ORCA input contains
`-m uma-s-1p2p1 ... --xtb-alpb chloroform`; the gas input uses the same UMA
model without `--xtb-alpb`. Early client audit records only
`host=node066.cluster` and `127.0.0.1` binds. The five ALPB initial jobs each
started one server on node066 with a distinct PID and local bind. The gas job
made 19 client calls to its one server, PID `765593`, which logged a clean
stop at job exit. The ALPB TS1 low-cost tier already contains one selected
row with `uma_opt-EE` as its analysis energy. These are launch and screening
checks; full DFT validation and ALPB server cleanup still require pass 2.

To check progress from `h -5`:

```bash
squeue -j 65689187,65689191 -o "%.18i %.10T %.10M %.20R"
sacct -j 65689187,65689191 --format=JobIDRaw,State,Elapsed,NodeList,ExitCode
```

The submission script is
[`evidence/task06/submit_smoke.py`](evidence/task06/submit_smoke.py).
Focused local checks passed with
`conda run -n UMA python -m pytest tests/test_uma_screening_workflow.py tests/test_uma_job_scope.py -q`
(11 passed).

### Pass 2 — completed result review

All target, collection, and finalization Slurm jobs finished on `node066` with
exit code `0:0`, including finalizer `65689187`. The TS branch collected 1/1
target and the reference branch 4/4, each with zero missing, skipped, or
errored outputs. Every `*-NT` flag in their final result rows is true. The TS
path completed `uma_sp` → `uma_sp_filter` → `uma_opt` → ωB97 `dft_preopt` →
`dft_hessian` → `dft_ts_opt` → `dft_freq` → `dft_solv_sp`; the molecule
references completed their ωB97 optimization, frequency, and solvent stages.
The completed rows contain no `dft_rank_sp-EE` column, and the run has
`low_cost` and `full` tiers only.

All 9 saved ALPB UMA single-point inputs and all 5 optimization inputs contain
`--xtb-alpb chloroform`. The gas check's 2 single-point and 1 optimization
inputs omit it. Both modes use `omol@uma-s-1p2p1`; the full TS result records
`guess_profile=wb97xd3-631g/gas` independently. The result contract resolves
the full analysis energy to `dft_solv_sp-EE`, the full TS geometry to
`dft_ts_opt-oc`, and the low-cost analysis energy to `uma_opt-EE`.

The five ALPB server logs each contain exactly one start and one stop for
their job PID. All 62 audited ALPB client calls were made from
`node066.cluster` to those five matching `127.0.0.1` binds. The gas job made
19 local client calls and logged one start and stop. Saved TS stage provenance
shows that `uma_sp` and `uma_opt` reused the same server PID (`765594` ALPB;
`765593` gas). A separate post-job PID probe could not get a new node066
allocation while the node was busy; the launcher stop events and completed
Slurm jobs are the cleanup evidence.

The [run report](evidence/task06/run-review/run_report.json) has
`overall_status="partial"`. It correctly marks the ligand **invalid** because
its ωB97 frequency has one imaginary mode at **−65.61 cm⁻¹**, rather than the
zero required for a minimum. TS1 has one imaginary mode at **−1077.84 cm⁻¹**
but remains **unreviewed**, so its status is `review`. The full barrier is
therefore **invalid** (`dependency_invalid:ligand` and
`dependency_review:TS1`), despite the completed numerical energy. The
low-cost barrier tier is `ready`. No large-scale benchmark or ligand repair was
attempted in this functional check.

The compact [review evidence](evidence/task06/run-review/) contains the
[Slurm status](evidence/task06/run-review/slurm_status.txt), collection and
analysis reports, four saved UMA ORCA input examples, client/server logs, and
the small analysis tables. Full calculator outputs and all checkpoint parquets
remain in the cluster run root. Task 07 must present this as a successful
workflow execution whose original finalization had an invalid full scientific
barrier, not as a validated barrier result.

### Run-local ligand repair

After the integration review, the free N-methylpyrrole ligand was recalculated
on the Mac with 10 ORCA cores. The seed displaced the original ωB97 geometry
by 0.30 Å along its −65.61 cm⁻¹ methyl torsion. The same ωB97 optimization,
analytic frequency, and chloroform SMD single-point stages all terminated
normally. The lowest frequency is now **+24.97 cm⁻¹**; the optimized energy is
−249.415200659364 Eh, 0.080 kcal/mol below the original stationary point.

The corrected DFT values and their Mac calculator provenance were merged into
the run-local ligand checkpoints and the reference aggregate tables. Original
files were backed up under `workflow/repair_backups/ligand_methyl_torsion_mac_20260930/`
before replacement. The shared reference library was not changed. Rebuilt
analysis now classifies the ligand `ready`, all four reference states `ready`,
TS1 `review`, the low-cost barrier `ready`, and the full barrier `review`.
This is still not a validated full barrier until TS1 is reviewed. The
[post-repair report](evidence/task06/run-review/post-repair/repair_report.json)
and [analysis report](evidence/task06/run-review/post-repair/analysis_report.json)
record the change; the preceding snapshot remains as evidence of the original
cluster finalization.
