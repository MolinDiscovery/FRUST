# 07 — Document and review the UMA workflow

## Goal

Give users a practical guide to choosing gas or ALPB(chloroform) UMA screening,
understanding its results, and running ωB97 validation. Use the completed
Task 06 smoke check to verify the examples and describe any observed limits.
This task is documentation work; it does not start another screening campaign.

## Work

1. Review the existing UMA material in the setup, catalyst-screen, workflow,
   and vibration guides. Update the OET fork information using the Task 01
   findings. State the tested UMA, OET, and FRUST revisions where useful.
2. Lead with a small `ft.workflows.catalyst_screen(...)` example. Show the
   input, gas and ALPB screening choices, the `uma_sp` → `uma_sp_filter` →
   `uma_opt` sequence, and the resulting `*-EE` and `*-oc` columns. Explain
   `low_cost`, `dft_ranked`, and `full` with a compact table.
3. Show `include_dft_rank_sp=False` and `True` for a full run, including which
   saved result tiers exist. Explain that the default UMA full run skips the
   ranking SP, while a `dft_ranked` request always performs it.
4. Show how to choose the guess/constraint profile independently of the UMA
   potential. State that the reviewed UMA gas profile can be selected
   explicitly and that no reviewed UMA ALPB profile exists yet. Do not label
   gas or ωB97 constraints as ALPB references.
5. Use the Task 06 artifacts to check the example ORCA input, solvent option,
   server behavior, stage labels, result metadata, and any chemistry-specific
   failure description. Keep claims tied to what the smoke run actually
   demonstrated. Run `conda run -n UMA mkdocs build --strict` and fix broken
   links or examples.

> **Task 06 result:** The original cluster finalization marked the full
> barrier `invalid` because the ligand had a −65.61 cm⁻¹ methyl torsion. A
> subsequent run-local Mac reoptimization removed it (+24.97 cm⁻¹ lowest
> frequency), so the rebuilt full barrier is `review`: TS1's single
> −1077.84 cm⁻¹ imaginary mode remains unreviewed. The guide must distinguish
> workflow success from a scientifically accepted barrier.

## Acceptance

- A reader can run a small gas or ALPB UMA screen, select the intended guess
  profile, and understand what each saved result tier contains.
- The OET fork explanation reflects the verified version and behavior rather
  than assuming an upstream fix.
- Examples and caveats agree with the Task 06 artifacts, and the strict docs
  build passes.

## Completion record

Completed 2026-09-30. The workflow example was checked against the current
FRUST API, and its measured values come from the Task 06 run executed at FRUST
`21d0346`, OET fork `1b4fcda`, and `omol@uma-s-1p2p1`. The practical guide is
[`docs/catalyst-screens/uma-screening.md`](../../docs/catalyst-screens/uma-screening.md).
It shows a two-row input, gas and ALPB screening, three result levels, both
full-run ranking-SP settings, gas guess-profile choices, and review status.

Updated the external-tool setup box and lower-level UMA guide for the verified
fork, FairChem runtime, explicit model selection, ALPB energy/gradient formula,
ORCA `Ext_Params`, and job-scoped server behavior. Added links from the
catalyst-screen overview, end-to-end guide, TS guess guide, and workflow-method
guide; the vibration guide now includes a portable run review example.

The numerical example and caveats were checked against Task 06's
[`run-review`](evidence/task06/run-review/) snapshot and
[`post-repair`](evidence/task06/run-review/post-repair/) records. The saved
ALPB ORCA input shows `--xtb-alpb chloroform`; the server audit shows
compute-node loopback calls and one server lifecycle per initial job.
`conda run -n UMA mkdocs build --strict` passed. A construction-only API check
confirmed `wf.show_stages()` has the documented stage and solvent columns and
the default UMA full run has `("low_cost", "full")` tiers.

The run remains a functional smoke check with TS1 at `review`; no accuracy
benchmark or scientific TS approval was added by this documentation task.
