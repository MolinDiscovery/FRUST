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

Pending. Add the FRUST and OET revisions, cluster job and artifact paths,
observed stage sequence, server evidence, test results, and limitations here.
