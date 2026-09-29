# 06 — Run a small end-to-end check and document UMA use

## Goal

Verify the finished UMA screen and ωB97 validation path on the cluster with a
small representative target, then document how to run and inspect it. This is
a functional integration check, not the later accuracy benchmark.

## Work

1. Commit and push local FRUST library changes, then update the HPC checkout
   from GitHub. Install the recorded OET fork revision and UMA checkpoint on
   the compute-node-accessible filesystem.
2. Run a small catalyst-screen target on `kemi1`, preferably node066 when
   available, through ALPB-corrected UMA selection and the ωB97 `full` path.
   Keep enough ORCA inputs and outputs to verify the solvent flag, stages,
   result labels, and server placement. Check gas-phase UMA selection with a
   separate small calculation. Avoid a broad screening campaign.
3. Select the ranking-SP-disabled setting for this small full run and verify
   that no DFT ranking SP ran. Check that the full validation stages completed
   or reported a chemical failure clearly, and that the UMA server exited
   with the job. Cover the ranking-SP-enabled path with a focused workflow
   test in task 05.
4. Update user-facing setup and workflow docs with a compact example showing
   the input, stage sequence, output columns, gas/ALPB choice, and both
   ranking-SP settings. Update the OET fork explanation based on task 01.
   Run `mkdocs build --strict` after doc changes.

## Acceptance

- Saved small runs demonstrate both UMA environment choices and their matching
  profiles. The ALPB run also demonstrates the full validation path, input
  settings, stage names, metadata, and compute-node-only server communication.
- The server is gone after the job. Numerical-frequency reuse has already
  passed task 03's focused check.
- The relevant tests and strict documentation build pass in the UMA environment.
- Any chemistry-specific failure in the small target is reported accurately;
  the integration check does not conceal it as a passing validation result.

## Completion record

Pending. Add the FRUST and OET revisions, cluster job and artifact paths,
observed stage sequence, server evidence, test results, and limitations here.
