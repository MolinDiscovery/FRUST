# 05 — Run bounded compute-node checks

## Goal

Verify the new stage paths with a few real calculations and server evidence.
This is a functional check; the later accuracy and candidate-recovery
benchmark is separate.

## Pass 1 — Submit and return control

1. Commit and push local FRUST library changes, then update the HPC checkout
   from GitHub. Never edit FRUST library code directly over SSH or in the
   mounted HPC checkout. Record FRUST, OET fork, UMA model, and environment
   versions in this task's Completion record.
2. Prepare small representative TS and reference inputs, starting with TS1
   and a reviewed gas guess/constraint profile. Exercise gas UMA full and
   ALPB UMA full sufficiently to check OptTS, NumFreq, thermochemistry, and
   the solvent flag. Keep TS3/TS4 ALPB outside this bounded check; their
   shallow modes are a separate chemistry problem.
3. Submit a small g-xTB → UMA SP → ωB97 path to confirm the hybrid selection
   handoff. Use `kemi1` and node066 when available; record actual allocation
   rather than assuming it. Avoid a broad screen or long benchmark.
4. Inspect early scheduler and process logs. Confirm each submitted job starts
   at most one UMA server on its compute node, all UMA client calls originate
   there and target that node's loopback address, and the saved ORCA files
   contain the intended gas/ALPB flags. Record job IDs, run paths, and a
   simple user progress command. **Once jobs are running correctly, return
   control.** Mark this task `running`; do not keep an agent turn open while
   waiting for chemistry results.

## Pass 2 — Review when results arrive

5. On the user's result update, collect and inspect stage outputs, server
   start/stop logs, thermal quantities, portable-run status, and saved result
   tiers. Confirm that NumFreq reused the job's UMA server and that no
   server remains after job exit.
6. Inspect the TS imaginary mode and reference minima. Use FRUST vibration
   views when needed; ask the user for chemical review if a mode remains
   ambiguous. Report `review` or `invalid` rather than claiming a valid
   barrier from successful process exits.
7. Fix any implementation issue locally, rerun only the necessary bounded
   check, and record the outcome here. Stop once the functional paths are
   verified; leave the comparative benchmark for later.

## Acceptance

- Both UMA environments and the hybrid stage handoff are evidenced by saved
  ORCA inputs, outputs, manifests, and correctly labelled result tiers.
- UMA TS/reference results and ΔE‡/ΔG‡ are produced where frequencies and
  quality permit; failures and unresolved mode review are explicit.
- One server per job serves all its UMA stages and NumFreq calls on the
  compute node, and stops with the job.

## Completion record

Pending. For the running phase, record submission and handoff details. After
review, record evidence paths, stage outcomes, quality, and any limitation.
