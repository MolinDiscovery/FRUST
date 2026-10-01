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
   handoff. Include one bounded optional ωB97 comparison on an UMA-selected
   candidate to check that method-specific barriers remain separate. Use
   `kemi1` and node066 when available; record actual allocation rather than
   assuming it. Avoid a broad screen or long benchmark.
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

- Both UMA environments, the hybrid stage handoff, and the optional comparison
  are evidenced by saved ORCA inputs, outputs, manifests, and correctly
  labelled result tiers.
- UMA TS/reference results and ΔE‡/ΔG‡ are produced where frequencies and
  quality permit; failures and unresolved mode review are explicit.
- One server per job serves all its UMA stages and NumFreq calls on the
  compute node, and stops with the job.

## Completion record

**Running, submitted 2026-10-01.** This is Pass 1 only. Chemistry results,
imaginary modes, thermochemistry, and server cleanup still need Pass 2 review.

The local FRUST branch and HPC checkout were updated through GitHub to
`d39b441` (`feature/uma-screening`) before the first submission and
`02a2864` before the repaired gas/ALPB submission. The OET fork is
`1b4fcda`; the compute-node runtime is
`/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu` with OET
`2.0.1.dev10+g1b4fcda92`, FairChem `2.23.0`, and CPU Torch `2.13.0+cpu`.
The submission environment uses Python `3.12.8`. Every UMA job requests
`omol@uma-s-1p2p1`. TS guesses use the reviewed
`omol-uma-s-1p2p1/gas` profile; this is **not** an ALPB geometry profile.

All three checks use one N-methylpyrrole/TMP boron catalyst system at `rpos=2`
and TS1. The gas full path admits up to two refined candidates; the ALPB full
path admits one. Both calculate ligand, dimer, HBpin, and H₂ references.
The hybrid TS-only path retains two g-xTB optimized geometries for UMA SP
reranking before ωB97 refinement. Its original run root is
`/lustre/hpc/kemi/jmni/results/uma-method-variants-task05-20261001`.
The repaired gas/ALPB run root is
`/lustre/hpc/kemi/jmni/results/uma-method-variants-task05-20261001-repair1`.

| Check | Jobs | Pass 1 state |
| --- | --- | --- |
| Gas UMA full TS and references, repaired | TS `65713420`; references `65713422`–`65713425`; collectors `65713421`, `65713426`; finalizer `65713427` | Started on node066 at FRUST `02a2864`; reference jobs crossed into `uma_min_opt` and `uma_freq`. |
| ALPB UMA full TS and references, repaired | TS `65713431`; references `65713433`–`65713436`; collectors `65713432`, `65713437`; finalizer `65713438` | TS started on node066 at FRUST `02a2864` and entered UMA SP; references remained queued at handoff. |
| g-xTB → UMA SP → ωB97 TS, original | Stage jobs `65713293`, `65713295`, `65713298`, `65713300`, `65713302`; collector `65713304` | Initial job ran g-xTB Opt and UMA SP reranking on node066, selecting one of two geometries; ωB97 refinement started. |
| Optional ωB97 comparison, repaired | Dependent launcher `65713462` | Queued with `afterany:65713427`; after gas finalization it submits one selected UMA candidate and independent ωB97 references. Child job IDs will be in `comparison/submission.json`. |

The first gas/ALPB full jobs **failed** after their early UMA stages because
the screening stages requested two calculation cores and the later full UMA
stages requested four. The job-scoped UMA server correctly rejected a second
server configuration. All original full TS/reference jobs failed; the original
finalizers `65713287`, `65713306` and comparison launcher `65713307` were
canceled. The hybrid jobs did not enter the failing full UMA stage transition
and were left running. FRUST `02a2864` now pins the server core budget to the
job allocation while retaining each stage's own calculation core setting.
The focused UMA tests passed (24), followed by the full fast suite (407 passed,
13 slow tests deselected). Repaired submissions were made only after the HPC
checkout was fast-forwarded through GitHub.

The original five gas jobs each wrote **one** server-start record with a
distinct PID and `hostname=node066.cluster`; all binds were `127.0.0.1`.
Audited gas and hybrid clients also called node066 loopback binds. Early
ALPB clients likewise used node066 loopback binds. Saved ALPB TS and reference
ORCA inputs contained `--xtb-alpb chloroform`; gas inputs omitted it.
The repaired five gas target jobs also each wrote one server log with
`server_cores=4`, one distinct loopback bind, and `slurm_job_nodelist=node066`.
Audited reference client calls retained the same bind as they entered
`uma_min_opt` and `uma_freq`. Recheck completion of OptTS/NumFreq and server
cleanup against the repaired jobs in Pass 2. The repaired ALPB TS started one
server on node066 before the handoff; inspect its later UMA stages and reference
jobs as they progress.

The [submission scripts](evidence/task05/submit_checks.py) and
[dependent comparison launcher](evidence/task05/submit_comparison.py) are
committed. Run artifacts stay in the HPC result root. From a cluster login
node, check the principal jobs with:

```bash
squeue -j 65713427,65713438,65713304,65713462 -o "%.18i %.12T %.12M %.20R"
sacct -j 65713427,65713438,65713304,65713462 --format=JobIDRaw,State,Elapsed,NodeList,ExitCode
```

When results arrive, review the two full UMA bundles, the hybrid tier
handoff, the comparison launcher and its child jobs, saved ORCA gas/ALPB
flags, all server start/stop records, and TS/reference vibration quality.
