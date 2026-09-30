# Post-task 08 — Optional UMA ALPB(chloroform) geometry profile

> **Optional, after Task 07.** This task is a later attempt to complete an
> ALPB-specific `tsguess2` reference profile. It does not block UMA screening,
> full ωB97 validation, or the functional smoke test. Do not start these
> calculations as part of Tasks 05–07.

## Starting point

The existing ALPB-corrected UMA reference run used `omol@uma-s-1p2p1` with
the GFN2-xTB ALPB(chloroform) minus gas correction for both optimization and
numerical frequencies. The saved review summary is in
`fruits/results-2026/uma-task04-references-20260929/review-summary.json`, with
the individual `*_alpb-chloroform.parquet` files alongside it.

| State | Existing ALPB imaginary modes (cm⁻¹) | Next action |
| --- | --- | --- |
| TS1 | mode 0: −1100.67 | Recheck the saved reactive-mode review; retain if sound. |
| TS2 | mode 0: −333.33 | Recheck the saved reactive-mode review; retain if sound. |
| TS3 | modes 0–2: −151.03, −35.59, −22.41 | Resolve the two unwanted shallow modes and verify TS3 identity. |
| TS4 | modes 0–1: −166.39, −35.94 | Resolve the unwanted shallow mode and verify TS4 identity. |
| INT3 | none | Recheck the saved minimum and retain if sound. |

The successful **gas** TS3 retry started from a displaced ALPB TS3 geometry
and finished with one imaginary mode. Earlier Hessian-guided gas retries failed
because ORCA could not find `private_input.hess`; that was an input-path
failure, not evidence that the ALPB potential cannot give a first-order saddle.
The gas result makes a careful ALPB retry worthwhile, but does not guarantee
one. The gas and ALPB potentials have different stationary points.

## Work

1. Reopen the existing ALPB parquets and saved `ft.plot_vibs`/py3Dmol viewers.
   Verify atom order, chemical roles, normal termination, and which imaginary
   mode is reactive for each state. Preserve the accepted gas profile as a
   separate reference; never relabel gas coordinates as ALPB.
2. Focus new calculations on TS3 and TS4. Try a small number of documented,
   chemically motivated displacements along their unwanted shallow modes,
   followed by direct **ALPB-corrected** UMA `OptTS` and `NumFreq`. The tested
   retry workflow in `evidence/task04/retry_references.py` is a starting point.
   If needed, try the accepted gas saddle as an alternative seed, then
   reoptimize and recompute frequencies with the ALPB potential. Record each
   seed, mode index, displacement sign and scale, and result. Limit exploratory
   retries to two well-motivated attempts per unresolved state before review.
3. Run each calculation inside a compute-node job with one reusable UMA server
   for optimization and numerical-frequency calls. Use `node066` on `kemi1`
   when available, and verify that client and server remain on the compute
   node. Keep the pinned UMA/OET versions visible in the job record.
4. Review every candidate before activation: exactly one imaginary mode for a
   TS, none for INT3, intended reactive motion in an animated py3Dmol view,
   plausible core geometry, and correct atom/role mapping. Rotate the viewer
   to inspect the three-dimensional arrangement. Compare frequencies with
   matched ωB97 references where available; a matched TMP/thiophene ωB97
   comparison for TS3/TS4 is still missing. Ask the user to inspect any
   ambiguous TS3/TS4 assignment.
5. Only if all five ALPB states pass review, extract the ALPB candidate JSON,
   register a distinct `omol-uma-s-1p2p1/alpb-chloroform` profile, and test
   exact resolution, constraint values, and generated guesses. Document the
   calculation environment separately from the profile used to construct the
   initial guesses. Keep a profile incomplete or unregistered if any required
   state remains unresolved.

## Stop and report

If the bounded retries still leave extra modes or uncertain chemical identity,
record the attempted seeds and results and stop. Continue to use an explicitly
selected gas UMA or ωB97 guess profile for ALPB screening. A gas optimization
followed by an ALPB single point can be evaluated as a screening choice, but
it is **not** an ALPB-optimized geometry profile. This post-task does not
include the later scientific screening benchmark.

## Completion record

Pending. Start only after Task 07 and a separate decision to pursue this
optional calculation work. Record job IDs, source parquets, review viewers,
accepted mode indices and frequencies, profile status, and focused tests here.
