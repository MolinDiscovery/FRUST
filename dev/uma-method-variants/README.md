# UMA method variants — sequential implementation plan

One agent works through these tasks in order. Read this page and the next
pending task before editing code. Record concrete results in that task's
**Completion record**, then update the status table here. This table is the
progress log; no separate running diary is needed.

## The four workflow choices

`[C]` means that the TS reactive core is constrained. The constraints come
from an explicitly selected, reviewed TS-guess profile. They are released for
the final `OptTS` search.

| Choice | Screening and selection | Final barrier |
| --- | --- | --- |
| 1. Existing UMA screening | UMA SP and constrained UMA Opt select one candidate | ωB97 TS and reference calculations |
| 2. UMA final, one candidate | UMA SP and constrained UMA Opt select one candidate | UMA TS and reference calculations |
| 3. g-xTB then UMA ranking | g-xTB SP and constrained Opt, then UMA SP reranks those geometries | ωB97 TS and reference calculations; no UMA Opt |
| 4. UMA final, several candidates | UMA SP and constrained UMA Opt retain several distinct candidates | UMA TS and reference calculations, with a barrier for each acceptable candidate |

An optional ωB97 calculation on selected candidates and corresponding
references can compare final UMA and ωB97 barriers. It is a separate method
result; energies from the two methods must not be mixed in one barrier.

The new UMA stage blocks behind choices 2–4 are:

```text
A. UMA full result
TS guesses → prune → GFN-FF Opt [C] → UMA SP → filter → UMA Opt [C]
           → release constraints → UMA Hessian/OptTS → UMA NumFreq ─┐
References ───────────────────────────────→ UMA Opt → UMA NumFreq ──┤
                                                               UMA ΔE‡ and ΔG‡

B. UMA ranking after g-xTB
TS guesses → prune → GFN-FF Opt [C] → g-xTB SP → broad filter
           → g-xTB Opt [C] → UMA SP → rerank → ωB97 full result

C. Focused TS refinement inside A
Several retained guesses → UMA Opt [C] → release constraints
                        → UMA Hessian/OptTS → UMA NumFreq → mode review
```

Path C is the TS branch of A, not a third independent final method. The
constrained UMA optimization stays in place to settle the rest of the
structure before `OptTS`.

The built-in `omol@uma-s-1p2p1` potential can run in gas phase or with FRUST's
GFN2-xTB ALPB(chloroform) difference. The latter must use the same corrected
energy and gradient for SP, constrained optimization, TS search, and numerical
frequency displacements. No reviewed UMA ALPB geometry profile exists yet;
choose the reviewed UMA gas or ωB97 gas guess profile explicitly. Do not
rename gas constraints as an ALPB profile.

## Task order

| Task | Status | Depends on | Deliverable |
| --- | --- | --- | --- |
| [01 — Stage and result contracts](01-stage-and-result-contracts.md) | Complete | Existing UMA screening workflow | Public choices, stage labels, tier semantics, provenance, and minimal stage-plan scaffold |
| [02 — UMA TS refinement](02-uma-ts-refinement.md) | Complete | 01 | Constrained UMA Opt → released UMA OptTS → NumFreq with mode controls |
| [03 — UMA full references and analysis](03-uma-full-analysis.md) | Complete | 02 | UMA minima, thermochemistry, balanced ΔE‡/ΔG‡, and portable result quality |
| [03a — Per-candidate UMA barriers](03a-uma-candidate-barriers.md) | Complete | 03 | Portable barrier and quality for each retained TS candidate |
| [04 — g-xTB then UMA ranking](04-gxtb-uma-ranking.md) | Complete | 03a | UMA SP reranking on g-xTB optimized candidates; ωB97 path preserved |
| [04a — Optional ωB97 comparison](04a-optional-wb97-comparison.md) | Complete | 04 | Separate ωB97 characterization of selected UMA candidates and references |
| [05 — Bounded compute-node checks](05-compute-node-checks.md) | Running | 04a | Real gas/ALPB UMA checks, hybrid ranking, and a small comparison; submit then return control |
| [06 — User documentation](06-user-documentation.md) | Pending | 05 | Reader-facing examples verified against the completed checks |

The larger accuracy and candidate-recovery benchmark is **outside this
sequence**. A small real calculation here is a functional check. The optional
UMA ALPB TS-guess geometry profile remains a separate post-task in
[the earlier plan](../uma-workflow/08-optional-alpb-profile.md).

## Decisions and boundaries

- Preserve the existing `screening="uma-gas"` and
  `screening="uma-alpb-chloroform"` plus ωB97 behavior. Current runs and their
  `low_cost`, `dft_ranked`, and `full` analysis must remain readable.
- The final UMA path needs its own truthful stage and calculator labels.
  `dft_*` columns must not contain UMA results. An UMA-only `full` result is a
  method-specific characterization with modes and thermochemistry; it is not
  independent DFT validation.
- The hybrid path applies UMA SP **after** g-xTB constrained optimization.
  Keep enough diverse g-xTB candidates for UMA to rerank; expose the g-xTB
  and UMA selection limits separately.
- Frequencies are part of `full` only when ORCA returns the required thermal
  quantities. Never infer ΔG from electronic energy alone. TS mode review and
  minimum checks use the existing portable-run quality rules.
- Each submitted job starts at most one UMA server for all its UMA calls,
  including numerical frequencies. The client and server run on the same
  compute node and the server exits with that job.

## Working rules

1. Use the local FRUST development repository for library changes. Commit and
   push before updating the HPC checkout from GitHub. HPC run artifacts may be
   written separately, with backups before material changes.
2. Run project Python and tests in the `UMA` conda environment. Use small mock
   or saved-result tests for stage and analysis logic; reserve new chemistry
   calculations for Task 05.
3. Keep target expansion and chemistry identical between `wf.run(...)` and
   `wf.submit(...)`. `targets()` must remain calculation-free.
4. Do not use an unresolved TS3/TS4 ALPB mode to claim success. If a bounded
   real check runs long, record the job and output paths, return control, and
   resume review when the user reports results.
5. Keep scheduler details, revisions, and repair history in these development
   records. User-facing pages should lead with runnable inputs and interpretable
   outputs.
