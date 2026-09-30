# 02 — Add UMA TS refinement

## Goal

Turn retained TS guesses into unconstrained UMA stationary points with
numerical frequencies. Keep the constrained UMA optimization before the TS
search so the rest of each candidate can settle.

## Work

1. Reuse or extend the existing `uma_opt` constrained screening stage. Accept
   several retained candidates per chemical target, keep their parent row and
   guess/constraint profile, and preserve structural diversity through the
   handoff. Do not silently choose only the lowest screening energy.
2. Release the reactive-core constraints after `uma_opt`. Build a UMA Hessian
   or numerical-frequency seed as needed by ORCA, then run UMA `OptTS` and a
   final UMA numerical frequency. Decide whether the seed Hessian can be
   reused and record the decision in the saved stage plan.
3. Use the same UMA model and environment for SP, constrained Opt, TS search,
   and frequency displacements. In ALPB mode the GFN2-xTB chloroform
   correction must affect both energies and gradients at every geometry;
   inspect the generated ORCA files to confirm the flag is present.
4. Reuse one job-scoped UMA server across all stages and numerical-frequency
   calls in a submitted job. Ensure failure and cancellation stop it. Do not
   start a second server when switching from optimization to NumFreq.
5. Carry the final geometry, frequencies, displacement vectors, and
   provenance into the portable result. Apply the existing TS quality checks:
   one imaginary mode is necessary, and its displacement must be reviewable
   against the intended reaction coordinate. Keep a `review` result when the
   mode is ambiguous; a calculator exit code alone is insufficient.

## Acceptance

- The path is `UMA Opt [C] → release constraints → UMA Hessian/OptTS → UMA
  NumFreq`, with distinct UMA stage and result names selected in Task 01.
- Multiple candidates can be refined and traced to their source guesses.
  Gas and ALPB variants use the requested potential consistently.
- Focused tests cover constraint release, stage order, server reuse and
  cleanup, saved frequency data, and quality flags. Use mocks or saved data
  here; Task 05 is the bounded live chemistry check.

## Completion record

Pending. Record implementation choices, tests, and any ORCA limitation here
before starting Task 03.
