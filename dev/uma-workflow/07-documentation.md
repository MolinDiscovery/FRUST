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

## Acceptance

- A reader can run a small gas or ALPB UMA screen, select the intended guess
  profile, and understand what each saved result tier contains.
- The OET fork explanation reflects the verified version and behavior rather
  than assuming an upstream fix.
- Examples and caveats agree with the Task 06 artifacts, and the strict docs
  build passes.

## Completion record

Pending. Record the FRUST revision, pages changed, Task 06 artifacts used to
verify examples, strict build result, and any unresolved documentation limits.
