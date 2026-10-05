# Energy profiles with missing data points

**Status:** Implemented · **Created:** 2026-10-05

## Target behavior

```python
import numpy as np
import frust as ft

profiles = {
    "TMP": [("Reactants", 0.0), ("TS1", 27.7), ("Int1", 3.8),
            ("TS2", 21.2), ("Product", -0.3)],
    "Pip": [("Reactants", 0.0), ("TS1", np.nan), ("Int1", 7.3),
            ("TS2", 22.5), ("Product", -0.3)],
}

fig, ax = ft.plot_energy_profile(
    profiles,
    annotate_energies=True,
    overlay_annotate="energy",
    show_state_labels=True,
)
```

Expected: TMP retains its normal FRUST curve. Pip has a gap at TS1; its
Int1–TS2–Product segment retains the same curve style. Its isolated Reactants
point remains visible. All states keep their reaction-coordinate positions,
and only available energies receive numerical labels.

These numbers are an illustrative subset of the p26 benchmark, not a complete
atom-balanced profile.

## Problem

`plot_energy_profile` currently sends nonfinite energy values to
`PchipInterpolator`, which raises `ValueError: y must contain only finite
values`. This prevents plotting an otherwise useful incomplete profile.

In p26, the notebook therefore switches an entire theory panel to a separate
Matplotlib renderer if any catalyst has a missing point. WB97, g-xTB and
r²SCAN-3c become straight dashed profiles, while complete UMA profiles use
FRUST's smooth curves. Styling, axes and annotations differ unnecessarily.

The task is to handle gaps inside FRUST's existing plotting API so callers
can use the same renderer for complete and incomplete profiles.

## Required behavior

- Accept explicit missing energies represented by `np.nan` or `None`, while
  retaining their state labels and positions. Continue rejecting infinity and
  malformed energy values with informative errors.
- Draw each contiguous run of available energies separately. Never join,
  interpolate, extrapolate or substitute values across a missing state.
- Keep the existing interpolation and styling for runs of two or more
  available points. Draw an isolated available point as a marker.
- Determine gaps independently for each curve. One incomplete catalyst must
  not change the appearance of complete catalysts or other panels.
- Keep overlay alignment, catalyst colours, legend entries, axis formatting
  and existing annotation controls consistent. Missing points receive neither
  energy text such as `nan` nor numeric markers; their state labels remain.
- Handle missing first/last states and missing Product values. Draw a
  main-to-product or side-reaction connector only when its required endpoints
  and intervening segment are available; do not create a bridge over a gap.
- Handle an entirely missing curve or panel without an interpolation error
  or nonfinite axis limits. Define and document how the existing legend and
  state labels behave in these cases.
- Preserve existing output for fully finite profiles, including product
  reference behavior, side reactions, label placement and duplicate-energy
  handling. Missing reference energies must not invent relative values or
  suppress available overlay annotations through invalid comparisons.

> A missing tuple is not necessarily an explicit missing point. Preserve the
> existing semantics of differently shaped profiles; use labelled missing
> values to express a gap. Do not infer unprovided energies or chemistry.

Frequency or reaction-mode quality flags remain the caller's responsibility.
The caller can pass a missing value when a point should be excluded. FRUST
should not decide whether a calculated transition state is scientifically
acceptable.

## Implementation work

1. Inspect parsing, shared state-position alignment, interpolation,
   annotations and connectors under `frust/vis/energy_profile/`.
2. Normalize supported missing values without losing state identity. Split
   finite runs while keeping their original x positions and segment boundaries.
   Reuse existing drawing behavior rather than introducing a second style.
3. Apply the same finite-value handling to overlays, product references,
   connectors, label comparisons and automatic limits. Avoid replacing NaNs
   with zero or simply dropping states before assigning x positions.
4. Add focused regression coverage in `tests/test_vis.py`, asserting actual
   artist coordinates, annotation contents and the absence of lines across
   missing states. Include complete and incomplete overlays together.
5. Update `docs/visualization/energy-profiles.md` and the public NumPy docstring
   with the example above and the precise missing-data behavior. If adding a
   figure, generate it reproducibly from the same example and verify it visually.

## Acceptance checks

- Interior gap, consecutive gaps, missing endpoints, one available point,
  and all-missing input render without crashing.
- `None` and `np.nan` produce equivalent explicit gaps; infinities remain
  invalid. There are no `nan`/`None` energy annotations.
- Complete and incomplete catalyst overlays keep their original aligned
  state positions, colours and curve style. Complete curves are unaffected.
- Product-reference and side-reaction cases retain correct labels and never
  draw misleading connectors across missing states.
- Render an example representative of p26: complete UMA curves, WB97 Pip
  missing TS1, and r²SCAN-3c Pip missing TS4. Confirm consistent appearance
  and visible gaps, with numerical labels on available points.
- Run focused tests in UMA:

  ```bash
  conda run -n UMA python -m pytest tests/test_vis.py
  ```

- After documentation changes, run `conda run -n UMA mkdocs build --strict`.

## Scope

This file records the task; it does not implement the change. General label
collision avoidance and common axis scaling across separately composed panels
are separate improvements. Do not change the p26 energies, calculation results,
or notebooks as part of this task unless separately requested.

Library implementation must follow the local repository → GitHub → HPC update
workflow; do not edit the mounted cluster library checkout.

## Implementation result

`None` and `np.nan` now reserve labelled gaps. Available runs retain PCHIP
curves, isolated states remain markers, and connectors and Product references
require finite values. All-missing profiles retain configured legends and state
labels with finite axes. Shorter profiles still keep their previous semantics.

Regression coverage checks artist coordinates, annotations, overlays, endpoints,
consecutive gaps, all-missing curves, side pathways, and Product references.
The documentation includes the example above and a reproducible figure; the
asset script also renders illustrative UMA/WB97/r²SCAN-3c panel cases without
changing benchmark data or notebooks.
