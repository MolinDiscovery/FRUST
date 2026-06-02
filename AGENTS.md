# Project Environment

Use `conda activate UMA` to test things in this project. It should have most packages / software.

# Documentation Style

When writing documentation, prefer "show, then explain" over abstract description. The goal is that a reader can understand the workflow by looking at the examples, tables, structures, and outputs before reading much prose.

## User-Facing Docs

For user-facing docs:
* Start with the practical mental model or concrete artifact the user should understand.
* Show realistic FRUST inputs and outputs. For dataframe workflows, prefer compact tables that show the relevant columns and values.
* Keep code examples focused on the action being taught. Avoid long setup or dataframe-construction blocks unless constructing the dataframe is the lesson.
* Use example output blocks, directory trees, before/after snippets, and small mapping tables to make behavior obvious.
* Explain just enough around the example so the user can generalize it.
* Prefer current FRUST naming conventions in examples, especially `substrate_name` and canonical stage columns such as `*-oc`, while mentioning legacy names only when useful.
* Use admonitions for important defaults, caveats, or safety notes. Do not use admonitions for maintainer-only details such as how documentation assets are generated.
* Keep documentation reader-focused. Do not expose repository internals, asset paths, or build details unless the reader needs them to use FRUST.

## Public API Style

For user-facing examples, prefer the lazy top-level FRUST namespace:

```python
import frust as ft
```

Use this for common public helpers such as `ft.Stepper`, `ft.show_steps`,
`ft.lowest_energy_rows`, `ft.summarize_ts_vibrations`, molecule builders, file IO helpers, and
visualization helpers. Larger workflow domains should stay namespaced, for
example `ft.pipes.run_mols(...)` and `ft.cluster.submit_jobs(...)`. Deeper
imports remain valid when documenting a specific module, but the top-level
namespace is the preferred notebook/user-facing style.

The top-level namespace is deliberately two-tiered:
* Use direct `ft.<helper>` aliases for the curated, common notebook/toolbox API,
  such as `ft.Stepper`, `ft.show_steps`, `ft.lowest_energy_rows`,
  `ft.write_xyz`, `ft.plot_mols`, `ft.plot_vibs`,
  `ft.summarize_ts_vibrations`, `ft.create_mol_per_rpos`, and
  `ft.create_ts_per_rpos`.
* Use stable lazy domain namespaces when the example or explanation is about a
  specific area: `ft.vis.plot_mols(...)`, `ft.utils.write_xyz(...)`,
  `ft.pipes.run_mols(...)`, `ft.cluster.submit_jobs(...)`, and staged pipeline
  modules such as `ft.pipelines.run_ts_per_rpos.run_init(...)`.

Do not mirror every module symbol into `ft.<name>`. Keep direct top-level
aliases curated for common user-facing helpers, and keep broader discoverability
under stable namespaces like `ft.vis` and `ft.utils`. Do not make repository
layout details part of the public API just because a module exists.

Do not expose or recommend internal helpers, provenance internals, parser/layout
internals, or private functions as part of the public API.

## DataFrame And Provenance Design

Keep FRUST dataframes compact, especially the main Stepper calculation tables.
Add new dataframe columns only when the value is useful for most rows and most
downstream analysis. Avoid adding sparse, stage-specific, provenance-only, or
single-workflow columns that will mostly contain missing values after the next
calculation step.

Prefer `df.attrs` for audit/provenance metadata that describes how a dataframe
was produced, filtered, embedded, merged, or submitted. Surface that metadata
through compact inspection helpers such as `ft.show_steps(df)` instead of
widening the primary dataframe. If a summary belongs in a helper table, prefer
using existing broad columns such as `options`, `input_rows`, `output_rows`, or
`dropped_rows` before adding many narrow columns.

For example, conformer-generation details belong in
`df.attrs["frust_conformers"]`; `ft.show_steps(df)` should summarize them in the
`initial_conformers` row rather than adding sparse columns such as
`n_confs_requested` or `n_confs_generated` to every calculation step.

## Catalyst Screen And TS Guess Architecture

The catalyst-screen workflow is dataframe-first:

```python
components = ft.screen.read("screen.csv")
systems = ft.screen.expand(components)
ts_guesses = ft.screen.create_ts_guesses(systems, ts_types=["TS1", "TS2", "TS3", "TS4"])
```

`frust.screen` owns user-facing CSV/component handling. It normalizes `role`,
`smiles`, optional `compound_name`, and substrate-only `rpos`, then expands
substrates and catalysts into explicit systems.

`frust.tsguess` owns TS construction. Built-in `TSSpec` objects define role
coordinates, distance/angle constraints, legacy `constraint_atoms` order, and
optional fragments such as H, H2, or HBpin. Assembly assigns atom roles
dynamically from the catalyst B-aryl-N scaffold, substrate `rpos`, and built-in
fragments. Do not add fixed atom-index logic for new chemistry.

Generated rows are self-describing: `constraint_roles` maps chemical roles to
atom indices, `constraint_spec` stores role-based distances/angles,
`coords_embedded` stores the guess geometry, and `connectivity_bonds` is only
stored graph/display connectivity. Optimizers should use `constraint_spec`, not
display bonds, to decide constrained geometry.

For TS2-TS4, embedding uses hard reactive-core anchors plus soft
substrate/HBpin frame anchors inspired by the TMP templates. When extending
this chemistry, preserve dynamic role assignment and add regression tests for
topology, hydride placement, frame orientation, and core metric deltas.

## Workflow Method Plan Architecture

The `frust.workflows` module is split by responsibility. Keep calculator
choices in `frust.workflows.methods`, concrete chemistry workflows and target
construction in `frust.workflows.factories`, and shared local/cluster execution
plumbing in `frust.workflows.core`.

Workflow targets should stay lightweight. `targets()` is for inspection and
scheduler preparation; it must not run expensive embedding or calculators. The
first expensive structure/dataframe construction step belongs in
`_prepare_initial_df(...)`, where it runs during `wf.run(...)` or inside the
submitted cluster job.

Preserve local/cluster parity. `wf.run(...)` and `wf.submit(...)` should use
the same target objects and stage definitions, so a local smoke test exercises
the same chemistry and method plan as production submission. `MethodPlan`
changes calculator engines/options only; it must not change chemistry target
expansion.

Keep workflow provenance compact. Put audit details in `df.attrs` and surface
them through helpers such as `ft.show_steps(df)` instead of adding sparse
workflow-only columns to the main calculation dataframe.

For public examples, use `ft.workflows.<factory>(...)` and
`ft.workflows.methods.<helper>(...)`. Do not expose private workflow helpers as
the documented API.

## Molecular And Structural Examples

Choose the representation that matches the concept being taught:

| Representation | Use when the point is |
| --- | --- |
| SMILES | molecular identity, input format, substrate lists, or pipeline starting points |
| 2D drawings | connectivity, atom labels, reactive positions, symmetry-unique sites, or functional groups |
| 3D views | geometry, conformers, transition states, atom distances/angles, vibrations, or exported XYZ structures |
| dataframe tables | how FRUST stores or moves molecular data between workflow stages |

For molecular and structural docs:
* When a structure is generated from SMILES, make the transformation explicit if it matters: `SMILES -> embedded molecule -> atoms + coords_embedded -> optimized *-oc -> XYZ or 3D view`.
* Prefer FRUST’s own visualization utilities for figures and interactive views: `DrawMolSvg`, `DrawUniqueChGrid`, `MolTo3DGrid`, `RxnTo3DGrid`, `plot_mols`, `plot_row`, or `plot_vibs` as appropriate.
* Use external chemistry tools such as RDKit or py3Dmol only when FRUST does not already provide the needed capability, or when they are only the backend used to create a FRUST-compatible example.
* Make visuals match the exact structures, dataframe rows, or examples being discussed. Avoid generic placeholder molecules when the docs are about specific structures.
* Do not add figures as decoration. Add them when the spatial arrangement, molecular identity, or visual output is part of what the user needs to understand.
* Keep interactive 3D views compact and focused. Avoid linked viewers unless synchronized rotation is the point of the example.

## Analytical Figures

FRUST can create analytical figures such as energy profiles and regression plots. Treat these like structural visuals: they should teach the workflow by showing the input data, the FRUST call, and the resulting figure.

Choose the plotting tool that matches the concept being taught:

| Figure type | Use when the point is |
| --- | --- |
| Energy profile | relative energies, barriers, reaction stages, competing pathways, or side reactions |
| Regression plot | agreement between two methods, scaled vs reference energies, outliers, or benchmark quality |

For energy-profile docs:
* Prefer FRUST's own `plot_energy_profile` for reaction coordinate diagrams.
* Show the compact input first, usually as a table with state labels and relative energies.
* Make the figure correspond exactly to the shown states and energies.
* Use chemically meaningful labels and units, usually kcal/mol.
* For overlays or side reactions, show which rows or pathways become which curves.
* Avoid decorative profiles. Add an energy profile when the barrier ordering, pathway comparison, or reaction progression is part of what the user needs to understand.

For regression-plot docs:
* Prefer FRUST's own `plot_regression_outliers` for benchmark and scaling figures.
* Show the dataframe columns being compared before showing the plot.
* Be explicit about what each axis represents, including whether values are scaled, DFT, xTB, free energies, or electronic energies.
* Be precise about metrics. In FRUST regression plots, distinguish direct prediction error from fit residuals. If reporting RMSD for method agreement, use the direct paired error between `x` and `y` after any intended scaling, not the residual around the fitted regression line.
* Align paired data before plotting or computing metrics. Missing values should be dropped consistently across both compared columns.
* Show outlier labels only when they help the reader identify molecules or reactions; otherwise avoid clutter.

## Generated Documentation Assets

For generated documentation assets:
* Prefer reproducible asset-generation scripts over one-off notebook/manual exports when a doc page embeds images, SVGs, HTML viewers, XYZ files, or analytical plots.
* Put asset-generation scripts in `scripts/` with names like `build_<topic>_assets.py`.
* The script should build the same molecules, dataframes, structures, or plots used in the documentation examples.
* Prefer FRUST’s own path for generating assets. For example, use `embed_mols`, `Stepper.build_initial_df`, `write_xyz`, and `MolTo3DGrid(export_HTML=...)` when documenting dataframe-to-XYZ-to-3D workflows.
* For analytical figures, use the same dataframe rows, energy columns, labels, and FRUST plotting function shown in the docs, such as `plot_energy_profile` or `plot_regression_outliers`.
* Use RDKit or py3Dmol only as supporting backends when needed, not as a replacement for FRUST’s public workflow.
* Generated assets should be committed when the docs need to render without running code, but the docs should not explain asset-generation internals unless the reader needs them.
* Keep asset viewers compact and focused. For 3D grids, set `cell_size`, `columns`, and `linked` deliberately; avoid linked viewers unless synchronized rotation is part of the lesson.
* After generating assets, run `mkdocs build --strict` to verify links and embedded outputs.
