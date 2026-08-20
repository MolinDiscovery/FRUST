# Top-Level API

The recommended notebook import is:

```python
import frust as ft
```

FRUST resolves this curated API lazily, so importing the package does not load
every calculator and visualization dependency immediately.

## Common Direct Helpers

| Area | Direct helpers |
| --- | --- |
| Calculation | `ft.Stepper` |
| Dataframe inspection | `ft.show_steps`, `ft.show_timing`, `ft.lowest_energy_rows`, `ft.map_substrate_names` |
| Conformer pruning | `ft.prune_conformers` |
| Vibration analysis | `ft.inspect_ts_vibrations`, `ft.summarize_ts_vibrations` |
| Structure preparation | `ft.create_mol_per_rpos`, `ft.embed_mols`, `ft.show_spec_profiles` |
| File IO | `ft.read_ts_type_from_xyz`, `ft.write_xyz`, `ft.write_xyz_structures` |
| Results and schema | `ft.result_column`, `ft.get_result`, `ft.get_free_energy`, `ft.upgrade_dataframe`, `ft.upgrade_legacy_constraints`, `ft.normalize_dataframe`, `ft.energy_columns`, `ft.normal_termination_columns` |
| Visualization | `ft.plot_mols`, `ft.plot_conformers`, `ft.plot_row`, `ft.plot_vibs`, `ft.plot_energy_profile`, `ft.plot_regression_outliers` |
| Molecular viewers | `ft.MolTo3DGrid`, `ft.RxnTo3DGrid`, `ft.DrawMolSvg`, `ft.DrawUniqueChGrid` |
| Cluster | `ft.ClusterConfig`, `ft.Resources`, `ft.submit_jobs`, `ft.submit_screen_chain` |

## Domain Namespaces

Use namespaces when the task belongs to a larger workflow domain:

| Namespace | Use |
| --- | --- |
| `ft.workflows` | Recommended workflow objects and method plans |
| `ft.screen` | Catalyst-screen input, portable run analysis, and inspectable reference libraries |
| `ft.structures` | Calculation-free MOLS/INT3 generation, typed targets, state registry, planning, and deferred builders |
| `ft.cluster` | Submission configuration and lower-level cluster helpers |
| `ft.pipes` | Supported compact helper workflows |
| `ft.pipelines` | Explicit staged pipeline modules |
| `ft.vis` | Complete visualization API, including reusable scene helpers |
| `ft.utils` | Curated dataframe, molecule, analysis, and IO helpers |
| `ft.tsguess2` | Current connected-SMILES TS guess backend |

Direct helpers are intentionally curated. Broader discoverability belongs under
the stable namespaces rather than mirroring every module symbol into `ft`.

For calculation-free geometry inspection, keep the chemistry domain explicit:

```python
systems = ft.screen.expand(ft.screen.read("screen.csv"))

mols = ft.structures.create_mols(
    systems,
    states=["HH", "int1", "int2"],
    n_confs=1,
)
int3 = ft.structures.create_int3_guesses(systems, n_confs=1)
```

Both calls return canonical embedded-structure dataframes. They do not run xTB
or DFT.
