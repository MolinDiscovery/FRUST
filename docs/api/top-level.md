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
| Structure preparation | `ft.create_mol_per_rpos`, `ft.create_ts_per_rpos`, `ft.embed_mols`, `ft.embed_ts` |
| File IO | `ft.read_ts_type_from_xyz`, `ft.write_xyz`, `ft.write_xyz_structures` |
| Schema | `ft.normalize_dataframe`, `ft.energy_columns`, `ft.normal_termination_columns` |
| Visualization | `ft.plot_mols`, `ft.plot_conformers`, `ft.plot_row`, `ft.plot_vibs`, `ft.plot_energy_profile`, `ft.plot_regression_outliers` |
| Molecular viewers | `ft.MolTo3DGrid`, `ft.RxnTo3DGrid`, `ft.DrawMolSvg`, `ft.DrawUniqueChGrid` |
| Cluster | `ft.ClusterConfig`, `ft.Resources`, `ft.submit_jobs`, `ft.submit_chain`, `ft.submit_screen_chain` |

## Domain Namespaces

Use namespaces when the task belongs to a larger workflow domain:

| Namespace | Use |
| --- | --- |
| `ft.workflows` | Recommended workflow objects and method plans |
| `ft.screen` | Catalyst-screen input normalization, expansion, and TS generation |
| `ft.cluster` | Submission configuration and lower-level cluster helpers |
| `ft.pipes` | Supported compact helper workflows |
| `ft.pipelines` | Explicit staged pipeline modules |
| `ft.vis` | Complete visualization API, including reusable scene helpers |
| `ft.utils` | Curated dataframe, molecule, analysis, and IO helpers |
| `ft.tsguess2` | Current connected-SMILES TS guess backend |

Direct helpers are intentionally curated. Broader discoverability belongs under
the stable namespaces rather than mirroring every module symbol into `ft`.
