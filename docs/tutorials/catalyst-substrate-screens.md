# Catalyst And Substrate Screens

This tutorial follows one catalyst screen from component CSV to an inspectable
workflow object. The dedicated [Catalyst Screens](../catalyst-screens/overview.md)
section contains the full input, TS-guess, and execution reference.

The path is:

```text
component CSV
    -> normalized components
    -> substrate-catalyst systems
    -> tsguess2 dataframes
    -> inspect one target
    -> local smoke test
    -> staged cluster run
```

## 1. Read The Example Screen

FRUST includes a compact documentation input at
[`docs/examples/screen.csv`](../examples/screen.csv):

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,,pyrrole
substrate,COC1=CC=CO1,methoxyfuran,"3,5",furan
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

```python
import frust as ft

components = ft.screen.read("docs/examples/screen.csv")
components[["role", "compound_name", "rpos"]]
```

| role | compound_name | rpos |
| --- | --- | --- |
| substrate | `n_methyl_pyrrole` |  |
| substrate | `methoxyfuran` | `3,5` |
| catalyst | `tmp_bcat` |  |

Blank substrate `rpos` means “use symmetry-unique aromatic C-H positions.” A
single integer or a comma/semicolon-separated value selects explicit RDKit atom
indices. Draw labels from the exact SMILES before entering them:

```python
ft.DrawUniqueChGrid(components.loc[components["role"].eq("substrate"), "smiles"])
```

## 2. Expand Components Into Systems

```python
systems = ft.screen.expand(components)
systems[["system_name", "substrate_name", "catalyst_name", "rpos"]]
```

| system_name | substrate_name | catalyst_name | rpos |
| --- | --- | --- | --- |
| `n_methyl_pyrrole__tmp_bcat` | `n_methyl_pyrrole` | `tmp_bcat` |  |
| `methoxyfuran__tmp_bcat` | `methoxyfuran` | `tmp_bcat` | `3,5` |

This explicit table is useful before any geometry work. With multiple catalysts,
`ft.screen.expand(...)` creates every substrate-catalyst pair and prefixes extra
metadata with `substrate_` or `catalyst_`.

## 3. Generate One-Conformer TS Guesses

Use one conformer while checking the wiring:

```python
ts_guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
)

ts_guesses.keys()
```

The default backend is `tsguess2`. It builds a connected TS-like graph, assigns
v2 roles, anchors those roles to the built-in coordinate specification, and
embeds the conformer.

```python
ts1 = ts_guesses["TS1"]
ts1[["structure_type", "rpos", "cid", "tsguess_backend", "ts_spec_id"]].head()
```

| structure_type | rpos | cid | tsguess_backend | ts_spec_id |
| --- | ---: | ---: | --- | --- |
| `TS1` | 2 | 0 | `tsguess2` | `TS1::builtin::methylpyrrole_v2` |
| `TS1` | 3 | 0 | `tsguess2` | `TS1::builtin::methylpyrrole_v2` |
| `TS1` | 3 | 0 | `tsguess2` | `TS1::builtin::methylpyrrole_v2` |
| `TS1` | 5 | 0 | `tsguess2` | `TS1::builtin::methylpyrrole_v2` |

The role sets differ by TS family:

| TS family | Stored roles |
| --- | --- |
| `TS1` | `cat_B`, `cat_N`, `substrate_C`, `transfer_H` |
| `TS2` | `cat_B`, `cat_N`, `substrate_C`, `B_transfer_H`, `N_transfer_H` |
| `TS3` | `cat_B`, `pin_B`, `substrate_C`, `transfer_H` |
| `TS4` | `cat_B`, `pin_B`, `substrate_C`, `transfer_H` |

See [TS Guess DataFrames](../catalyst-screens/ts-guesses.md) for the graph
construction, v2 role meanings, constraints, and backend compatibility notes.

## 4. Inspect Before Calculating

Plot one row from every family:

```python
for ts_type in ["TS1", "TS2", "TS3", "TS4"]:
    ft.plot_row(ts_guesses[ts_type], 0)
```

Then inspect the row-level constraint model:

```python
row = ts_guesses["TS2"].iloc[0]

sorted(row["constraint_roles"])
row["constraint_spec"]
row["ts_core_metrics"]
```

`constraint_roles` maps chemical roles to indices in this row.
`constraint_spec` defines the distances and angles used by constrained
optimizers. `connectivity_bonds` is only the stored/drawn graph and should not
be used to infer constraints.

!!! warning "A well-embedded core is not a validated TS"

    `ts_core_metrics` measures agreement with the built-in starting
    specification. Validate the final optimized structure and imaginary mode
    before using a barrier.

## 5. Build The Production Workflow

For local-to-cluster work, use one workflow object rather than manually
repeating the dataframe cascade:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method="r2scan-3c",
    n_confs=None,
    top_n=10,
    dft=True,
)
```

`wf.targets()` is lightweight: it expands systems, TS families, and reactive
positions, but it does not embed conformers or run calculators.

```python
targets = wf.targets()

len(targets)
targets[0].tag
targets[0].metadata
```

The first target represents one `TS family + system + rpos`. Inspect the active
calculation and scheduler groups before running:

```python
wf.show_stages(execution="dft_staged")[
    ["group", "stage", "engine", "options", "constraint", "lowest"]
]
```

By default, `screen_ts(...)` inserts PRISM pruning after TS-guess generation.
Install `prism-pruner` where the workflow runs, customize it with
`prune_initial={...}`, or set `prune_initial=False` when every initial conformer
must be retained.

## 6. Run One Local Target

Before a production submission, run one target with the same chemistry and
method plan:

```python
df = wf.run(
    targets=[0],
    out_dir="debug/screen_ts",
    execution="dft_staged",
    n_cores=4,
    mem_gb=20,
)

ft.show_steps(df)
```

Successful targets are compacted by default to their final parquet plus
`timing.json`. Use `target_retention="all"` when debugging and you want every
intermediate checkpoint.

For an inexpensive wiring check, make the workflow deliberately small:

```python
smoke = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1"],
    method="r2scan-3c",
    n_confs=1,
    top_n=3,
    dft=False,
    prune_initial=False,
)
```

`dft=False` skips DFT geometry refinement, Hessian, `OptTS`, frequency, and
solvent stages. The screen still applies the configured low-cost stages and the
DFT pre-single-point cutoff.

## 7. Submit The Same Workflow

```python
from frust.cluster import ClusterConfig, Resources

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/screen_ts",
)

result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
    stage_resources={
        "init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "optts": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "solv": Resources(cpus=24, mem_gb=20, timeout_min=3600),
    },
)
```

The collector job writes `merged.parquet` and `collection_report.json` after
the target chains finish. Read the merged dataframe and provenance normally:

```python
import pandas as pd

merged = pd.read_parquet(result.collection_output)
ft.show_steps(merged)
ft.show_timing(merged, detail="workflow")
```

Continue with:

- [Running Catalyst Screens](../catalyst-screens/running.md) for execution
  modes, pruning, retention, and production checks.
- [Workflow Method Plans](../workflows/workflow-methods.md) for calculator
  presets and stage replacement.
- [Transition States](../troubleshooting/transition-states.md) for final TS
  validation.
