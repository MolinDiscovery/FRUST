# Quickstart

Use workflow objects for new production work. A workflow keeps the chemistry,
calculator plan, local smoke test, cluster submission, and result collection in
one object.

```python
import frust as ft
```

The top-level namespace exposes common notebook helpers such as `ft.Stepper`,
`ft.show_steps`, `ft.plot_mols`, and `ft.write_xyz`. Larger domains remain
grouped under `ft.workflows`, `ft.screen`, `ft.cluster`, `ft.vis`, and
`ft.utils`.

## Inspect A Catalyst Screen

Start with the included example component table:

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,,pyrrole
substrate,COC1=CC=CO1,methoxyfuran,"3,5",furan
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

Create a workflow object without running any calculations:

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

Inspect the scientific targets first:

```python
targets = wf.targets()

len(targets)
targets[0].tag
targets[0].metadata
```

`wf.targets()` expands systems, TS families, and reactive positions. It does
not embed conformers or run calculators.

Inspect the active stage and scheduler groups:

```python
wf.show_stages(execution="dft_staged")[
    ["group", "stage", "engine", "options", "constraint", "lowest"]
]
```

The default TS backend is `tsguess2`. It builds connected TS-like graphs and
stores v2 role-based constraints in every generated row. See
[TS Guess DataFrames](../catalyst-screens/ts-guesses.md) before extending the
chemistry or interpreting role names.

!!! info "Default initial pruning"

    `screen_ts(...)` prunes redundant initial conformers with PRISM before the
    first xTB stage. Install `prism-pruner` in the environment that runs the
    workflow, customize the stage with `prune_initial={...}`, or pass
    `prune_initial=False` when every conformer must be retained.

## Run One Smoke Target

Configure the required external programs before starting calculations. The
[External Tool Setup](external-tool-setup.md) page covers xTB, g-xTB, ORCA, and
ORCA-External-Tools.

Run one target locally before submitting the full screen:

```python
df = wf.run(
    targets=[0],
    out_dir="debug/screen_ts",
    execution="dft_staged",
    n_cores=4,
    mem_gb=20,
)

ft.show_steps(df)
ft.show_timing(df)
```

Successful targets keep only their final parquet and `timing.json` by default.
Pass `target_retention="all"` when you want every intermediate checkpoint for
debugging.

## Submit The Same Workflow

Once the local target behaves as expected, submit the same object:

```python
cluster = ft.ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/screen_ts",
)

result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
)
```

By default, a final collector writes `merged.parquet` and
`collection_report.json`. Use `wf.show_stages(execution="dft_staged")` before
adding stage-specific resource overrides.

## Build A Plain Molecule DataFrame

Use `Stepper` when you want direct dataframe-by-dataframe calculator control:

```python
step = ft.Stepper(save_output_dir=False)

df = step.build_initial_df("CCO", name="ethanol", n_confs=1)
df[["substrate_name", "structure_type", "molecule_role", "cid", "smiles"]]
```

| substrate_name | structure_type | molecule_role | cid | smiles |
| --- | --- | --- | ---: | --- |
| ethanol | `MOL` | `structure` | 0 | `CCO` |

Add calculation stages explicitly when the matching backend is configured:

```python
df = step.gxtb(df, name="gxtb_opt", options={"opt": None})
ft.show_steps(df)
```

After optimization, FRUST appends columns such as `gxtb_opt-NT`,
`gxtb_opt-EE`, and `gxtb_opt-oc` and records calculator provenance in
`df.attrs`.

## Choose The Right Layer

| Need | Public entry point |
| --- | --- |
| One object for local testing, cluster production, and collection | `ft.workflows` |
| Normalize and inspect catalyst-screen systems or generated guesses | `ft.screen` |
| Direct control over dataframe calculation stages | `ft.Stepper` |
| Compact supported helper functions for existing scripts | `ft.pipes` |
| Lower-level cluster submission and legacy stage chains | `ft.cluster` |

For new local-to-cluster workflows, start with `ft.workflows`. The lower layers
remain useful when you deliberately need their additional control.

Continue with [Workflow Overview](../workflows/overview.md),
[Workflow Method Plans](../workflows/workflow-methods.md), or
[Running Catalyst Screens](../catalyst-screens/running.md).
