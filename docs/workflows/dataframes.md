# DataFrames And Results

FRUST uses pandas dataframes as the handoff between workflow stages.

That means most FRUST outputs are ordinary tables. Each row is a structure or
conformer, and each calculation stage adds new columns to the same table. This
is useful because you can inspect results with normal pandas commands instead
of learning a special result format.

The short version is:

```text
rows = structures or conformers
columns = metadata, coordinates, energies, status flags, and saved outputs
```

## A Tiny Example Table

Imagine FRUST has embedded two conformers for the same molecule. Before any
calculation, the dataframe might look conceptually like this:

| substrate_name | structure_type | rpos | cid | atoms | coords_embedded |
| --- | --- | --- | --- | --- | --- |
| anisole | MOL | 2 | 0 | `["C", "H", ...]` | `[(x, y, z), ...]` |
| anisole | MOL | 2 | 1 | `["C", "H", ...]` | `[(x, y, z), ...]` |

After an optimization stage called `xtb_opt`, FRUST adds stage-prefixed
columns:

| substrate_name | cid | xtb_opt-NT | xtb_opt-EE | xtb_opt-oc |
| --- | --- | --- | --- | --- |
| anisole | 0 | `True` | `-42.1` | optimized coordinates |
| anisole | 1 | `True` | `-42.4` | optimized coordinates |

The row is still the same conformer. The new columns tell you what happened
during the calculation stage.

## What One Row Means

Most rows represent one structure or conformer at a particular point in the
workflow.

One ligand can generate several reactive positions. Each reactive position can
generate several conformers. Each conformer can pass through several calculation
stages. FRUST keeps those possibilities visible as rows and columns.

Common identity columns include:

- `structure_id`: stable identity used for grouping conformers;
- `state_id`: canonical chemical state, for example `int2`, `TS3`, or `INT3`;
- `state_kind`: `minimum`, `transition_state`, or `constrained_minimum`;
- `substrate_name`: the ligand or substrate identity;
- `structure_type`: for example `MOL`, `TS1`, `TS2`, or `INT3`;
- `molecule_role`: for example ligand, transition state, or intermediate role;
- `rpos`: reactive position;
- `cid`: conformer id.

These columns are not just labels. FRUST uses them for grouping, especially
when keeping only the lowest-energy conformers.

The canonical columns are the interpretation layer. Historical
`structure_type` and `molecule_role` columns remain available for compatibility
and display, but analysis code should prefer `structure_id`, `state_id`, and
`state_kind`.

## DataFrame Attributes

FRUST stores lightweight provenance in `df.attrs`. The initial dataframe builder
records how the starting rows were made:

```python
df = step.build_initial_df("CCO", name="ethanol")
df.attrs["frust_initial_df"]
```

Output:

```python
{
    "input_kind": "smiles",
    "workflow": None,
    "n_confs": 1,
    "n_cores": 8,
    "optimization": "none",
    "max_iters": 100,
    "select_mols": None,
    "ts_type": None,
    "ts_optimize": None,
    "step_type": None,
    "resolved_step_type": None,
}
```

Calculation stages use a separate `frust_steps` block:

```python
df = step.gxtb(df, name="gxtb_opt", options={"opt": None})
df.attrs["frust_steps"]["gxtb_opt"]
```

Output:

```python
{
    "engine": "gxtb",
    "columns": ["gxtb_opt-EE", "gxtb_opt-NT", "gxtb_opt-oc"],
    "options": {"opt": None},
    "calculator": {
        "name": "gxtb",
        "mode": "direct_gxtb",
        "backend": "tooltoad.gxtb.gxtb_calculate",
        "resources": {"n_cores": 8},
        "executables": {
            "gxtb": {
                "path": "/cluster/apps/g-xtb-2.0.0/bin/xtb",
                "configured": "/cluster/apps/g-xtb-2.0.0/bin/xtb",
                "source": "GXTB_EXE",
                "resolved": True,
            }
        },
    },
}
```

`frust_initial_df` describes input construction. `frust_steps` describes
calculation stages, result columns, and calculator provenance. The nested
`calculator` block is the preferred place to inspect which backend, resources,
and external executables were used. FRUST does not store raw molecule objects
or full input dictionaries in dataframe attributes; row-level identity stays in
columns such as `substrate_name`, `smiles`, `structure_type`, `rpos`, and `cid`.

Workflow outputs also carry a compact `frust_results` contract. Resolve a
column by meaning instead of spelling a workflow-specific prefix:

```python
energy_col = ft.result_column(df, purpose="analysis")
energies = ft.get_result(df, purpose="analysis")
coords_col = ft.result_column(df, key="coords", purpose="optimized")
```

For a completed DFT workflow, `purpose="analysis"` resolves to
`dft_solv_sp-EE` for MOLS, screen TS, and INT3 results. Their optimization
columns stay chemically distinct: `dft_opt-oc` for minima/INT3 and
`dft_ts_opt-oc` for transition states.

Common executable sources:

| Calculator path | Source |
| --- | --- |
| Normal xTB | `XTB_EXE`, or `xtb` resolved from `PATH` |
| Direct g-xTB | `GXTB_EXE` |
| ORCA | `ORCA_EXE` |
| OET wrappers | `OET_TOOLS/bin/...` |

For day-to-day inspection, use `show_steps(...)` to flatten the most useful
parts of `df.attrs["frust_steps"]` into a readable table:

```python
import frust as ft

ft.show_steps(df)
```

Example output from a merged workflow:

| step | engine | mode | options | input_rows | output_rows | dropped_rows |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `initial_prune` | `prism_pruner` | `geometry_pruning` | `modes=moi,rmsd coords_col=coords_embedded moi_max_deviation=0.01 rmsd_max_rmsd=0.25 heavy_atoms_only=True graph_source=connectivity_bonds` | 50 | 18 | 32 |
| `gxtb_opt` | `gxtb` | `direct_gxtb` | `opt` | 18 | 18 |  |
| `DFT-SP-solvent` | `orca` | `direct` | `wB97X-D3 6-31+G** TightSCF SP NoSym` | 18 | 18 |  |

This example used an explicit `rmsd_max_rmsd=0.25` pruning override. The
default `prune_initial=True` workflow configuration uses `1.25`.

For merged workflow outputs, the default summary collapses stored variants such
as `xtb_opt__variant_001` into one logical `xtb_opt` row. The compact columns
`n_variants` and `n_sources` show how many provenance variants and source files
were represented.

The helper also includes compact `executables` and `environment` columns when
you ask for the full view:

```python
ft.show_steps(df, detail="full")
```

Use the default summary view for quick inspection. Multiline ORCA
`xtra_inp_str` values are collapsed to one line so markdown output stays
readable. Use `detail="full"` when you need per-target `__variant_*` rows, the
full call-level `input` block, executable paths, environment paths,
compatibility aliases such as `gxtb_exe`, or the backend callable name.

!!! note
    Provenance is best effort. If an executable is not discoverable, FRUST keeps
    the configured value and records `resolved=False` instead of turning
    metadata collection into a new failure mode.

### Geometry Pruning Metadata

When initial conformer pruning runs, FRUST records it as a normal step in
`df.attrs["frust_steps"]`:

```python
df.attrs["frust_steps"]["initial_prune"]
```

The metadata includes:

| Key | Meaning |
| --- | --- |
| `engine` | `prism_pruner` |
| `options` | PRISM modes, thresholds, coordinate column, and grouping columns |
| `row_counts` | total input, output, and dropped rows |
| `filtering.groups` | per-group input, output, dropped rows, and selected `cid` values |
| `timing` | elapsed time and row counts for `ft.show_timing(...)` |

The grouping columns are inferred from the dataframe by default. This prevents
FRUST from pruning across different TS types, substrates, catalysts, molecule
roles, or reactive positions.

## Timing Metadata

Completed `Stepper` and workflow runs store timing in dataframe attrs, so the
main calculation table stays focused on chemistry columns.

```python
ft.show_timing(df)
```

Example output:

| step | engine | elapsed | input_rows | output_rows | mean_row | max_row | core_hours |
| --- | --- | --- | ---: | ---: | --- | --- | ---: |
| `gxtb_opt` | `gxtb` | `18m 12s` | 120 | 20 | `9.1s` | `43.0s` | 2.43 |
| `DFT_opt` | `orca` | `7h 31m` | 20 | 20 | `22m 33s` | `41m 08s` | 180.4 |

To inspect the slowest stored row diagnostics:

```python
ft.show_timing(df, detail="rows")
```

For workflow target, stage, and stage-group timing:

```python
ft.show_timing(df, detail="workflow")
```

!!! info "Timing lives in attrs"

    Timing is stored in `df.attrs["frust_steps"][step]["timing"]` and
    `df.attrs["frust_workflow_timing"]`. FRUST does not add runtime columns to
    every calculation row by default.

## Coordinates

Most `Stepper` stages need:

- `atoms`: element symbols;
- one coordinate column.

The first coordinate column is often:

```text
coords_embedded
```

After an optimization, the optimized coordinates are stored in a column ending
with:

```text
-oc
```

For example:

```text
xtb_opt-oc
gxtb_preopt-oc
orca_opt-oc
```

When you run the next calculation stage, `Stepper` automatically uses the most
recent coordinate column. This lets a workflow move naturally from embedded
coordinates to low-cost optimization, then ORCA refinement.

## Stage Names And Suffixes

Every calculation stage has a prefix. You usually choose it with `name=`.

```python
df = step.xtb(
    df,
    name="xtb_opt",
    options={"gfn": 2, "opt": None},
)
```

This produces columns such as:

```text
xtb_opt-EE
xtb_opt-NT
xtb_opt-oc
```

The common suffixes are:

| Suffix | Meaning | First thing to do with it |
| --- | --- | --- |
| `-NT` | Normal termination | Filter failed rows |
| `-EE` | Electronic energy | Rank conformers or structures |
| `-GE` | Gibbs energy | Compare thermochemistry when available |
| `-oc` | Optimized coordinates | Use as input to the next stage |
| `-vibs` | Vibrations | Inspect frequency jobs |
| `-error` | Row-level exception text | Debug failed rows |

## Mini-Tutorial: Inspect A Result DataFrame

Start by loading a parquet file:

```python
import pandas as pd

df = pd.read_parquet("runs/example.parquet")
```

Look at the columns:

```python
print(df.columns.tolist())
```

Find calculation status columns:

```python
nt_cols = [col for col in df.columns if col.endswith("-NT")]
print(nt_cols)
```

Keep only rows where the final stage succeeded:

```python
final_nt = nt_cols[-1]
df_ok = df[df[final_nt]]
```

Find energy columns:

```python
energy_cols = [col for col in df.columns if col.endswith("-EE")]
print(energy_cols)
```

Sort by the latest energy:

```python
final_energy = energy_cols[-1]
df_ranked = df_ok.sort_values(final_energy)
```

Inspect the best few rows:

```python
df_ranked[
    ["substrate_name", "structure_type", "rpos", "cid", final_energy]
].head()
```

This is often the first useful analysis after a workflow finishes.

## Mini-Tutorial: Keep The Lowest Conformers

Many `Stepper` methods accept `lowest=...`.

```python
df = step.xtb(
    df,
    name="xtb_opt",
    options={"gfn": 2, "opt": None},
    lowest=5,
)
```

This tells FRUST to group rows by available structure identity columns, then
keep up to five low-energy conformers per group after the stage finishes.

In practical terms, this means:

```text
many conformers -> cheap optimization -> keep the best few -> expensive stage
```

That is the normal screening pattern. Run cheap calculations broadly, then
spend expensive ORCA time only on the most relevant rows.

If you already have a completed results dataframe, use `ft.lowest_energy_rows`
to apply the same grouping rule after the fact:

```python
import pandas as pd
import frust as ft

df_ligs = pd.read_parquet("ligs.parquet")

df_low = ft.lowest_energy_rows(df_ligs)
```

For canonical workflow outputs, FRUST uses the semantic analysis-energy
contract and groups by `structure_id`. Legacy dataframes without a contract
fall back to the latest energy column. To keep more rows, or to rank by a
specific stage:

```python
df_low = ft.lowest_energy_rows(
    df_ligs,
    n=5,
    energy_col="xtb-gfn-opt-EE",
)
```

The helper normalizes legacy `ligand_name` columns to `substrate_name` before
grouping, so older parquet files follow the same identity rules as current
`Stepper(lowest=...)` runs.

To migrate an older completed result once, use:

```python
legacy = pd.read_parquet("old-int3.parquet")
current = ft.upgrade_dataframe(legacy)
energy_col = ft.result_column(current, purpose="analysis")
```

The migration maps unambiguous prefixes such as `DFT-solv` to
`dft_solv_sp`. Ambiguous historical names such as `DFT-SP` require workflow
provenance; strict migration raises instead of silently choosing the wrong
meaning.

## Failed Rows

FRUST tries not to abort an entire dataframe because one row fails. Instead, it
stores failure information in stage-specific columns.

For a stage named `gxtb_opt`, look for:

```text
gxtb_opt-NT
gxtb_opt-error
```

Example:

```python
failed = df[df["gxtb_opt-NT"] == False]
failed[["substrate_name", "cid", "gxtb_opt-error"]].head()
```

Use `-error` first. Only dig into saved calculation files if the error message
does not explain the problem.

## Step Metadata

`Stepper` stores a record of the stages in:

```python
df.attrs["frust_steps"]
```

Example:

```python
for name, meta in df.attrs.get("frust_steps", {}).items():
    print(name, meta)
```

This can tell you which engine was used, what options were passed, and whether
special routes such as UMA or g-xTB were active. For executable provenance,
prefer the nested calculator block:

```python
meta = df.attrs["frust_steps"]["gxtb_opt"]
meta["calculator"]["executables"]["gxtb"]
```

This metadata is useful when you come back to an old parquet file and need to
remember how it was produced.

## Parquet Outputs

FRUST workflows commonly write parquet files because they preserve dataframe
columns efficiently.

A typical analysis loop is:

1. submit or run a workflow;
2. collect parquet outputs;
3. load them with pandas;
4. filter on `-NT`;
5. sort or group by `-EE`;
6. inspect coordinates or saved files for the most interesting rows.

If a submitit run produces many parquet files, use the packaged command:

```bash
merge_parquet --input-dir runs/example --output merged.parquet --recursive
```

Then load the merged result:

```python
df = pd.read_parquet("merged.parquet")
```

## Schema Helpers

For quick scripts, plain pandas is often enough. For reusable analysis code,
FRUST also provides helpers:

```python
from frust.schema import energy_columns, normal_termination_columns, normalize_dataframe

df = normalize_dataframe(df)
energies = energy_columns(df)
nt_cols = normal_termination_columns(df)
```

These helpers are useful when comparing older parquet files with newer results,
because they normalize legacy column names and locate common output columns.
