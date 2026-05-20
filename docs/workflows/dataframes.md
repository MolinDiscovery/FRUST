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

After an xTB optimization called `xtb_opt`, FRUST adds stage-prefixed columns:

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

- `substrate_name`: the ligand or substrate identity;
- `structure_type`: for example `MOL`, `TS1`, `TS2`, or `INT3`;
- `molecule_role`: for example ligand, transition state, or intermediate role;
- `rpos`: reactive position;
- `cid`: conformer id.

These columns are not just labels. FRUST uses them for grouping, especially
when keeping only the lowest-energy conformers.

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

Common executable sources:

| Calculator path | Source |
| --- | --- |
| Normal xTB | `XTB_EXE`, or `xtb` resolved from `PATH` |
| Direct g-xTB | `GXTB_EXE` |
| ORCA | `ORCA_EXE` |
| OET wrappers | `OET_TOOLS/bin/...` |

!!! note
    Provenance is best effort. If an executable is not discoverable, FRUST keeps
    the configured value and records `resolved=False` instead of turning
    metadata collection into a new failure mode.

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
coordinates to xTB optimization, then ORCA refinement.

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
