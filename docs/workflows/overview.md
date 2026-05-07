# Workflow Overview

FRUST workflows are easiest to understand as a pipeline from a small input
table to a results dataframe.

!!!info "Naming"

    **Ligand** and **substrate** are used interchangeably for historical reasons.

You usually start with ligand or substrate information, often a CSV or a pandas dataframe with a `smiles` column. FRUST turns those inputs into molecular structures, embeds conformers, runs calculation stages, filters or ranks the results, and writes parquet files that can be inspected later.

For example, the input might be as small as:

```python
import pandas as pd

ligands = pd.DataFrame({"smiles": ["c1ccccc1", "COc1ccccc1"]})
```

The details can get technical, but the mental model is simple:

```mermaid
flowchart TD
    A["Input table<br/>CSV or pandas DataFrame"]
    B["Structure generation<br/>molecules, TS guesses, intermediates"]
    C["Conformer embedding<br/>RDKit plus optional cleanup"]
    D["Calculation stages<br/>xTB, g-xTB, ORCA, UMA"]
    E["Result DataFrame<br/>stage-prefixed columns"]
    F["Parquet outputs<br/>analysis and ranking"]
    G["Chemical inspection<br/>frequencies, modes, conformers, failures"]

    A --> B --> C --> D --> E --> F --> G
```

!!! tip "Recommended reading order"

    Start with this overview, then read
    [TS Guess Generation](ts-guess-generation.md) and
    [Optimization Pipeline](optimization-pipeline.md) before launching a large
    TS screen.

## The Three Layers

FRUST has three workflow layers. Most users move from top to bottom only when
they need more control.

### 1. High-Level Pipelines

Use `frust.pipes` when you want FRUST to do the usual structure generation and
calculation sequence for you.

These functions are the simplest entry points:

- `run_mols(...)`: start from molecule inputs and run the molecule workflow.
- `run_ts_per_lig(...)`: use one transition-state template for each ligand.
- `run_ts_per_rpos(...)`: expand a template over reactive positions and run
  each generated structure.

These are good when you are asking a direct screening question, such as “run
this standard workflow for this ligand table.”

Example:

```python
from frust.pipes import run_mols

df = run_mols(ligands, save_output_dir=False)
```

### 2. Stepper

Use `Stepper` when you want to control the calculation stages yourself.

`Stepper` does not create molecules from scratch. It takes a dataframe that
already contains atoms, coordinates, and metadata, then adds new columns as each
calculation stage finishes.

Typical `Stepper` calls look like:

```python
df = step.xtb(df, name="xtb_preopt", options={"gfnff": None, "opt": None})
df = step.gxtb(df, name="gxtb_opt", options={"opt": None})
df = step.orca(df, name="orca_sp", options={"r2scan-3c": None, "SP": None})
```

Use this layer when you want to inspect intermediate results, change engine
options, keep only the lowest conformers after a specific stage, or mix xTB,
g-xTB, ORCA, and UMA in a custom order.

### 3. Submitit Submission

Use `frust.cluster` when the workflow should be launched through `submitit`.

The Slurm backend is for real cluster runs. The local backend is mainly for
checking that the submission wiring works before sending jobs to Slurm.

There are two submission styles:

- `submit_jobs(...)`: submit independent jobs, usually one pipeline run per
  generated structure or input group.
- `submit_chain(...)`: submit a dependent chain where each stage waits for the
  previous stage to finish.

Example:

```python
from frust.cluster import ClusterConfig, Resources, submit_jobs

submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_mols",
    out_dir="runs/example",
    cluster=ClusterConfig(backend="slurm", partition="kemi1"),
    resources=Resources(cpus=16, mem_gb=50, timeout_min=14400),
)
```

See [Cluster Submission](../cluster/submission.md) for the submitit interface.

## Choosing An Entry Point

If you are new, start here:

```mermaid
flowchart TD
    A["Do you want a standard FRUST workflow?"] -->|Yes| B["Use frust.pipes locally<br/>or submit_jobs on a cluster"]
    A -->|No| C["Use Stepper directly"]
    B --> D["Do you need Slurm?"]
    D -->|No| E["Call the pipeline in Python"]
    D -->|Yes| F["Use frust.cluster"]
    F --> G["Do stages need dependencies<br/>or different resources?"]
    G -->|Yes| H["submit_chain"]
    G -->|No| I["submit_jobs"]
```

In practice, choose the smallest layer that answers your question:

- use `run_mols(...)` for ordinary molecule screening;
- use `run_ts_per_lig(...)` when one TS template should be applied to each
  ligand;
- use `run_ts_per_rpos(...)` when reactive positions should be expanded from a
  template;
- use `submit_jobs(...)` to run those high-level workflows through submitit;
- use `submit_chain(...)` for staged TS or INT workflows where each stage has
  its own resources.

## What Happens During A Run

A typical high-level run does the following:

1. Read or receive the input ligand table.
2. Build molecule or TS-like structures from templates and SMILES.
3. Embed one or more conformers.
4. Create an initial dataframe with atoms, coordinates, conformer ids, and
   structure metadata.
5. Run one or more calculation stages through `Stepper`.
6. Add stage-prefixed output columns to the dataframe.
7. Optionally keep the lowest-energy conformers per structure group.
8. Write parquet output for later analysis.

The high-level functions hide many details, but they still return ordinary
pandas dataframes. That is intentional: you can sort, filter, plot, merge, and
store results using standard pandas tools.

After a run, it is normal to do something like:

```python
df_ok = df[df["xtb_opt-NT"]]
df_ok.sort_values("xtb_opt-EE").head()
```

## External Methods

UMA and g-xTB are both used through the same `Stepper.orca(...)` idea: ORCA owns
the calculation workflow, and an external backend supplies energies and
gradients.

UMA example:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p1",
)
```

Direct g-xTB through Tooltoad:

```python
df = step.gxtb(df, options={"opt": None})
```

ORCA-driven g-xTB, useful when ORCA should own an optimizer such as `OptTS`:

```python
df = step.orca(df, options={"OptTS": None}, gxtb=True)
```

Use direct `Stepper.gxtb(...)` for normal g-xTB calculations. Use
ORCA-driven g-xTB when you specifically want ORCA's optimizer, TS machinery, or
finite-difference `NumFreq` behavior around the external g-xTB gradients.

## What To Inspect First

After a run, start with:

- `*-NT` columns: whether each calculation stage terminated normally;
- `*-EE` columns: electronic energies for ranking and filtering;
- `*-oc` columns: optimized coordinates from optimization stages;
- `*-error` columns: row-level backend errors;
- `df.attrs["frust_steps"]`: metadata about the stages that produced the
  dataframe.

For more detail on column names and dataframe conventions, see
[DataFrames And Results](dataframes.md).

For the chemical checks to run before trusting a result, see
[Inspecting Results](inspecting-results.md).
