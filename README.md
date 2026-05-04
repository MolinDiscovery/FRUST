# FRUST

FRUST is a research codebase for building and screening frustrated Lewis pair substrates. In practice, it is the layer between substrate inputs and calculation outputs: template-based structure generation, conformer handling, staged xTB and ORCA runs, and dataframe/parquet results that are easy to inspect later.

## Status

This is active research software. It is useful now, but it is still evolving, and the API should not be treated as fixed. A lot of the workflow development still happens in notebooks, and external tools such as xTB and ORCA are expected to exist in the environment already.

## What It Does

- builds TS and related structures from templates and substrate inputs
- embeds, filters, and carries conformers through screening workflows
- runs staged xTB and ORCA calculations through dataframe-based pipelines
- stores energies, geometries, and metadata in parquet-friendly tables

## Installation

FRUST requires Python 3.10+.

```bash
git clone <repository-url>
cd FRUST
pip install -e .
```

If you want the optional extras used in notebooks, analytics, or cluster runs:

```bash
pip install -e ".[analytics,cluster,notebooks]"
```

What this does not install for you:

- xTB
- ORCA
- ORCA-External-Tools, UMA, or g-xTB
- machine- or cluster-specific environment setup

The only packaged CLI entry point at the moment is `merge_parquet`.

## External Tools

UMA runs through ORCA-External-Tools 2.x. Set `OET_TOOLS` to the
orca-external-tools root; the legacy `UMA_TOOLS` environment variable is also
accepted for existing setups. `Stepper.orca(..., uma="omol@uma-s-1p1")` keeps
the compact FRUST shorthand, but FRUST translates it to OET's native `-t omol
-m uma-s-1p1` arguments.

UMA server logs are transient by default. `uma_keep_logs="on_failure"` preserves
logs only when the UMA-backed ORCA stage fails; use `uma_keep_logs=True` and
`uma_log_dir="UMA-logs"` when you want to keep every server log.

g-xTB support is intentionally deferred. OET 2.0.0 includes an `oet_gxtb`
wrapper that requests gradients, but it targets the older standalone `gxtb`
executable and `.gxtb/.eeq/.basisq` parameter-file interface. Current upstream
g-xTB v2 uses `xtb --gxtb --grad`, so FRUST should integrate that separately.

## Where To Start

If you want the high-level workflow API, start with [frust/pipes.py](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/frust/pipes.py). That is where the project-level entry points live, including `run_ts_per_rpos`, `run_ts_per_lig`, and `run_mols`.

If you want more control over individual stages, use [frust/stepper.py](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/frust/stepper.py). `Stepper` is the lower-level dataframe workflow layer for chaining xTB and ORCA calculations one step at a time.

High-level example:

```python
import pandas as pd
from frust.pipes import run_ts_per_lig

ligands = pd.DataFrame({"smiles": ["C1=CC=CN1", "c1ccccc1"]})

df = run_ts_per_lig(
    ligands,
    ts_guess_xyz="structures/ts1.xyz",
    n_confs=2,
    DFT=False,
    save_output_dir=False,
)
```

Lower-level example:

```python
import pandas as pd
from frust.embedder import embed_mols
from frust.stepper import Stepper
from frust.utils.mols import create_mol_per_rpos

ligands = pd.DataFrame({"smiles": ["COc1cccc(OC)c1", "Cc1cccc(N(C)C)c1"]})
mols = create_mol_per_rpos(ligands)
embedded = embed_mols(mols, n_confs=2)

step = Stepper(step_type="MOLS", save_output_dir=False)
df = step.build_initial_df(embedded)
df = step.xtb(df, name="xtb_sp", options={"gfn": 2})
df = step.orca(df, name="hf_sp", options={"HF": None, "STO-3G": None, "SP": None})
```

## Cluster Submission

FRUST also includes a small packaged submission layer for running workflows through `submitit` on either Slurm or a local executor. The public surface is:

```python
from frust.cluster import submit_jobs, submit_chain, ClusterConfig, Resources
```

Independent jobs:

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_mols",
    out_dir="runs/example",
    cluster=ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/example"),
    resources=Resources(cpus=16, mem_gb=50, timeout_min=14400),
)
```

Dependent stage chain:

```python
from frust.cluster import submit_chain, ClusterConfig, Resources

submit_chain(
    csv_path="datasets/example.csv",
    preset="ts_per_rpos",
    ts_xyz="structures/ts1.xyz",
    out_dir="runs/ts1",
    cluster=ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/ts1"),
    stage_resources={"run_init": Resources(cpus=24, mem_gb=20, timeout_min=7200)},
)
```

## Repository Layout

- [frust/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/frust) contains the package code
- [frust/pipes.py](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/frust/pipes.py) and [frust/pipelines/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/frust/pipelines) contain workflow entry points
- [dev/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/dev) contains exploratory notebooks and development runs
- [structures/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/structures) contains TS templates and related structural inputs
- [datasets/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/datasets) contains project input tables
- [scripts/](/Users/jacobmolinnielsen/Developer/FrustActivationProject/FRUST/scripts) contains supporting helpers outside the packaged API

## Using And Contributing

FRUST is meant to be practical for collaborators, not polished for every possible user. The best way to get oriented is to read the pipeline functions, look at the notebooks in `dev/`, and treat the code and examples together as the documentation. If you run into bugs or want to build on it, issues and contributions are welcome.
