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
- machine- or cluster-specific environment setup

The only packaged CLI entry point at the moment is `merge_parquet`.

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
df = step.xtb(df, options={"gfn": 2})
df = step.orca(df, options={"HF": None, "STO-3G": None, "SP": None})
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