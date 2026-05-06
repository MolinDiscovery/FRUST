# Quickstart

FRUST has two main entry styles:

- high-level functions in `frust.pipes` for common workflow runs;
- `Stepper` for explicit dataframe-by-dataframe calculation chains.

## High-Level Workflow

Use high-level pipeline functions when you want FRUST to generate structures,
embed conformers, run the standard staged calculations, and return one results
dataframe.

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

## Stepper Workflow

Use `Stepper` when you want direct control over each xTB, g-xTB, or ORCA stage.
The input is a FRUST dataframe with atom and coordinate columns.

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

Use `frust.cluster` when you want FRUST to submit workflows through `submitit`
instead of running them directly in the current Python process.

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

See [cluster submission](../cluster/submission.md) for independent jobs,
dependent chains, local testing, presets, and common errors.
