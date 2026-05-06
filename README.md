# FRUST

FRUST is research software for building and screening frustrated Lewis pair
substrate structures. It provides template-based structure generation,
conformer handling, staged xTB and ORCA workflows, UMA and g-xTB integrations,
and dataframe/parquet outputs for later analysis.

FRUST is active research software. It is useful now, but the API and workflow
defaults should not be treated as fixed.

## Install

FRUST requires Python 3.10 or newer.

```bash
git clone <repository-url>
cd FRUST
pip install -e .
```

Optional extras are available for analytics, cluster submission, notebooks, and
documentation:

```bash
pip install -e ".[analytics,cluster,notebooks,docs]"
```

FRUST does not install external quantum chemistry tools such as xTB, ORCA,
ORCA-External-Tools, UMA, or g-xTB. See the documentation for setup details.

## Quickstart

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

For direct staged control, start with `frust.stepper.Stepper`.

## Documentation

Read the documentation online:

- [FRUST documentation](https://molindiscovery.github.io/FRUST/)

To preview the docs locally while editing them:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Repository Layout

- `frust/` contains the Python package.
- `frust/pipes.py` and `frust/pipelines/` contain workflow entry points.
- `frust/stepper.py` contains the dataframe calculation layer.
- `docs/` contains the MkDocs documentation source.
- `datasets/` and `structures/` contain project input tables and structural
  templates.
- `scripts/` contains supporting helpers outside the packaged API.
