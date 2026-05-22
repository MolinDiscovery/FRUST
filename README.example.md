# FRUST

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/molindiscovery/FRUST)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://github.com/molindiscovery/FRUST/actions/workflows/docs.yml/badge.svg)](https://github.com/molindiscovery/FRUST/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs)](https://molindiscovery.github.io/FRUST/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://docs.astral.sh/ruff/)
[![DOI](https://img.shields.io/badge/preprint-ChemRxiv-purple)](https://doi.org/10.26434/chemrxiv.15003686/v1)

FRUST is research software for building, screening, and analyzing frustrated
Lewis pair substrate structures. It combines template-based structure
generation, conformer handling, staged xTB/g-xTB/ORCA calculations, UMA
integration, and dataframe/parquet outputs for downstream analysis.

FRUST is active research software. It is useful now, but the API and workflow
defaults should not be treated as fixed.

## Install

FRUST requires Python 3.10 or newer.

```bash
git clone https://github.com/molindiscovery/FRUST.git
cd FRUST
pip install -e .
```

Optional extras are available for analysis, cluster submission, notebooks, and
documentation:

```bash
pip install -e ".[analytics,cluster,notebooks,docs]"
```

FRUST does not install external quantum chemistry programs. Install and
configure xTB, ORCA, ORCA-External-Tools, g-xTB, or UMA separately when your
workflow needs them.

## Quickstart

For notebooks and quick calculations, use the top-level FRUST toolbox:

```python
import frust as ft

step = ft.Stepper(save_output_dir=False)

df = step.build_initial_df("CCO", name="ethanol")
df = step.gxtb(df, name="gxtb_opt", options={"opt": None})

ft.show_steps(df)
```

If you already have coordinates, pass an XYZ file or XYZ block. FRUST preserves
the supplied geometry:

```python
df = step.build_initial_df("ethanol.xyz")

xyz = """3
water
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.0 0.75 -0.24
"""

df = step.build_initial_df(xyz, name="water")
```

For workflow-scale FRUST structure generation, use the namespaced pipeline API:

```python
import pandas as pd
import frust as ft

substrates = pd.DataFrame(
    {
        "substrate_name": ["anisole"],
        "smiles": ["COc1ccccc1"],
    }
)

df = ft.pipes.run_mols(substrates, n_confs=2, DFT=False)
```

## Documentation

The user documentation is available at:

<https://molindiscovery.github.io/FRUST/>

To preview the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## What FRUST Provides

- Structure generation for frustrated Lewis pair substrate workflows.
- Conformer embedding and dataframe construction from SMILES, XYZ files, RDKit
  molecules, and FRUST structure dictionaries.
- Staged xTB, g-xTB, ORCA, ORCA-driven g-xTB, and ORCA+UMA calculations.
- Lightweight provenance in `df.attrs`, including calculator resources,
  options, executable resolution, and calculation input details.
- Helpers for dataframe inspection, energy filtering, vibration summaries, and
  molecular visualization.
- Cluster submission utilities for larger workflow runs.

## Repository Layout

```text
frust/      Python package
docs/       MkDocs documentation source
tests/      Unit tests
scripts/    Supporting project scripts
datasets/   Example and project input tables
structures/ Structural templates and starting geometries
```

## How to Cite

If you use FRUST, please cite the ChemRxiv preprint:

> Nielsen, J. M.; Rasmussen, M. H.; Jensen, J. H.  
> *Computational Prediction of Substrate Scope of a Homogeneous Catalyst: The
> Case of Metal-free C-H Borylation by a Frustrated Lewis Pair Catalyst*.  
> ChemRxiv, 2026. <https://doi.org/10.26434/chemrxiv.15003686/v1>

BibTeX:

```bibtex
@article{Nielsen_2026_FRUST,
  title = {Computational Prediction of Substrate Scope of a Homogeneous Catalyst: The Case of Metal-free C-H Borylation by a Frustrated Lewis Pair Catalyst},
  author = {Nielsen, Jacob M. and Rasmussen, Maria H. and Jensen, Jan H.},
  year = {2026},
  month = may,
  publisher = {American Chemical Society (ACS)},
  doi = {10.26434/chemrxiv.15003686/v1},
  url = {https://doi.org/10.26434/chemrxiv.15003686/v1}
}
```

## Acknowledgements

FRUST builds on a broad open-source and computational chemistry ecosystem,
including RDKit, pandas, NumPy, SciPy, matplotlib, xTB, ORCA, ORCA-External-
Tools, g-xTB, UMA, Tooltoad, submitit, and MkDocs.

## License

FRUST is distributed under the MIT License. See [LICENSE](LICENSE).

## Notes For Maintainers

Useful badges to add when the corresponding infrastructure exists:

```md
[![CI](https://github.com/molindiscovery/FRUST/actions/workflows/ci.yml/badge.svg)](https://github.com/molindiscovery/FRUST/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/molindiscovery/FRUST/branch/main/graph/badge.svg)](https://codecov.io/gh/molindiscovery/FRUST)
[![PyPI](https://img.shields.io/pypi/v/FRUST)](https://pypi.org/project/FRUST/)
```
