# FRUST

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/molindiscovery/FRUST)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://github.com/molindiscovery/FRUST/actions/workflows/docs.yml/badge.svg)](https://github.com/molindiscovery/FRUST/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs)](https://molindiscovery.github.io/FRUST/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://docs.astral.sh/ruff/)
[![DOI](https://img.shields.io/badge/preprint-ChemRxiv-purple)](https://doi.org/10.26434/chemrxiv.15003686/v1)

FRUST is research software for building, screening, and analyzing frustrated
Lewis pair substrate structures. It provides template-based structure
generation, conformer handling, staged xTB/g-xTB/ORCA workflows, UMA
integration, and dataframe/parquet outputs for downstream analysis.

FRUST is active research software. It is useful now, but the API and workflow
defaults should not be treated as fixed.

## Install

FRUST requires Python 3.10 or newer.

```bash
git clone https://github.com/MolinDiscovery/FRUST.git
cd FRUST
pip install -e .
```

Optional extras are available for analysis, cluster submission, notebooks, and
documentation:

```bash
pip install -e ".[analytics,cluster,notebooks,docs]"
```

FRUST does not install external quantum chemistry tools such as xTB, ORCA,
ORCA-External-Tools, UMA, or g-xTB. See the documentation for setup details.

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

For workflow-scale work, create one object that can be inspected, run locally,
or submitted to a cluster:

```python
import frust as ft

wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method="r2scan-3c",
    n_confs=None,
    dft=True,
)

wf.targets()[:3]
wf.show_stages(execution="dft_staged")
```

The production screen backend is `tsguess2`. See the
[quickstart](https://molindiscovery.github.io/FRUST/getting-started/quickstart/)
for a local smoke test, external-tool requirements, and cluster submission.

## Documentation

<a href="https://molindiscovery.github.io/FRUST/">
  <img
    src="https://img.shields.io/badge/Read_the_docs-FRUST_documentation-blue?style=for-the-badge&logo=readthedocs"
    alt="FRUST documentation"
  >
</a>

To preview the docs locally while editing them:

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

- `frust/` contains the Python package.
- `frust/workflows/` contains the recommended local-to-cluster workflow objects.
- `frust/pipes.py` and `frust/pipelines/` contain supported lower-level and
  legacy workflow entry points.
- `frust/stepper.py` contains the dataframe calculation layer.
- `tests/` contains unit tests.
- `docs/` contains the MkDocs documentation source.
- `datasets/` and `structures/` contain project input tables and structural
  templates.
- `scripts/` contains supporting helpers outside the packaged API.

## How to Cite

If you use FRUST, please cite the ChemRxiv preprint:

> Nielsen, J. M.; Rasmussen, M. H.; Jensen, J. H.<br>
> *Computational Prediction of Substrate Scope of a Homogeneous Catalyst: The
> Case of Metal-free C-H Borylation by a Frustrated Lewis Pair Catalyst*.<br>
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
including RDKit, pandas, NumPy, SciPy, matplotlib, xTB, ORCA,
ORCA-External-Tools, g-xTB, UMA, Tooltoad, submitit, and MkDocs.

## License

FRUST is distributed under the MIT License. See [LICENSE](LICENSE).
