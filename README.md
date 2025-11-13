# FRUST – Frustrated Activation Pipeline

A computational pipeline for automated **frustrated Lewis pair (FLP) activation**

and related **transition-state (TS) workflows**, built for high-throughput,

reproducible calculations on both laptops and HPC clusters.

---

## Status

> ⚠️ **Early Development Phase**  
>
> The API and internal structure are still evolving. Expect breaking changes
>
> between versions while the research project is ongoing.

---

## Overview

**FRUST** (Frustrated Activation) is a research tool that automates much of the

dirty work involved in exploring FLP-type activation mechanisms and related

reactions:

- Generate and transform molecules from simple input (e.g. SMILES, TS templates)
- Enumerate conformers and pre-optimize them
- Build transition-state guesses for specific activation patterns
- Run xTB/DFT workflows (often via ORCA) in a structured, restartable way
- Collect results into tidy tables (Parquet/CSV) ready for downstream analysis

The package is designed around *pipelines* and *steps* that can be chained,

composed, and re-used, making it easier to go from “a list of ligands” to

“TS energies, geometries, and diagnostics” without dozens of ad-hoc scripts.

Although originally developed for **FLP-mediated C–H activation and borylation

systems**, FRUST is intended to be general enough for other small-molecule TS

studies.

---

## Key Features

- 🧩 **Automated molecular transformations**
    - Build FLP activation scenarios from SMILES and TS templates
    - Define bonds to break/form and generate TS-like guess structures
- 🧬 **Conformational sampling**
    - RDKit-based conformer generation
    - UFF (and optionally xTB) pre-optimization
- ⛰️ **Transition state workflows**
    - TS guess generation for different “TS stages” (e.g. TS1, TS3, TS4 etc.)
    - XTB pre-scans and ORCA DFT refinements
    - Support for constrained optimizations and follow-up SP/frequency steps
- ⚙️ **Tight integration with quantum chemistry codes**
    - [xTB] for fast pre-screening and geometry optimization
    - [ORCA] for DFT single-points, optimizations, and vibrational analysis
    - Structured directory layouts and restart-friendly job files
- 🧵 **HPC-friendly execution**
    - Explicit control over cores and memory per step
    - Designed to play well with Slurm job arrays and batch workflows
    - Results collected into `.parquet` files for large-scale analysis
- 📦 **Data-centric outputs**
    - Pandas-friendly tables with energies, geometries, metadata
    - Easy integration with external active-learning pipelines

> Note: Machine-learned accelerators (e.g. UMA) can be integrated via ORCA’s
>
> external interfaces in some workflows, but this is considered advanced usage
>
> and may not be fully stable yet.

---

## Installation

### Prerequisites

- **Python** 3.10+
- **Recommended**: a Conda or mamba environment
- **RDKit** (for conformer generation and molecular manipulation)
- Optional but strongly recommended:
    - [xTB] installed and available on `PATH`
    - [ORCA] installed and accessible (especially on HPC)

The exact installation of xTB/ORCA is environment- and cluster-specific and is

not handled by this package.

### Install from Source

```bash
git clone <repository-url>
cd FRUST
pip install -e .
```

This installs FRUST in **editable** (“development”) mode so code changes are

picked up without reinstalling.

### Dependencies

Most core Python dependencies are installed automatically, including (but not

limited to):

- `rdkit` – molecular manipulation and conformer generation
- `numpy` – numerical computations
- `pandas` – data handling and analysis
- `matplotlib` – basic plotting / visualization
- `tqdm` – progress bars and basic CLI feedback

You may also have additional, more specialized dependencies depending on which

parts of the package you use (e.g. optional ML/backend tooling).

---

## Quick Start

Right now FRUST is primarily driven through **Python** and **Jupyter

notebooks**. The CLI is intentionally minimal while the APIs stabilize.

### Minimal Python Example

Example: generate TS jobs for a set of ligands and run a single TS pipeline

entry using xTB/DFT.

```python
import pandas as pd

from frust.pipes import create_ts_per_rpos, run_ts_per_rpos

# Example input data: SMILES + rpos information
df = pd.read_csv("datasets/ir_borylation.csv")
smiles_list = list(dict.fromkeys(df["smiles"]))

# A TS template structure (e.g. from a reference calculation)
TS_TEMPLATE_XYZ = "structures/ts1_template.xyz"

# 1) Build per-ligand job descriptions
job_inputs = create_ts_per_rpos(smiles_list, TS_TEMPLATE_XYZ)

# Pick one job (e.g. index 0) and run a small test
job = job_inputs[0]
tag = list(job.keys())[0]

results_df = run_ts_per_rpos(
    job,
    n_confs=1,          # small number of conformers for testing
    n_cores=10,         # cores for xTB/ORCA
    mem_gb=30,          # memory budget in GB
    debug=False,
    DFT=True,           # enable DFT refinement
    out_dir="test_run",
    output_parquet=f"test_run/{tag}.parquet",
    save_output_dir=True,
    work_dir="local_test",
)

print(results_df.head())
```

This pattern is typical for FRUST:

1. Build a **list of jobs** from simple input (SMILES + TS template).
2. Run a **pipeline function** that handles individual jobs (xTB → DFT, etc.).
3. Get a nice **DataFrame / Parquet file** you can analyze further.

---

## Project Structure (High-Level)

The exact layout may evolve, but the repo is roughly organized as:

- `frust/`  

  Core Python package with:
    - Pipeline and step definitions (`frust.pipes`)
    - Embedding / transformation utilities (`embedder.py`, `transformers.py`)
    - Execution helpers (`stepper.py`, monitoring, simple I/O utilities)
- `scripts/`  

  Small command-line helpers and submission scripts, e.g.:
    - `submit.py`, `submit2.py`, `submit3.py` – Slurm/HPC helpers
    - Utility scripts to merge `.parquet` outputs, test jobs, etc.
- `playground/`  

  Local scratch space for results, dev experiments, and temporary runs.  

  (Only selected `.py` / `.ipynb` files are tracked; most large output trees

  are intentionally **not** under version control.)
- `dev/`  

  Development notebooks and prototypes (see next section).
- `datasets/`  

  Example input data and reference datasets, e.g.:
    - `ir_borylation.csv` – FLP borylation dataset with SMILES and active sites

No formal `tests/` directory yet – validation currently happens through

targeted development / playground notebooks and small smoke scripts. A

pytest suite will be added once core APIs stabilize.

Because this is active research code, some directories are intentionally

lightly structured and used as a “lab bench” (especially `playground/` and

`dev/`).

---

## Development Notebooks

Several Jupyter notebooks live under `dev/` and act as both documentation and

integration tests for the core pipeline:

- `dev0_pipe_init.ipynb` – pipeline initialization / basic wiring
- `dev1_generic_lig_identi.ipynb` – ligand identification / mapping logic
- `dev2_pipe_fix_dirs.ipynb` – directory and output layout experiments
- `dev3_pipe_test_run.ipynb` – first end-to-end test runs
- `dev4_pipe_test_constrains.ipynb` – constrained optimizations and edge cases

These notebooks are not part of the public API, but they are useful references

for how the system is intended to be used.

---

## Pipeline Workflow

A typical FRUST pipeline for an FLP activation study looks like:

1. **Molecular transformation**

   - Take input structures (SMILES + TS template / reference TS)
   - Apply bond-breaking / bond-forming rules to build TS-like guess structures
   - Map ligand positions (`rpos`) into the TS template
1. **Conformer generation**

   - Enumerate conformers using RDKit
   - Apply filters and pre-selection (energy windows, RMSD pruning, etc.)
1. **Pre-optimization**

   - Use UFF or xTB (e.g. GFN-FF / GFN2-xTB) to get reasonable geometries
   - Optionally use constraints to preserve key activation motifs
1. **Quantum chemical refinement**

   - ORCA DFT optimizations and/or single-point calculations
   - Frequency calculations when needed (e.g. to confirm TS nature)
   - Optional external-method hooks (e.g. UMA or other ML potentials)
1. **Analysis & output**

   - Collect energies, geometries, and metadata into pandas DataFrames
   - Write `.parquet` and/or `.csv` files
   - Provide hooks for plotting, filtering, and ranking candidates

Different reactions (e.g. “TS1”, “TS3”, “TS4” stages of a catalytic cycle)

may correspond to different pipeline variants in `frust.pipes`.

---

## Environment Variables & External Tools

There is currently no global configuration object. Most resource controls

(e.g. `n_cores`, `mem_gb`, `debug`) are passed directly as keyword arguments

to pipeline functions such as `run_ts_per_rpos` and `run_mols`.

Some advanced workflows (UMA-integrated optimizations) require the

environment variable `UMA_TOOLS` to be set to the path of ORCA's external

tools directory. Add this to your `~/.env` or shell profile, e.g.:

```bash
export UMA_TOOLS="/path/to/orca/external-tools"
```

If `UMA_TOOLS` is missing, UMA-related functions will raise a runtime error.

> A lightweight configuration layer may return later; for now explicit
>
> arguments keep runs transparent while the API evolves.

---

## Research Context

FRUST is being developed as part of an ongoing PhD project on **computational

catalyst discovery**, with a focus on:

- Frustrated Lewis pair (FLP) activation mechanisms
- Automated transition-state generation for small-molecule reactions
- High-throughput ligand screening for catalytic systems
- Integration with active-learning and genetic algorithm workflows

The code is research-grade rather than polished “product” software: clarity,

reproducibility, and flexibility are prioritized over backward compatibility.

---

## Contributing

While this is primarily a research codebase, contributions and feedback are

very welcome, especially if you:

- Want to use FRUST in your own FLP / TS studies
- Have ideas for making the pipelines more general or robust
- Spot bugs or edge cases in RDKit/xTB/ORCA handling

Suggested contribution flow:

1. Fork the repository and create a feature branch.
2. Install in development mode: `pip install -e .`
3. Run a small smoke example (e.g. the Quick Start snippet) to ensure your environment works.
4. If adding new logic, mirror usage in a dev or playground notebook; when a formal test suite is introduced, include pytest coverage.
5. Open a pull request with a clear description of the change and its motivation.

---

## License

This project is licensed under the **MIT License**.  

See the [`LICENSE`](LICENSE) file for the full text.

---

## Author

**Jacob Molin Nielsen**  

Email: <jacob.molin@me.com>

---

## Acknowledgments

- Jensen Group and collaborators for scientific discussions and support  
- RDKit developers for the molecular modeling toolkit  
- xTB and ORCA teams for the quantum chemistry engines FRUST relies on  
- The frustrated Lewis pair and computational catalysis communities for

  continuous inspiration

> *Note: This project is under active development. APIs, internal structure, and
>
> supported workflows may change significantly as the research evolves.*