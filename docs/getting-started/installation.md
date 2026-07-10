# Installation

This page gets FRUST installed as a Python package. If you want to run ORCA,
xTB, UMA, or g-xTB calculations, do this page first and then continue to
[External Tool Setup](external-tool-setup.md).

## 1. Create A Python Environment

Use Python 3.10 or newer. A conda environment is a convenient default:

```bash
conda create -n frust python=3.12
conda activate frust
```

If you already have a working computational chemistry environment, activate
that instead.

## 2. Install FRUST

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/MolinDiscovery/FRUST.git
cd FRUST
python -m pip install -e .
```

This installs the `frust` Python package and the `merge_parquet` command.

## 3. Add Optional Extras

Install only the extras you need:

```bash
python -m pip install -e ".[cluster]"
python -m pip install -e ".[docs]"
```

Or install the common extras together:

```bash
python -m pip install -e ".[analytics,cluster,notebooks,docs]"
```

The extras add Python packages for optional workflows. They do not install
external chemistry programs.

!!! info "PRISM pruning for screen TS workflows"

    `ft.workflows.screen_ts(...)` prunes the initial conformer ensemble by
    default. Install PRISM in the environment that runs those workflows:

    ```bash
    python -m pip install prism-pruner
    ```

    If PRISM is not available, disable that stage explicitly with
    `prune_initial=False`.

## 4. Check The Python Install

```bash
python - <<'PY'
import frust
print("FRUST imports successfully")
PY
```

If that works, the Python package is installed.

## 5. Decide Which Tools You Need

For basic package imports, dataframe utilities, docs, and code exploration, you
can stop here.

For actual calculations, install and configure the relevant external tools:

- PRISM for default initial conformer pruning in `ft.workflows.screen_ts(...)`.
- xTB for xTB stages.
- ORCA and Open MPI for ORCA stages.
- ORCA-External-Tools for UMA and ORCA-driven g-xTB.
- g-xTB for default workflow `xtb_sp` and `xtb_opt` stages, direct
  `Stepper.gxtb(...)`, and ORCA-driven g-xTB.

The next page walks through those paths and smoke tests:
[External Tool Setup](external-tool-setup.md).
