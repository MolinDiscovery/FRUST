# FRUST Documentation

FRUST is a Python workflow package for building and screening frustrated Lewis
pair substrate structures. It connects substrate inputs to calculation outputs:
template-based structure generation, conformer handling, staged xTB and ORCA
runs, UMA and g-xTB integrations, and parquet-backed result tables.

FRUST is active research software. The workflows are useful now, but APIs and
defaults may still change as the project evolves.

## Start Here

- [Installation](getting-started/installation.md) covers the basic Python
  package setup.
- [External tool setup](getting-started/external-tool-setup.md) covers `.env`,
  xTB, ORCA, OET, UMA prerequisites, and g-xTB paths.
- [Quickstart](getting-started/quickstart.md) shows the shortest practical
  high-level and `Stepper` workflows.
- [Workflow overview](workflows/overview.md) explains when to use the high-level
  pipeline functions, staged cluster pipelines, or direct `Stepper` calls.
- [DataFrames and results](workflows/dataframes.md) documents the main input and
  output conventions used across workflows.

## External Tools

FRUST does not install quantum chemistry engines for you. These integrations
must be configured in the environment where calculations run:

- ORCA and xTB for standard calculations.
- ORCA-External-Tools for UMA and ORCA-driven g-xTB.
- A g-xTB-capable `xtb` executable for direct and ORCA-driven g-xTB.

Start with [External tool setup](getting-started/external-tool-setup.md) when
using UMA or ORCA-driven g-xTB. Then use the [UMA](external-tools/uma.md) and
[g-xTB](external-tools/gxtb.md) pages for method-specific FRUST usage.

## API Reference

The [API reference](api/stepper.md) focuses on the public surface most users
touch directly: `Stepper`, `frust.pipes`, cluster submission helpers, dataframe
schema helpers, and selected external-tool utilities.
