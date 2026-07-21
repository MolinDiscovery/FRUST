# Utility Modules

Common helpers are also available lazily as `ft.<helper>` and
`ft.utils.<helper>`. See the [Top-Level API](top-level.md) for the curated
mapping.

## DataFrames

::: frust.utils.dataframes
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - show_steps
        - show_timing
        - lowest_energy_rows
        - map_substrate_names

## Analysis

::: frust.utils.analytics
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - inspect_ts_vibrations
        - summarize_ts_vibrations

## Molecule Builders

::: frust.utils.mols
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - create_mol_per_rpos

## g-xTB

::: frust.utils.gxtb
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"

## UMA

::: frust.utils.uma
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"

## Pruning

::: frust.utils.pruning
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - prune_conformers
      filters:
        - "!^_"

## IO

::: frust.utils.io
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"
