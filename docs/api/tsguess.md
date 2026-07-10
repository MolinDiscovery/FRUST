# TS Guess API

Most users generate transition-state guesses through
`ft.screen.create_ts_guesses(...)` or `ft.workflows.screen_ts(...)`. Both use
`tsguess2` by default.

## Production Backend

`frust.tsguess2` contains the current connected-SMILES builders, v2
specifications, and dataframe generator.

::: frust.tsguess2
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"

See [TS Guess DataFrames](../catalyst-screens/ts-guesses.md) for the row schema,
role meanings, and an end-to-end example.

## Original Assembly Backend

`frust.tsguess` is the original role-assembly backend. It remains available for
compatibility and development of existing assembly-based workflows, but it is
not the default screen path.

::: frust.tsguess
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"
