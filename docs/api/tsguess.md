# TS Guess API

`frust.tsguess` contains the public transition-state guess objects used by the
screen workflow. Most users call it through `ft.screen.create_ts_guesses(...)`;
the direct API is useful when extending or inspecting built-in TS specs.

::: frust.tsguess
    options:
      show_root_heading: true
      show_root_full_path: false
      filters:
        - "!^_"
