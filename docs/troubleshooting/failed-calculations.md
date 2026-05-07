# Failed Calculations

Start with the dataframe. FRUST records most row-level failures in
stage-prefixed columns, so you can often diagnose the problem before opening
raw ORCA or xTB files.

## Triage Checklist

```python
nt_cols = [col for col in df.columns if col.endswith("-NT")]
error_cols = [col for col in df.columns if col.endswith("-error")]

df[[*nt_cols, *error_cols]].head()
```

For a failed stage named `orca_opt`, inspect:

```python
failed = df[df["orca_opt-NT"] == False]
failed[["substrate_name", "rpos", "cid", "orca_opt-error"]].head()
```

!!! tip "Read `*-error` before raw outputs"

    `*-error` usually tells you whether the problem happened in FRUST,
    Tooltoad, ORCA, xTB, g-xTB, UMA, or file handling.

## Common Symptoms

??? question "`ModuleNotFoundError` or missing optional dependency"

    Install the matching extra. For cluster submission:

    ```bash
    pip install -e ".[cluster]"
    ```

    For docs:

    ```bash
    pip install -e ".[docs]"
    ```

??? question "ORCA or xTB executable is not found"

    Check your `.env` and shell environment. FRUST does not install quantum
    chemistry engines.

    ```bash
    echo "$ORCA_EXE"
    echo "$XTB_EXE"
    echo "$GXTB_EXE"
    ```

    See [External Tool Setup](../getting-started/external-tool-setup.md).

??? question "g-xTB rows fail immediately"

    Confirm `GXTB_EXE` points to a g-xTB-capable `xtb` binary:

    ```bash
    "$GXTB_EXE" --help | grep -- --gxtb
    "$GXTB_EXE" --help | grep -- --grad
    ```

    For ORCA-driven g-xTB, also confirm the OET g-xTB wrapper is configured.
    See [g-xTB With FRUST](../external-tools/gxtb.md).

??? question "A few rows fail, but the dataframe returns"

    This is expected. FRUST tries not to abort the full dataframe because one
    conformer or reactive position failed. Filter with `*-NT` before ranking
    or plotting results.

## When To Save Backend Files

If `*-error` is not enough, rerun a small subset with saved output enabled:

```python
step = Stepper(
    step_type="TS1",
    save_output_dir="debug_outputs",
)

df_debug = step.orca(
    df_subset,
    name="debug_orca",
    options={"r2scan-3c": None, "Opt": None},
    save_step=True,
)
```

Saved files let you inspect ORCA output, xTB logs, optimized structures, and
intermediate files directly.

!!! warning "Do not debug from a huge screen first"

    Reproduce the failure with one ligand, one reactive position, and one
    conformer before changing cluster resources or workflow settings globally.
