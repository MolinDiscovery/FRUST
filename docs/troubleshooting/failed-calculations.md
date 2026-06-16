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

## Workflow Failure Reports

For submitted workflows, start from the collection report. The merged parquet
usually contains only successful targets, while skipped targets are summarized
in a separate failure table:

```python
import frust as ft

failures = ft.workflows.inspect_failures(result.collection_report)

failures[
    [
        "target",
        "ts_type",
        "substrate_name",
        "catalyst_name",
        "rpos",
        "cid",
        "failed_stage",
        "error",
        "backend_hint",
    ]
]
```

Example output:

| target | ts_type | substrate_name | catalyst_name | rpos | cid | failed_stage | error | backend_hint |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `TS2__substrate_001__catalyst_001__r2` | `TS2` | `C6_wb97` | `NEt` | 2 | 42 | `OptTS` | `RuntimeError: Orca calculation did not terminate normally` | `ORCA finished by error termination in Startup` |

To answer a specific missing-structure question, filter the failure table by the
same labels you used in the analysis notebook:

```python
missing = failures.query(
    "ts_type == 'TS2' and substrate_name == 'C6_wb97' "
    "and catalyst_name == 'NEt' and rpos == 2"
)

missing[["target", "cid", "failed_stage", "error", "backend_hint"]]
```

The main status values are:

| `problem` | Meaning |
| --- | --- |
| `failed_stage` | A row exists, but at least one `*-NT` column is false or missing. |
| `missing_output` | The collector expected a target parquet that was not written. |
| `read_error` | A parquet or collection report could not be read. |
| `unknown_skipped` | A skipped file exists, but FRUST could not extract a failed row from it. |

!!! note "Use the report before walking directories"

    `failed_stage` points to the first failed calculation prefix, such as
    `OptTS`. `error` comes from the matching `OptTS-error` column when present.
    `backend_hint` is a short line from saved backend output, such as
    `OptTS-orca.out`, when FRUST has that text in the skipped parquet.

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
