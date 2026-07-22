# Workflow Method Plans

A workflow defines the chemistry and target graph. A method plan defines which
calculator runs at each calculation stage. The execution mode decides whether
those stages run in one process or as dependent jobs.

```text
Workflow   = chemistry, targets, and ordered stages
MethodPlan = calculator engine and options for each calculation stage
execution  = local/cluster grouping of those stages
```

This tutorial creates one screen workflow, inspects it, changes selected
calculator choices, runs one target, and submits the same object.

## Create The Workflow

```python
import frust as ft

method = ft.workflows.methods.preset("r2scan-3c")

wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS4"],
    method=method,
    n_confs=None,
    top_n=20,
    dft=True,
)
```

No conformers or calculators have run yet. Inspect the targets first:

```python
targets = wf.targets()

targets[0].tag
targets[0].metadata
```

Representative target:

| tag | ts_type | system_name | rpos |
| --- | --- | --- | ---: |
| `TS1__n_methyl_pyrrole__tmp_bcat__r2` | `TS1` | `n_methyl_pyrrole__tmp_bcat` | 2 |

Each target is one TS family, substrate-catalyst system, and reactive position.
The expensive `tsguess2` conformer generation runs later, inside `wf.run(...)`
or the submitted `init` job.

## Inspect The Active Method

Ask the workflow which preset entries it actually uses:

```python
wf.show_stages(execution="dft_staged")[[
    "group",
    "stage",
    "method_key",
    "engine",
    "options",
    "constraint",
    "lowest",
]]
```

The default `r2scan-3c` screen is:

| group | stage | engine | options | constraint | lowest |
| --- | --- | --- | --- | --- | ---: |
| `init` | `prepare` | `prepare` |  | false |  |
| `init` | `initial_prune` | `prism_pruner` | `modes=moi,rmsd moi_max_deviation=0.01 rmsd_max_rmsd=0.5` | false |  |
| `init` | `xtb_preopt` | `xtb` | `gfnff opt` | true |  |
| `init` | `xtb_sp` | `gxtb` |  | false |  |
| `init` | `xtb_opt` | `gxtb` | `opt` | true | 10 |
| `init` | `dft_rank_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |
| `init` | `dft_preopt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` | true | 1 |
| `dft_hessian` | `dft_hessian` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `dft_ts_opt` | `dft_ts_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` | false |  |
| `dft_freq` | `dft_freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `dft_solv_sp` | `dft_solv_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |

!!! note "The table is workflow-specific"

    A method preset can contain keys a particular workflow does not use. For
    example, `raw_mols(..., dft=True)` uses `dft_opt`, `dft_freq`, and `dft_solv_sp` but
    does not use the TS-only `dft_hessian` and `dft_ts_opt` stages. Prefer
    `wf.show_stages()` over reading the full preset mapping.

## Configure Initial Pruning

`screen_ts(...)` enables the PRISM stage by default. A dictionary replaces
individual defaults:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS4"],
    method="r2scan-3c",
    prune_initial={
        "modes": ("moi", "rmsd"),
        "moi_max_deviation": 0.01,
        "rmsd_max_rmsd": 0.25,
    },
)
```

Here `0.25` is an explicit override; the workflow default is `0.5`. Install
`prism-pruner` in the environment that runs the `init` stage, or pass
`prune_initial=False` when pruning should be skipped.

## Replace Calculator Stages

Built-in presets use direct g-xTB for `xtb_sp` and `xtb_opt`. To compare with a
GFN2-xTB initialization, create a new immutable plan:

```python
method = (
    ft.workflows.methods.preset("r2scan-3c")
    .replace(
        xtb_sp=ft.workflows.methods.xtb(gfn=2),
        xtb_opt=ft.workflows.methods.xtb(gfn=2, opt=True),
    )
)

comparison = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS4"],
    method=method,
)
```

`MethodPlan` changes calculator settings only. It does not change systems,
reactive positions, `tsguess2` role assignment, or initial pruning.

Register a reusable plan name for the current Python session if needed:

```python
ft.workflows.methods.register_preset("my-r2scan-gfn2-init", method)
```

## Run One Target Locally

```python
df = wf.run(
    targets=[0],
    out_dir="debug/screen_ts",
    execution="dft_staged",
    n_cores=4,
    mem_gb=20,
)

ft.show_steps(df)
ft.show_timing(df)
```

After a successful run, the default `target_retention="compact_success"` keeps:

```text
debug/screen_ts/
└── TS1__n_methyl_pyrrole__tmp_bcat__r2/
    ├── init.hess.optts.freq.solv.parquet
    └── timing.json
```

Use `target_retention="all"` when you deliberately want `ts_guess.parquet`,
`init.parquet`, and every staged checkpoint.

## Choose An Execution Mode

| execution | Local behavior | Cluster behavior |
| --- | --- | --- |
| `single_job` | Run all stages for each target in one call | Submit one job per target |
| `dft_staged` | Run staged checkpoint groups | Submit `init`, then dependent DFT groups |
| `fully_staged` | Run one checkpoint per stage | Submit one dependent job per stage |

DFT workflows default to `dft_staged` for submission. Non-DFT workflows default
to `single_job`.

## Submit The Same Object

```python
from frust.cluster import ClusterConfig, Resources

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/screen_ts",
)

result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
    stage_resources={
        "init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "dft_hessian": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "dft_ts_opt": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "dft_freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "dft_solv_sp": Resources(cpus=24, mem_gb=20, timeout_min=7200),
    },
)
```

The keys in `stage_resources` come from the `group` column returned by
`wf.show_stages(execution="dft_staged")`. Omitted groups use the workflow
resource default.

The automatic collector writes:

```text
runs/screen_ts/
├── <target>/
│   ├── <final-stage>.parquet
│   └── timing.json
├── merged.parquet
└── collection_report.json
```

```python
import pandas as pd

merged = pd.read_parquet(result.collection_output)
ft.show_steps(merged)
```

Use `wf.collect(...)` manually for recovery or a custom merge path.

## Other Workflow Layers

| API | Use |
| --- | --- |
| `ft.workflows` | Recommended local-to-cluster workflow objects |
| `ft.pipes` | Compact supported helpers and existing scripts |
| `ft.Stepper` | Explicit dataframe-by-dataframe calculator control |
| `ft.cluster.submit_screen_chain(...)` | Existing lower-level screen chain submission |

See [Workflow Method Plans](../workflows/workflow-methods.md) for the complete
preset reference and [Cluster Submission](../cluster/submission.md) for
lower-level submission APIs.
