# Running Catalyst Screens

For production work, use a workflow object. It gives you one place to inspect
targets, choose a method plan, run a local smoke test, submit staged cluster
jobs, and collect outputs.

```python
import frust as ft

method = ft.workflows.methods.preset("r2scan-3c")

wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method=method,
    n_confs=None,
    top_n=10,
    dft=True,
)
```

By default, `screen_ts(...)` prunes geometrically redundant initial conformers
after TS guess generation and before the first xTB stage. Pass
`prune_initial=False` only when you want to keep every generated conformer.

## Inspect Targets

Targets are lightweight. Calling `wf.targets()` expands the screen into
target descriptions, but it does not embed TS conformers or run calculators.

```python
targets = wf.targets()
targets[:2]
```

Representative target metadata:

| tag | metadata |
| --- | --- |
| `TS1__n_methyl_pyrrole__tmp_bcat__r2` | `{"ts_type": "TS1", "system_name": "n_methyl_pyrrole__tmp_bcat", "rpos": 2}` |
| `TS1__n_methyl_pyrrole__tmp_bcat__r3` | `{"ts_type": "TS1", "system_name": "n_methyl_pyrrole__tmp_bcat", "rpos": 3}` |

This makes the target count visible before expensive RDKit or ORCA work starts.

## Inspect Stages

```python
wf.show_stages(execution="dft_staged")[
    ["group", "stage", "engine", "options", "constraint", "lowest"]
]
```

Typical `screen_ts(..., method="r2scan-3c", dft=True)` stages:

| group | stage | engine | options | constraint | lowest |
| --- | --- | --- | --- | --- | ---: |
| `init` | `prepare` | `prepare` |  |  |  |
| `init` | `initial_prune` | `prism_pruner` | `modes=moi,rmsd moi_max_deviation=0.01 rmsd_max_rmsd=1.25` | false |  |
| `init` | `xtb_preopt` | `xtb` | `gfnff opt` | true |  |
| `init` | `xtb_sp` | `gxtb` |  | false |  |
| `init` | `xtb_opt` | `gxtb` | `opt` | true | 10 |
| `init` | `dft_pre_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |
| `init` | `dft_pre_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` | true | 1 |
| `hess` | `hess` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `optts` | `optts` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` | false |  |
| `freq` | `freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `solv` | `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |

`constraint=True` stages render row-level constraints from `constraint_roles`
and `constraint_spec`. This is why a screen-generated dataframe does not need
fixed TS atom indices.

## Initial Conformer Pruning

The default pruning stage compares conformers within each independent TS
family. Rows are grouped by available identity columns such as `system_name`,
`substrate_name`, `catalyst_name`, `structure_type`, `molecule_role`, and
`rpos`, so `TS1` is not pruned against `TS2`, and one substrate/catalyst pair is
not pruned against another.

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method="r2scan-3c",
    prune_initial={
        "modes": ("moi", "rmsd"),
        "moi_max_deviation": 0.01,
        "rmsd_max_rmsd": 0.25,
    },
)
```

To include rotamer-corrected RMSD pruning, opt in explicitly:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    method="r2scan-3c",
    prune_initial={
        "modes": ("moi", "rmsd", "rot_corr_rmsd"),
    },
)
```

!!! note "PRISM is an optional dependency"

    The pruning stage lazy-loads PRISM only when it runs. Install
    `prism-pruner` in the workflow environment, or set `prune_initial=False` to
    skip pruning.

## Run One Local Smoke Test

Run one target locally before submitting the full screen:

```python
df = wf.run(
    targets=[0],
    out_dir="debug/screen_ts",
    execution="dft_staged",
    n_cores=4,
    mem_gb=20,
)

ft.show_steps(df)
```

For early wiring checks, reduce cost:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1"],
    method="r2scan-3c",
    n_confs=1,
    top_n=3,
    dft=False,
    prune_initial=False,
)
```

With `dft=False`, `screen_ts(...)` still runs through `DFT-pre-SP` and then
keeps the lowest DFT single-point row. It skips `DFT-pre-Opt`, `Hess`, `OptTS`,
`Freq`, and `DFT-solv`.

Then inspect the generated and optimized structures:

```python
ft.plot_mols(df, range(0, min(6, len(df))))
df[["custom_name", "rpos", "DFT-pre-SP-EE", "DFT-pre-SP-NT"]].head()
```

## Submit A Staged Cluster Run

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
        "hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "optts": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "solv": Resources(cpus=24, mem_gb=20, timeout_min=3600),
    },
)
```

With `execution="dft_staged"`, cheap generation and pre-screening stay in the
`init` job. Hessian, `OptTS`, final frequency, and solvent single point then
run as dependent jobs with their own resources.

By default, `wf.submit(...)` also submits a collector job. When the target jobs
finish, successful target directories are compacted:

```text
runs/screen_ts/
├── TS1__n_methyl_pyrrole__tmp_bcat__r2/
│   ├── init.hess.optts.freq.solv.parquet
│   └── timing.json
├── merged.parquet
└── collection_report.json
```

`merged.parquet` contains collected normal-termination outputs. The collection
report lists collected, skipped, missing, errored, and compacted targets. Failed
or skipped targets keep intermediate checkpoint parquets for debugging.

```python
import pandas as pd

merged = pd.read_parquet(result.collection_output)
ft.show_steps(merged)
```

!!! note "Where conformers are generated"

    `wf.targets()` stays lightweight. TS conformers are generated during
    `wf.run(...)` or inside the submitted `init` job for each target.

!!! tip "Keep every checkpoint"

    Pass `target_retention="all"` to `wf.run(...)` or `wf.submit(...)` when you
    want to keep `init.parquet`, `init.hess.parquet`, and other intermediate
    checkpoint files for successful targets.

## Working Directly With `Stepper`

Use `ft.screen.create_ts_guesses(...)` directly when you want to inspect or
customize a dataframe-by-dataframe workflow.

```python
components = ft.screen.read("docs/examples/screen.csv")
systems = ft.screen.expand(components)
ts_guesses = ft.screen.create_ts_guesses(systems, ts_types=["TS4"], n_confs=5)

step = ft.Stepper(n_cores=8, save_output_dir=False)

ts4_pruned = step.prune_conformers(
    ts_guesses["TS4"],
    modes=("moi", "rmsd"),
)

ts4_preopt = step.xtb(
    ts4_pruned,
    name="xtb_preopt",
    options={"gfnff": None, "opt": None},
    constraint=True,
)

ts4_lowest = ft.lowest_energy_rows(ts4_preopt)
```

With screen-generated rows, `constraint=True` works row-first:

1. If `constraint_roles` and `constraint_spec` are present, `Stepper` renders
   those role-based constraints.
2. If they are absent, `Stepper` falls back to the older `step_type` and
   `constraint_atoms` behavior.

## Older Convenience APIs

| API | Status | Use when |
| --- | --- | --- |
| `ft.pipes.run_screen_ts_per_rpos(...)` | Supported helper | You want one local function call for the standard screen cascade |
| `ft.cluster.submit_screen_chain(...)` | Supported staged helper | You are maintaining older scripts that call the screen chain directly |
| `ft.workflows.screen_ts(...)` | Recommended | You want method presets, target inspection, local smoke tests, cluster submission, and collection in one object |

The older APIs use `frust.screen` and its selected TS backend; unless explicitly
overridden, that is the same `tsguess2` generation used by the workflow object.
Prefer the workflow object for new screens because it makes local and cluster
behavior easier to compare.

## Production Checklist

Before submitting a large screen:

| Check | Example |
| --- | --- |
| Normalize the input | `components = ft.screen.read("docs/examples/screen.csv", strict=True)` |
| Confirm systems and target count | `systems = ft.screen.expand(components)` and `len(wf.targets())` |
| Inspect `rpos` labels | `ft.DrawUniqueChGrid([...])` |
| Generate one-conformer guesses | `ft.screen.create_ts_guesses(systems.head(1), n_confs=1)` |
| Plot one row per TS family | `ft.plot_row(ts_guesses["TS3"], 0)` |
| Confirm pruning stage | `wf.show_stages(execution="dft_staged")` |
| Run a local smoke target | `wf.run(targets=[0], out_dir="debug/screen_ts", execution="dft_staged")` |
| Inspect workflow provenance | `ft.show_steps(df)` |
| Confirm resources match stage groups | `wf.show_stages(execution="dft_staged")` |
| Inspect final TS quality | Final structure, final frequency, and imaginary mode |
