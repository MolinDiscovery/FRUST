# Workflow Method Plans

A workflow defines the chemistry and target graph. A method plan defines which
calculator runs at each calculation stage. The execution mode decides whether
those stages run in one process or as dependent jobs.

```text
Workflow   = chemistry, targets, and ordered stages
MethodPlan = calculator engine and options for each calculation stage
execution  = local/cluster grouping of those stages
```

This tutorial creates one catalyst-screen workflow, inspects it, changes
selected calculator choices, runs a local smoke test, and submits the same
object.

## Create The Workflow

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    csv_path="docs/examples/screen.csv",
    screening="gxtb-default",
    method="r2scan-3c",
    level="full",
    scope="barriers",
    n_confs=None,
    top_n=20,
)
```

The three method blocks are independent:

| Block | Controls | Example values |
| --- | --- | --- |
| `screening` | Inexpensive geometry generation and optimization | `"gxtb-default"`, `"uma-gas"`, `"uma-alpb-chloroform"` |
| `ranking` | Optional single-point selection on screened geometries | `"uma-alpb-chloroform"`, `"wb97xd3-631g"` |
| `method` | Final validation calculator and thermochemistry | `"r2scan-3c"`, `"uma-gas-opt-alpb-chloroform"` |

No conformers or calculators have run yet. Inspect the calculation plan first:

```python
plan = wf.plan()
plan[["branch", "state_id", "system_name", "rpos", "action"]]
```

| branch | state_id | system_name | rpos | action |
| --- | --- | --- | ---: | --- |
| `transition_states` | `TS1` | `n_methyl_pyrrole__tmp_bcat` | 2 | `calculate` |
| `transition_states` | `TS1` | `n_methyl_pyrrole__tmp_bcat` | 3 | `calculate` |
| ... | ... | ... | ... | ... |
| `references` | `ligand` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |
| `references` | `dimer` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |

Each transition-state target is one TS family, substrate-catalyst system, and
reactive position. The expensive `tsguess2` conformer generation runs later,
inside `wf.run(...)` or the submitted `init` job.

## Inspect The Active Method

Ask the workflow which preset entries it actually uses:

```python
wf.show_stages(execution="dft_staged")[[
    "branch",
    "group",
    "stage",
    "engine",
    "options",
    "constraint",
    "lowest",
]]
```

The default `r2scan-3c` screen with `top_n=20` is:

| branch | group | stage | engine | options | constraint | lowest |
| --- | --- | --- | --- | --- | --- | ---: |
| `transition_states` | `init` | `prepare` | `prepare` |  | false |  |
| `transition_states` | `init` | `initial_prune` | `prism_pruner` | `modes=moi,rmsd moi_max_deviation=0.01 rmsd_max_rmsd=0.5` | false |  |
| `transition_states` | `init` | `xtb_preopt` | `xtb` | `gfnff opt` | true |  |
| `transition_states` | `init` | `xtb_sp` | `gxtb` |  | false |  |
| `transition_states` | `init` | `xtb_opt` | `gxtb` | `opt` | true | 20 |
| `transition_states` | `init` | `dft_rank_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |
| `transition_states` | `init` | `dft_preopt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` | true |  |
| `transition_states` | `dft_hessian` | `dft_hessian` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `transition_states` | `dft_ts_opt` | `dft_ts_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` | false |  |
| `transition_states` | `dft_freq` | `dft_freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `transition_states` | `dft_solv_sp` | `dft_solv_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |
| `references` | `init` | `prepare` | `prepare` |  | false |  |
| `references` | `init` | `initial_prune` | `prism_pruner` | `modes=moi,rmsd moi_max_deviation=0.01 rmsd_max_rmsd=0.5` | false |  |
| `references` | `init` | `xtb_preopt` | `xtb` | `gfnff opt` | false |  |
| `references` | `init` | `xtb_sp` | `gxtb` |  | false |  |
| `references` | `init` | `xtb_opt` | `gxtb` | `opt` | false | 20 |
| `references` | `init` | `dft_rank_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |
| `references` | `dft_opt` | `dft_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` | false | 1 |
| `references` | `dft_freq` | `dft_freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | false |  |
| `references` | `dft_solv_sp` | `dft_solv_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | false |  |

!!! note "The table is workflow-specific"

    A method preset can contain keys a particular workflow does not use. For
    example, `raw_mols(..., dft=True)` uses `dft_opt`, `dft_freq`, and
    `dft_solv_sp` but does not use the TS-only `dft_hessian` and `dft_ts_opt`
    stages. Prefer `wf.show_stages()` over reading the full preset mapping.

## Configure Initial Pruning

`catalyst_screen(...)` enables the PRISM stage by default. A dictionary replaces
individual defaults:

```python
wf = ft.workflows.catalyst_screen(
    csv_path="docs/examples/screen.csv",
    screening="gxtb-default",
    method="r2scan-3c",
    level="full",
    scope="barriers",
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
GFN2-xTB initialization, create a new immutable plan and use `screen_ts`, which
applies the method plan's screening stages directly:

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
    n_confs=None,
    top_n=20,
    dft=True,
)

comparison.show_stages(execution="dft_staged")[
    ["stage", "engine", "options"]
].query("stage in ('xtb_sp', 'xtb_opt')")
```

The `xtb_sp` and `xtb_opt` stages now use `xtb` with `gfn=2` instead of `gxtb`.

!!! note "`catalyst_screen` screening presets override method-plan xtb stages"

    `catalyst_screen(..., screening="gxtb-default")` uses the screening preset's
    own g-xTB stages, not the method plan's `xtb_sp`/`xtb_opt`. To customize
    screening calculators in `catalyst_screen`, build a custom `ScreeningPlan`
    with `ft.workflows.methods.ScreeningPlan(...)`.

`MethodPlan` changes calculator settings only. It does not change systems,
reactive positions, `tsguess2` role assignment, or initial pruning.

Register a reusable plan name for the current Python session if needed:

```python
ft.workflows.methods.register_preset("my-r2scan-gfn2-init", method)
```

## Change Screening And Ranking

The `screening` and `ranking` blocks accept preset names or custom plans:

```python
# UMA screening with UMA ranking and DFT validation
hybrid = ft.workflows.catalyst_screen(
    csv_path="docs/examples/screen.csv",
    screening="uma-gas",
    ranking="uma-alpb-chloroform",
    ranking_top_n=3,
    method="r2scan-3c",
    level="full",
    scope="barriers",
)

# Full UMA: screening and validation both use UMA gas
uma_full = ft.workflows.catalyst_screen(
    csv_path="docs/examples/screen.csv",
    screening="uma-gas",
    method="uma-gas",
    level="full",
    scope="barriers",
)
```

Use `ft.workflows.methods.screening_preset(...)` and
`ft.workflows.methods.ranking_preset(...)` to inspect or customize the
individual plans:

```python
screening = ft.workflows.methods.screening_preset("uma-alpb-chloroform")
ranking = ft.workflows.methods.ranking_preset("uma-alpb-chloroform")

screening.to_dict()
ranking.to_dict()
```

See [UMA Method Choices](../catalyst-screens/uma-screening.md) for all
screening/ranking/validation combinations.

## Run A Local Smoke Test

`catalyst_screen` runs all branches together. Start with a cheap level:

```python
smoke = ft.workflows.catalyst_screen(
    csv_path="docs/examples/screen.csv",
    screening="gxtb-default",
    method="r2scan-3c",
    level="low_cost",
    scope="barriers",
    n_confs=1,
    top_n=3,
    prune_initial=False,
)

run = smoke.run(
    out_dir="debug/catalyst_screen",
    execution="dft_staged",
    n_cores=4,
    mem_gb=20,
)

run.summary()
```

`wf.run(...)` returns a `ScreenRun` — the same portable analysis interface you
get after a cluster submission. `run.states()` lists every calculated state;
`run.barriers()` returns the barrier table.

After a successful run, the default `target_retention="compact_success"` keeps
each target's final parquet plus `timing.json`. Use `target_retention="all"`
when you deliberately want every staged checkpoint.

!!! tip "Single-target smoke tests use `screen_ts`"

    `catalyst_screen` runs all branches together. To test one TS target in
    isolation, use `ft.workflows.screen_ts(...)` with `wf.run(targets=[0], ...)`.

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
    log_dir="logs/catalyst_screen",
)

result = wf.submit(
    out_dir="runs/catalyst_screen",
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

Add `array=True, array_parallelism=2` to use staged arrays. Each target waits
for its own preceding group, and every group's array has its own limit of two
running elements. The [array guide](../cluster/arrays.md) shows the complete
target/job mapping and retry workflow.

The finalization job writes a portable run bundle:

```text
runs/catalyst_screen/
├── calculations/
│   ├── transition_states/merged.parquet
│   └── references/merged.parquet
├── analysis/
├── manifest.json
└── run_report.json
```

Open the bundle with `ft.screen.open_run(...)`:

```python
run = ft.screen.open_run(result.run_dir)
run.summary()
run.barriers()[["substrate_name", "ts_type", "rpos", "delta_e_kcal_mol", "quality_status"]]
```

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
