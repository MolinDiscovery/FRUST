# Workflow Method Plans

`ft.workflows` is the recommended high-level API for new FRUST runs that should
move cleanly from a local test to cluster submission. It keeps three decisions
separate:

| Concept | Owns | Example |
| --- | --- | --- |
| `Workflow` | chemistry, targets, stage graph | `ft.workflows.screen_ts(...)` |
| `MethodPlan` | calculator engines/options | `ft.workflows.methods.preset("r2scan-3c")` |
| execution mode | job grouping | `single_job`, `dft_staged`, `fully_staged` |

The same workflow and method can be used in both places:

```text
same Workflow + same MethodPlan
    -> local smoke test with wf.run(...)
    -> cluster production with wf.submit(...)
```

## One Screen TS Workflow

```python
import frust as ft

method = ft.workflows.methods.preset("r2scan-3c")

wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method=method,
    n_confs=None,
    top_n=20,
    dft=True,
)
```

For `screen_ts(...)`, PRISM pruning is part of the default stage graph. FRUST
prunes geometrically redundant initial conformers after `prepare` and before
the first xTB stage. Pass `prune_initial=False` only when you need to keep
every generated conformer.

Inspect targets before running:

```python
wf.targets()[:3]
```

Targets are lightweight descriptions of scientific work:

| field | Meaning |
| --- | --- |
| `tag` | stable output-directory and scheduler tag |
| `payload` | serializable data needed by the first stage |
| `metadata` | compact target information such as `ts_type`, `system_name`, and `rpos` |

Nothing expensive happens during `wf.targets()`. TS conformers are generated
when the workflow runs.

## Method Plans

Built-in method plans are selected by name:

```python
method = ft.workflows.methods.preset("r2scan-3c")
```

Preset names are forgiving: matching is case-insensitive, and underscores are
treated like hyphens. These calls resolve to the same built-in preset:

```python
ft.workflows.methods.preset("r2scan-3c")
ft.workflows.methods.preset("R2SCAN-3C")
ft.workflows.methods.preset("r2scan_3c")
```

If a workflow receives `method=None`, FRUST currently uses
`"wb97xd3-631g"`. Passing a string is clearer for notebooks and cluster scripts
because the calculation level is visible at the workflow construction site.

### Built-In Presets

| preset name | DFT stages | solvent stage | Use when |
| --- | --- | --- | --- |
| `"r2scan-3c"` | ORCA `r2SCAN-3c` composite method | ORCA `r2SCAN-3c` single point with SMD chloroform | You want the compact composite-method workflow currently used in most new examples. |
| `"wb97xd3-631g"` | ORCA `wB97X-D3/6-31G**` | ORCA `wB97X-D3/6-31+G**` single point with SMD chloroform | You want FRUST's legacy/default workflow behavior. |
| `"r2scan-def2svp"` | ORCA `R2SCAN/def2-SVP` | ORCA `R2SCAN/def2-SVPD` single point with SMD chloroform | You want a conventional R2SCAN/basis-set workflow instead of the `r2SCAN-3c` composite method. |

All three built-ins use the same stage ids. The low-cost initialization stages
are identical across presets; the ORCA options differ by preset.

| stage id | default engine | role |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | constrained GFNFF preoptimization |
| `xtb_sp` | `gxtb` | direct g-xTB single point ranking |
| `xtb_opt` | `gxtb` | constrained direct g-xTB optimization and conformer filtering |
| `dft_rank_sp` | `orca` | DFT single point before DFT optimization |
| `dft_preopt` | `orca` | constrained DFT preoptimization |
| `dft_opt` | `orca` | DFT optimization for molecule workflows |
| `dft_hessian` | `orca` | Hessian/frequency stage for TS optimization |
| `dft_ts_opt` | `orca` | ORCA `OptTS` |
| `dft_freq` | `orca` | final frequency check |
| `dft_solv_sp` | `orca` | final solvent single point |

`method.stages` is the reusable calculator map. To see which parts of that map
a specific workflow will actually run, inspect the workflow:

```python
wf.show_stages()[["group", "stage", "method_key", "engine", "options"]]
```

!!! note "Presets are larger than any one workflow"

    A preset contains both molecule-stage keys such as `dft_opt` and TS-stage
    keys such as `dft_hessian` and `dft_ts_opt`. The workflow decides which keys are active.
    For example, `raw_mols(..., dft=True)` uses `dft_opt`, `dft_freq`, and `dft_solv_sp`;
    `screen_ts(..., dft=True)` uses `dft_hessian`, `dft_ts_opt`, `dft_freq`, and `dft_solv_sp`.

!!! note "Pruning is not a method-plan setting"

    `MethodPlan` changes calculator engines and options. Initial conformer
    pruning is controlled by the workflow through `prune_initial`, because it
    changes which dataframe rows are sent to later calculator stages.

### Exact Built-In Stage Maps

Use these tables when you need to know what a preset means before running a
large cluster job. The `dft_solv_sp` stage also includes this ORCA extra input block:

```text
%CPCM
SMD TRUE
SMDSOLVENT "chloroform"
end
```

#### `r2scan-3c`

```python
method = ft.workflows.methods.preset("r2scan-3c")
```

| stage id | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `gxtb` |  |
| `xtb_opt` | `gxtb` | `opt` |
| `dft_rank_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `dft_preopt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `dft_hessian` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `dft_ts_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` |
| `dft_freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `dft_solv_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` plus SMD chloroform block |

#### `wb97xd3-631g`

```python
method = ft.workflows.methods.preset("wb97xd3-631g")
```

| stage id | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `gxtb` |  |
| `xtb_opt` | `gxtb` | `opt` |
| `dft_rank_sp` | `orca` | `wB97X-D3 6-31G** TightSCF SP NoSym` |
| `dft_preopt` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Opt NoSym` |
| `dft_hessian` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Freq NoSym` |
| `dft_ts_opt` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv OptTS NoSym` |
| `dft_freq` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Freq NoSym` |
| `dft_solv_sp` | `orca` | `wB97X-D3 6-31+G** TightSCF SP NoSym` plus SMD chloroform block |

#### `r2scan-def2svp`

```python
method = ft.workflows.methods.preset("r2scan-def2svp")
```

| stage id | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `gxtb` |  |
| `xtb_opt` | `gxtb` | `opt` |
| `dft_rank_sp` | `orca` | `R2SCAN def2-SVP TightSCF SP NoSym` |
| `dft_preopt` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Opt NoSym` |
| `dft_hessian` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Freq NoSym` |
| `dft_ts_opt` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv OptTS NoSym` |
| `dft_freq` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Freq NoSym` |
| `dft_solv_sp` | `orca` | `R2SCAN def2-SVPD TightSCF SP NoSym` plus SMD chloroform block |

For `ft.workflows.raw_mols(..., method="r2scan-3c", dft=True)`, the active
stages are molecule stages:

| group | stage | method_key | engine | options |
| --- | --- | --- | --- | --- |
| `init` | `prepare` |  | `prepare` |  |
| `init` | `xtb_preopt` | `xtb_preopt` | `xtb` | `gfnff opt` |
| `init` | `xtb_sp` | `xtb_sp` | `gxtb` |  |
| `init` | `xtb_opt` | `xtb_opt` | `gxtb` | `opt` |
| `init` | `dft_rank_sp` | `dft_rank_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `dft_opt` | `dft_opt` | `dft_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `dft_freq` | `dft_freq` | `dft_freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `dft_solv_sp` | `dft_solv_sp` | `dft_solv_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |

The same preset also contains `dft_hessian` and `dft_ts_opt`, but raw molecule workflows do
not run those TS-only stages. The `dft_freq` row is a normal minimum-frequency
calculation after `dft_opt`, so Gibbs-energy columns can be parsed from the
optimized molecule.

### Configure Initial Pruning

The default `screen_ts(...)` pruning configuration runs moment-of-inertia
screening followed by RMSD pruning:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    method="r2scan-3c",
    prune_initial=True,
)
```

Use a dictionary to change the thresholds or modes:

```python
wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    method="r2scan-3c",
    prune_initial={
        "modes": ("moi", "rmsd"),
        "moi_max_deviation": 0.01,
        "rmsd_max_rmsd": 0.25,
    },
)
```

Use `prune_initial=False` for debugging runs where every generated conformer
should be preserved.

!!! info "Install PRISM where the workflow runs"

    PRISM is imported only when pruning runs. A workflow that includes
    `initial_prune` needs `prism-pruner` installed in the local or cluster
    Python environment.

Replace individual stages when you want a different engine or options. For
example, the built-in presets use direct g-xTB for both `xtb_sp` and `xtb_opt`;
replace them if you need the older GFN2-xTB initialization behavior for a
comparison:

```python
method = (
    ft.workflows.methods.preset("r2scan-3c")
    .replace(
        xtb_sp=ft.workflows.methods.xtb(gfn=2),
        xtb_opt=ft.workflows.methods.xtb(gfn=2, opt=True),
    )
)
```

!!! note "Method plans are stage-specific"

    g-xTB stages use `ft.workflows.methods.gxtb(job="sp")` or
    `gxtb(job="opt")`. A `gxtb(job="sp")` stage has no options, so
    `show_stages()` leaves its options cell blank; `gxtb(job="opt")` displays
    `opt`. Do not pass xTB-only settings such as `{"gfn": 2}` to a g-xTB
    stage.

Register a preset for reuse in the current Python session:

```python
ft.workflows.methods.register_preset("my-r2scan-gfn2-init", method)
```

## Execution Modes

```python
df = wf.run(targets=[0], out_dir="debug/screen_ts", execution="dft_staged")
```

```python
result = wf.submit(out_dir="runs/screen_ts", cluster=cluster, execution="dft_staged")
```

That cluster call submits all targets. Because `stage_resources` is omitted,
every submitted job group uses `Resources(cpus=4, mem_gb=20, timeout_min=720)`.
It also submits a final collector job by default. When all target jobs have
finished, that collector writes:

```text
runs/screen_ts/
├── merged.parquet
└── collection_report.json
```

| execution | Local behavior | Cluster behavior |
| --- | --- | --- |
| `single_job` | run all stages for each target in one call | submit one job per target |
| `dft_staged` | run staged checkpoint files, then compact successful targets by default | submit dependent jobs for DFT stages |
| `fully_staged` | run one checkpoint per stage, then compact successful targets by default | submit one dependent job per stage |

For a DFT workflow, omitting `execution` also defaults to `dft_staged`. For a
non-DFT workflow, omitting it defaults to `single_job`.

Successful workflow targets keep only their final parquet and `timing.json` by
default. Pass `target_retention="all"` to `wf.run(...)` or `wf.submit(...)`
when you want to keep every intermediate checkpoint parquet.

Resource overrides are optional and use stage-group names:

```python
from frust.cluster import Resources

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

Use `wf.show_stages(execution="dft_staged")` and read the `group` column to see
the resource keys for a specific workflow. A raw molecule DFT workflow uses
`init`, `dft_opt`, `dft_freq`, and `dft_solv_sp`; a screen TS DFT workflow uses `init`,
`dft_hessian`, `dft_ts_opt`, `dft_freq`, and `dft_solv_sp`. In the default screen TS workflow,
`initial_prune` belongs to the `init` group.

!!! tip "Recommended production mode"

    Use `dft_staged` for production DFT workflows. It keeps cheap filtering
    together, then gives Hessian, `OptTS`, frequency, and solvent stages their
    own resources and scheduler jobs.

## Automatic Collection

```python
result.collection_output
result.collection_report
```

By default, `wf.submit(...)` uses `collect_require_normal_termination=True`.
The merged parquet contains targets whose final normal-termination columns are
all true. `collection_report.json` lists collected, skipped, missing, and
errored target outputs so failed calculations are visible.

For automatic collection, successfully collected targets are compacted by
default. Failed, skipped, missing, or non-normal-termination targets keep their
intermediate checkpoint files for debugging. Manual `wf.collect(...)` defaults
to `target_retention="all"` so recovery on old runs does not unexpectedly
delete files.

After the collector job finishes, load the merged output normally:

```python
import pandas as pd

merged = pd.read_parquet(result.collection_output)
ft.show_steps(merged)
```

Use `wf.collect(...)` manually for recovery, custom output paths, or old runs
submitted before automatic collection. Manual collection still reads the deepest
parquet file from each target directory and merges dataframe attrs.

## Relationship To Existing APIs

| API | Status | Use when |
| --- | --- | --- |
| `ft.workflows` | recommended high-level API | local test and cluster production should share one object |
| `ft.pipes` | supported helper layer | you want a quick local convenience function |
| `ft.Stepper` | supported low-level layer | you want full dataframe-by-dataframe calculator control |
| `ft.cluster.submit_chain(...)` | supported legacy chain layer | you are using transformer/template `.xyz` workflows |
| `ft.cluster.submit_screen_chain(...)` | supported screen-chain helper | you want the previous screen-chain API directly |

The workflow layer does not remove the lower layers. It packages the common
production pattern so the same chemistry and method choices can be reused
locally and on the cluster.
