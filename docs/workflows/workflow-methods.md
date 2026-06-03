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
    csv_path="screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method=method,
    n_confs=None,
    top_n=10,
    dft=True,
)
```

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

The default screen workflow uses these stage ids:

| stage id | default engine | role |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | constrained GFNFF preoptimization |
| `xtb_sp` | `xtb` | xTB single point ranking |
| `xtb_opt` | `xtb` | constrained xTB optimization and conformer filtering |
| `dft_pre_sp` | `orca` | DFT single point before DFT optimization |
| `dft_pre_opt` | `orca` | constrained DFT preoptimization |
| `hess` | `orca` | Hessian/frequency stage for TS optimization |
| `optts` | `orca` | ORCA `OptTS` |
| `freq` | `orca` | final frequency check |
| `solv` | `orca` | final solvent single point |

`method.stages` is the reusable calculator map. To see which parts of that map
a specific workflow will actually run, inspect the workflow:

```python
wf.show_stages()[["group", "stage", "method_key", "engine", "options"]]
```

For `ft.workflows.raw_mols(..., method="r2scan-3c", dft=True)`, the active
stages are molecule stages:

| group | stage | method_key | engine | options |
| --- | --- | --- | --- | --- |
| `init` | `prepare` |  | `prepare` |  |
| `init` | `xtb_preopt` | `xtb_preopt` | `xtb` | `gfnff opt` |
| `init` | `xtb_sp` | `xtb_sp` | `xtb` | `gfn=2` |
| `init` | `xtb_opt` | `xtb_opt` | `xtb` | `gfn=2 opt` |
| `init` | `dft_pre_sp` | `dft_pre_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `dft_opt` | `dft_opt` | `dft_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `solv` | `solv` | `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |

The same preset also contains `hess`, `optts`, and `freq`, but raw molecule
workflows do not run those TS-only stages.

Replace individual stages when you want a different engine or options:

```python
method = (
    ft.workflows.methods.preset("r2scan-3c")
    .replace(
        xtb_sp=ft.workflows.methods.gxtb(job="sp"),
        xtb_opt=ft.workflows.methods.gxtb(job="opt"),
    )
)
```

!!! note "Method plans are stage-specific"

    g-xTB stages use `ft.workflows.methods.gxtb(job="sp")` or
    `gxtb(job="opt")`. Do not pass xTB-only settings such as `{"gfn": 2}` to a
    g-xTB stage.

Register a preset for reuse in the current Python session:

```python
ft.workflows.methods.register_preset("my-r2scan-gxtb", method)
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

| execution | Local behavior | Cluster behavior |
| --- | --- | --- |
| `single_job` | run all stages for each target in one call | submit one job per target |
| `dft_staged` | write staged parquet files at DFT boundaries | submit dependent jobs for DFT stages |
| `fully_staged` | write one parquet per stage | submit one dependent job per stage |

For a DFT workflow, omitting `execution` also defaults to `dft_staged`. For a
non-DFT workflow, omitting it defaults to `single_job`.

Resource overrides are optional and use stage-group names:

```python
from frust.cluster import Resources

result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
    stage_resources={
        "init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "optts": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "solv": Resources(cpus=24, mem_gb=20, timeout_min=7200),
    },
)
```

Use `wf.show_stages(execution="dft_staged")` and read the `group` column to see
the resource keys for a specific workflow. A raw molecule DFT workflow uses
`init`, `dft_opt`, and `solv`; a screen TS DFT workflow uses `init`, `hess`,
`optts`, `freq`, and `solv`.

!!! tip "Recommended production mode"

    Use `dft_staged` for production DFT workflows. It keeps cheap filtering
    together, then gives Hessian, `OptTS`, frequency, and solvent stages their
    own resources and scheduler jobs.

## Collecting Results

```python
merged = wf.collect(
    "runs/screen_ts",
    output="screen_ts_merged.parquet",
    require_normal_termination=True,
)

ft.show_steps(merged)
```

`wf.collect(...)` reads the deepest parquet file from each target directory and
merges dataframe attrs so provenance inspection still works after collection.

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
