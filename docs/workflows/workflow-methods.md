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

All three built-ins use the same stage ids. The xTB stages are identical across
presets; the ORCA options differ by preset.

| stage id | default engine | role |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | constrained GFNFF preoptimization |
| `xtb_sp` | `xtb` | xTB single point ranking |
| `xtb_opt` | `xtb` | constrained xTB optimization and conformer filtering |
| `dft_pre_sp` | `orca` | DFT single point before DFT optimization |
| `dft_pre_opt` | `orca` | constrained DFT preoptimization |
| `dft_opt` | `orca` | DFT optimization for molecule workflows |
| `hess` | `orca` | Hessian/frequency stage for TS optimization |
| `optts` | `orca` | ORCA `OptTS` |
| `freq` | `orca` | final frequency check |
| `solv` | `orca` | final solvent single point |

`method.stages` is the reusable calculator map. To see which parts of that map
a specific workflow will actually run, inspect the workflow:

```python
wf.show_stages()[["group", "stage", "method_key", "engine", "options"]]
```

!!! note "Presets are larger than any one workflow"

    A preset contains both molecule-stage keys such as `dft_opt` and TS-stage
    keys such as `hess` and `optts`. The workflow decides which keys are active.
    For example, `raw_mols(..., dft=True)` uses `dft_opt`, `freq`, and `solv`;
    `screen_ts(..., dft=True)` uses `hess`, `optts`, `freq`, and `solv`.

### Exact Built-In Stage Maps

Use these tables when you need to know what a preset means before running a
large cluster job. The `solv` stage also includes this ORCA extra input block:

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
| `xtb_sp` | `xtb` | `gfn=2` |
| `xtb_opt` | `xtb` | `gfn=2 opt` |
| `dft_pre_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `dft_pre_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `hess` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `optts` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` |
| `freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` plus SMD chloroform block |

#### `wb97xd3-631g`

```python
method = ft.workflows.methods.preset("wb97xd3-631g")
```

| stage id | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `xtb` | `gfn=2` |
| `xtb_opt` | `xtb` | `gfn=2 opt` |
| `dft_pre_sp` | `orca` | `wB97X-D3 6-31G** TightSCF SP NoSym` |
| `dft_pre_opt` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Opt NoSym` |
| `hess` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Freq NoSym` |
| `optts` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv OptTS NoSym` |
| `freq` | `orca` | `wB97X-D3 6-31G** TightSCF SlowConv Freq NoSym` |
| `solv` | `orca` | `wB97X-D3 6-31+G** TightSCF SP NoSym` plus SMD chloroform block |

#### `r2scan-def2svp`

```python
method = ft.workflows.methods.preset("r2scan-def2svp")
```

| stage id | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `xtb` | `gfn=2` |
| `xtb_opt` | `xtb` | `gfn=2 opt` |
| `dft_pre_sp` | `orca` | `R2SCAN def2-SVP TightSCF SP NoSym` |
| `dft_pre_opt` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Opt NoSym` |
| `dft_opt` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Opt NoSym` |
| `hess` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Freq NoSym` |
| `optts` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv OptTS NoSym` |
| `freq` | `orca` | `R2SCAN def2-SVP TightSCF SlowConv Freq NoSym` |
| `solv` | `orca` | `R2SCAN def2-SVPD TightSCF SP NoSym` plus SMD chloroform block |

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
| `freq` | `freq` | `freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `solv` | `solv` | `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |

The same preset also contains `hess` and `optts`, but raw molecule workflows do
not run those TS-only stages. The `freq` row is a normal minimum-frequency
calculation after `dft_opt`, so Gibbs-energy columns can be parsed from the
optimized molecule.

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
`init`, `dft_opt`, `freq`, and `solv`; a screen TS DFT workflow uses `init`,
`hess`, `optts`, `freq`, and `solv`.

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
