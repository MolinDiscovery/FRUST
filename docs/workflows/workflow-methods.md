# Workflow Method Plans

`ft.workflows` is the recommended high-level API for new FRUST runs that should
move cleanly from a local test to cluster submission. It keeps four decisions
separate:

| Concept | Owns | Example |
| --- | --- | --- |
| `Workflow` | chemistry, targets, stage graph | `ft.workflows.screen_ts(...)` |
| `ScreeningPlan` | GFN-FF preparation and g-xTB or UMA screening potential | `ft.workflows.methods.screening_preset("uma-alpb-chloroform")` |
| `RankingPlan` | Optional UMA or DFT SP selection on optimized screening geometries | `ft.workflows.methods.ranking_preset("uma-alpb-chloroform")` |
| `MethodPlan` | Final UMA or DFT calculator options and thermochemistry recipe | `ft.workflows.methods.preset("uma-alpb-chloroform")` |
| execution mode | job grouping | `single_job`, `dft_staged`, `fully_staged` |

The same workflow and method can be used in both places:

```text
same Workflow + same ScreeningPlan + same MethodPlan
    -> local smoke test with wf.run(...)
    -> cluster production with wf.submit(...)
```

## Screening, Selection and Validation

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    n_confs=200,
    prune_initial=False,
    screening="gxtb-default",
    top_n=20,
    ranking="uma-alpb-chloroform",
    ranking_top_n=1,
    method="uma-gas-opt-alpb-chloroform",
    level="full",
    scope="barriers",
    dimer_reference="lowest",
)

wf.show_stages(detail="full")
```

This is one end-to-end workflow. The TS sequence is:

```text
200 requested guesses -> constrained GFN-FF Opt -> g-xTB SP
-> keep 20 -> constrained g-xTB Opt -> UMA + ALPB(chloroform) SP
-> keep 1 -> constrained gas UMA Opt -> unconstrained gas Hessian/OptTS/Freq
-> terminal UMA + ALPB(chloroform) SP
```

`top_n` controls the earlier screening cutoff; `ranking_top_n` controls the
finite-energy candidates passed to validation. Failed embeddings or calculations
can leave fewer candidates. `prune_initial=False` disables the optional initial
geometry pruning so every generated guess enters GFN-FF.

Reference targets use the same screening and selection blocks, followed by
unconstrained gas UMA minimum optimization, frequencies and the terminal solvent
SP. Each dimer alternative remains a separate target. All candidates selected by
ranking are validated; `ts_refine_n` instead controls the distinct candidates
selected in native UMA screening runs without a separate ranking block.

| Choice | Meaning | Examples |
|---|---|---|
| `screening` | Generate optimized candidate geometries | `gxtb-default`, `uma-gas` |
| `ranking` | Evaluate those geometries and choose validation inputs | `uma-alpb-chloroform`, `wb97xd3-631g` |
| `method` | Refine and characterize the selected inputs | `uma-gas-opt-alpb-chloroform`, `wb97xd3-631g` |

Change only `method` to select DFT validation of UMA-ranked geometries, or change
`ranking` to a DFT preset to select geometries for UMA validation. Ranking and
validation may use different solvents. The UMA optimization, Hessian and frequency
stages within validation must use one model and environment; the terminal SP may
add ALPB(chloroform).

!!! info "The selected geometry is the validation input"
    Validation consumes the selected screening rows directly. No new guesses are
    generated. The full stage table's `geometry_from` identifies the optimization
    supplying each calculation's coordinates; a ranking SP does not replace that
    geometry source.

Full runs retain the screening result tiers separately from validated results:

```python
run = ft.screen.open_run("results")
electronic_screen = run.barriers(level="uma_ranked")
validated = run.barriers(level="full")
```

The screening tier contains electronic barriers, not calculated Gibbs barriers.
Full gas UMA frequencies plus the terminal solvent SP assemble
`G = E_solv + (G_freq - E_freq)`. TS1/TS3 corrections remain in the separate
corrected-G column. `uma_rank_top_n` remains a compatible alias for
`ranking_top_n`; conflicting values are rejected.

### One Screen TS Workflow

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

For catalyst screens, choose the screening potential separately from the final
method. For example:

```python
wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    ts_types=["TS1"],
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,
)

wf.show_stages()[["branch", "stage", "engine", "solvent"]]
```

Here `uma_sp` selects conformers, `uma_opt` optimizes them with the same
ALPB-corrected UMA potential, and ωB97 supplies the final barrier. Use
`screening="uma-gas"` to omit the correction. The `ScreeningPlan` does not
choose the TS guess profile; `spec_profile` does. See the
[UMA method choices](../catalyst-screens/uma-screening.md) for all four paths,
result tiers, and mode review.

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

| preset name | Final stages | solvent treatment | Use when |
| --- | --- | --- | --- |
| `"uma-gas"` | UMA Hessian/OptTS/NumFreq for TSs; UMA Opt/NumFreq for minima | Gas UMA at every final stage | You want an UMA barrier and matching UMA references in gas phase. |
| `"uma-alpb-chloroform"` | UMA Hessian/OptTS/NumFreq for TSs; UMA Opt/NumFreq for minima | UMA plus GFN2-xTB ALPB(chloroform) minus gas correction | You want an UMA barrier and matching UMA references in the corrected environment. |
| `"r2scan-3c"` | ORCA `r2SCAN-3c` composite method | ORCA `r2SCAN-3c` single point with SMD chloroform | You want the compact composite-method workflow currently used in most new examples. |
| `"wb97xd3-631g"` | ORCA `wB97X-D3/6-31G**` | ORCA `wB97X-D3/6-31+G**` single point with SMD chloroform | You want FRUST's legacy/default workflow behavior. |
| `"r2scan-3c-solv"` | Solvent-inclusive ORCA `r2SCAN-3c` composite method | SMD chloroform in every DFT stage; no final solvent SP | You want the structure, frequencies, and ranking energies evaluated consistently in chloroform. |
| `"wb97xd3-631g-solv"` | Solvent-inclusive ORCA `wB97X-D3/6-31G**` | SMD chloroform in every DFT stage; no final solvent SP | You want the conventional wB97X-D3 workflow evaluated consistently in chloroform. |
| `"r2scan-def2svp"` | ORCA `R2SCAN/def2-SVP` | ORCA `R2SCAN/def2-SVPD` single point with SMD chloroform | You want a conventional R2SCAN/basis-set workflow instead of the `r2SCAN-3c` composite method. |

Without explicit ranking, final UMA presets require matching UMA screening.
With explicit ranking, the selected optimized geometries feed directly into
UMA validation. An UMA-only plan schedules no `dft_*` stage. The existing
gas-phase DFT presets end with `dft_solv_sp`;
the two `*-solv` DFT presets omit it because their DFT calculations already
include SMD chloroform.

The plan also records how molecular free energies are assembled:

| preset family | recorded expression |
| --- | --- |
| gas frequencies + solvent single point | `G = E_solv + (G_freq - E_freq)` |
| solvent-inclusive frequencies | `G = G_freq` |

```python
method = ft.workflows.methods.preset("r2scan-3c")
method.thermochemistry.to_dict()
method.fingerprint()
```

The fingerprint includes calculator settings and the thermochemistry recipe,
but not the human-readable plan name. The end-to-end screen uses it as part of
reference compatibility, preventing results from different scientific methods
from being silently mixed.

| stage id | default engine | role |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | constrained GFNFF preoptimization |
| `xtb_sp` | `gxtb` | direct g-xTB single point ranking |
| `xtb_opt` | `gxtb` | constrained direct g-xTB optimization and conformer filtering |
| `uma_sp` / `uma_opt` | `orca` with UMA | UMA screening single point and constrained optimization |
| `uma_rank_sp` | `orca` with UMA | UMA single point on g-xTB optimized geometries; no UMA optimization |
| `uma_hessian` / `uma_ts_opt` / `uma_freq` | `orca` with UMA | released TS Hessian, OptTS, and numerical frequencies for an UMA final barrier |
| `uma_min_opt` | `orca` with UMA | final UMA minimum optimization before numerical frequencies |
| `dft_rank_sp` | `orca` | DFT single point before DFT optimization |
| `dft_preopt` | `orca` | constrained DFT preoptimization |
| `dft_opt` | `orca` | DFT optimization for molecule workflows |
| `dft_hessian` | `orca` | Hessian/frequency stage for TS optimization |
| `dft_ts_opt` | `orca` | ORCA `OptTS` |
| `dft_freq` | `orca` | final frequency check |
| `dft_solv_sp` | `orca` | final solvent single point for gas-phase presets only |

For g-xTB → UMA SP → ωB97, `top_n` is the broad g-xTB cutoff and
`uma_rank_top_n` is the later UMA cutoff. For a full UMA result,
`ts_refine_n` controls how many distinct constrained UMA geometries proceed
to released OptTS and NumFreq. These cutoffs belong to the workflow, so
changing method presets does not change chemistry target expansion.

`method.stages` is the reusable calculator map. To see which parts of that map
a specific workflow will actually run, inspect the workflow:

```python
wf.show_stages()[["group", "stage", "method_key", "engine", "options", "solvent"]]
```

For the complete planned configuration, request the full view:

```python
full = wf.show_stages(detail="full")
full[[
    "stage",
    "options",
    "solvent",
    "detailed_inp_str",
    "xtra_inp_str",
    "calculator_kwargs",
    "read_files",
    "use_last_hess",
    "save_files",
    "prune_options",
]]
```

Multiline calculator input uses literal `\n` separators so the dataframe can
be printed with `.to_markdown()` without breaking rows. This is the planned
configuration; runtime-derived coordinates, memory directives, executable
paths, and generated constraint blocks are recorded in calculation provenance
and output files after execution.

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

#### Solvent-Inclusive Presets

Use a `*-solv` preset when solvent must affect both the geometries and the
energies used to rank structures. It adds the same SMD chloroform block to all
active DFT stages and deliberately does not schedule `dft_solv_sp`.

```python
import frust as ft

wf = ft.workflows.screen_ts(
    csv_path="docs/examples/screen.csv",
    ts_types=["TS1"],
    method="r2scan-3c-solv",
    dft=True,
)

wf.show_stages(execution="dft_staged")
```

The `solvent` column makes the implicit-solvent model visible for every DFT
stage:

| stage | options | solvent |
| --- | --- | --- |
| `dft_rank_sp` | `r2SCAN-3c TightSCF SP NoSym` | `SMD(chloroform)` |
| `dft_preopt` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` | `SMD(chloroform)` |
| `dft_hessian` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | `SMD(chloroform)` |
| `dft_ts_opt` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` | `SMD(chloroform)` |
| `dft_freq` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` | `SMD(chloroform)` |

| workflow | active solvent-inclusive DFT stages |
| --- | --- |
| `ft.workflows.mols(...)` | `dft_rank_sp -> dft_opt -> dft_freq` |
| `ft.workflows.screen_ts(...)` | `dft_rank_sp -> dft_preopt -> dft_hessian -> dft_ts_opt -> dft_freq` |
| `ft.workflows.int3(...)` | `dft_rank_sp -> dft_preopt -> dft_opt -> dft_freq` |

`r2scan-3c-solv` uses the r2SCAN-3c composite keyword at every DFT stage.
`wb97xd3-631g-solv` uses `wB97X-D3/6-31G**` at every DFT stage; it does not
move the gas-phase preset's `6-31+G**` final-single-point basis into the
optimization or frequency calculations.

The final analysis electronic energy for these workflows is `dft_freq-EE`:

```python
energy_column = ft.result_column(df, purpose="analysis")
# "dft_freq-EE"
```

!!! info "No terminal solvent-SP resource group"

    With a `*-solv` preset, `wf.show_stages()` has no `dft_solv_sp` row. Do
    not include a `"dft_solv_sp"` entry in `stage_resources`; the final DFT
    resource group is `dft_freq`.

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
    ts_types=["TS1", "TS2", "TS4"],
    method="r2scan-3c",
    prune_initial=True,
)
```

Use a dictionary to change the thresholds or modes:

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

To submit worker arrays, add `array=True, array_parallelism=2`. Each staged
group receives its own array; the limit applies separately to each array.
Single-job arrays can additionally use `targets_per_task=5` to run five
targets sequentially in one allocation and share a compatible UMA server.
Staged arrays require one target per element. See
[Slurm Arrays](../cluster/arrays.md) for job counts, resource limits, result
records, and explicit retries.

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
| `ft.cluster.submit_screen_chain(...)` | supported screen-chain helper | you want the previous screen-chain API directly |

The workflow layer does not remove the lower layers. It packages the common
production pattern so the same chemistry and method choices can be reused
locally and on the cluster.
