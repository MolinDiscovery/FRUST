# Workflow Method Plans

The new workflow API lets you describe the chemistry once, choose the
calculator method once, and then run the same object locally or submit it to a
cluster.

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

The mental model is:

```text
Workflow = chemistry, targets, and stage graph
MethodPlan = calculator engines and options for those stages
execution mode = how stages become local calls or cluster jobs
```

That separation is the point. A workflow decides *what chemical targets exist*.
A method plan decides *how each stage is calculated*. The same workflow can be
used for a one-target local smoke test and then submitted to Slurm.

```text
same Workflow + same MethodPlan
    -> local smoke test with wf.run(...)
    -> cluster production with wf.submit(...)
```

## Start With A Screen CSV

Use the normal catalyst-screen format:

```csv
role,smiles,compound_name,rpos
substrate,C1=CC=CO1,furan,
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,TMP,
```

Create a workflow object:

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

The workflow has not run any calculators yet. You can inspect the target list
first:

```python
targets = wf.targets()
targets[:4]
```

For this one-substrate, one-catalyst screen, representative targets look like:

| target index | tag | ts_type | system_name | rpos |
| ---: | --- | --- | --- | ---: |
| 0 | `TS1__furan__TMP__r0` | `TS1` | `furan__TMP` | 0 |
| 1 | `TS1__furan__TMP__r1` | `TS1` | `furan__TMP` | 1 |
| 2 | `TS2__furan__TMP__r0` | `TS2` | `furan__TMP` | 0 |
| 3 | `TS2__furan__TMP__r1` | `TS2` | `furan__TMP` | 1 |

Each target is one independent scientific unit: one TS type, one
substrate-catalyst system, and one reactive position. Conformers are generated
inside that target when the workflow runs.

!!! note "`n_confs=None`"

    `n_confs=None` means FRUST resolves the conformer count from molecule
    complexity. Use an integer such as `n_confs=20` when you want exactly that
    many conformers per target.

## See What The Workflow Prepares

The first stage of a screen TS workflow creates TS guesses. This is the same
structure-generation machinery used by `ft.screen.create_ts_guesses(...)`, but
the workflow keeps it attached to a target and a later calculation graph.

```python
ts_guesses = ft.screen.create_ts_guesses(
    ft.screen.expand(ft.screen.read("screen.csv")),
    ts_types=["TS1", "TS4"],
    n_confs=1,
)

ft.vis.ts_guess_scene(
    ts_guesses["TS4"],
    row_indices=[0, 1],
    show_roles=True,
    show_constraint_distances=True,
    show_constraint_angles=True,
)
```

<iframe
  src="../../assets/workflow-method-ts-guesses.html"
  title="Workflow method tutorial TS guesses"
  width="100%"
  height="430"
  loading="lazy"
  style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 6px;"
></iframe>

The viewer shows the kind of structure the workflow prepares before any xTB or
DFT calculation runs. Role labels, constraint distances, and constraint angles
are row-level data:
they move with the dataframe and are what later constrained stages use.

## Inspect The Method Plan

The built-in `r2scan-3c` method plan uses normal xTB for the inexpensive
filtering stages and ORCA's built-in `r2SCAN-3c` composite method for the DFT
stages.

```python
method = ft.workflows.methods.preset("r2scan-3c")
```

Compactly, the important stage choices are:

| stage | engine | options |
| --- | --- | --- |
| `xtb_preopt` | `xtb` | `gfnff opt` |
| `xtb_sp` | `xtb` | `gfn=2` |
| `xtb_opt` | `xtb` | `gfn=2 opt` |
| `dft_pre_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `dft_pre_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `hess` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `optts` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` |
| `freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |

!!! warning "A method plan does not change the chemistry"

    `MethodPlan` changes calculator engines and options. It does not change
    which TS targets exist, which reactive positions are expanded, or how the
    TS guess roles are assigned.

To see the stages this workflow will actually run, inspect the workflow instead
of reading the full preset map:

```python
wf.show_stages()[["group", "stage", "method_key", "engine", "options"]]
```

For the `screen_ts` workflow above, `dft_staged` uses TS-stage resource groups:

| group | stage | method_key | engine | options |
| --- | --- | --- | --- | --- |
| `init` | `prepare` |  | `prepare` |  |
| `init` | `xtb_preopt` | `xtb_preopt` | `xtb` | `gfnff opt` |
| `init` | `xtb_sp` | `xtb_sp` | `xtb` | `gfn=2` |
| `init` | `xtb_opt` | `xtb_opt` | `xtb` | `gfn=2 opt` |
| `init` | `dft_pre_sp` | `dft_pre_sp` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |
| `init` | `dft_pre_opt` | `dft_pre_opt` | `orca` | `r2SCAN-3c TightSCF SlowConv Opt NoSym` |
| `hess` | `hess` | `hess` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `optts` | `optts` | `optts` | `orca` | `r2SCAN-3c TightSCF SlowConv OptTS NoSym` |
| `freq` | `freq` | `freq` | `orca` | `r2SCAN-3c TightSCF SlowConv Freq NoSym` |
| `solv` | `solv` | `solv` | `orca` | `r2SCAN-3c TightSCF SP NoSym` |

For `ft.workflows.raw_mols(..., dft=True)`, the same method preset would show
`init -> dft_opt -> freq -> solv` groups instead. Raw molecule workflows do not
run the TS-only `hess` or `optts` stages. They do run `freq` after `dft_opt` so
thermochemistry, including Gibbs energies, is available for the optimized
molecule.

### Replace xTB Stages With g-xTB

If you want g-xTB for the xTB-like filtering stages, replace just those stage
specs:

```python
method = (
    ft.workflows.methods.preset("r2scan-3c")
    .replace(
        xtb_sp=ft.workflows.methods.gxtb(job="sp"),
        xtb_opt=ft.workflows.methods.gxtb(job="opt"),
    )
)
```

This is deliberately explicit. g-xTB does not take xTB keywords such as
`{"gfn": 2}`, so the stage says `gxtb(job="sp")` rather than pretending it is a
GFN2 calculation.

You can register a notebook/session preset when you reuse the same method plan:

```python
ft.workflows.methods.register_preset("my-r2scan-gxtb", method)

wf = ft.workflows.screen_ts(
    csv_path="screen.csv",
    method="my-r2scan-gxtb",
)
```

## Run One Target Locally

Before submitting hundreds of cluster jobs, run one target locally:

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

Typical `ft.show_steps(df)` output is compact:

| step | engine | options | columns | n_cores | memory_gb |
| --- | --- | --- | --- | ---: | ---: |
| `initial_conformers` | `embedder` | `requested=None; resolved=50; generated=50` |  |  |  |
| `xtb_preopt` | `xtb` | `gfnff opt` | `xtb_preopt-EE, xtb_preopt-NT, xtb_preopt-oc` | 2 |  |
| `xtb_sp` | `xtb` | `gfn=2` | `xtb_sp-EE, xtb_sp-NT` | 2 |  |
| `xtb_opt` | `xtb` | `gfn=2 opt; lowest=10` | `xtb_opt-EE, xtb_opt-NT, xtb_opt-oc` | 2 |  |
| `DFT-pre-SP` | `orca` | `r2SCAN-3c TightSCF SP NoSym` | `DFT-pre-SP-EE, DFT-pre-SP-NT` | 4 | 20 |

The exact rows depend on how far the local run went. The important part is that
the dataframe records both the calculator stages and compact provenance such as
initial conformer generation and `lowest=...` filtering without widening the
main calculation table.

The staged local output mirrors the cluster shape:

```text
debug/screen_ts/
└── TS1__furan__TMP__r0/
    ├── ts_guess.parquet
    ├── init.parquet
    ├── init.hess.parquet
    ├── init.hess.optts.parquet
    ├── init.hess.optts.freq.parquet
    └── init.hess.optts.freq.solv.parquet
```

!!! tip "Use local runs as smoke tests"

    A one-target local run is the fastest way to catch input CSV problems,
    missing executables, bad catalyst scaffolds, and unexpected method-plan
    options before submitting a production screen.

## Choose An Execution Mode

The workflow controls chemistry. The execution mode controls how the stages are
grouped into jobs.

| execution | What happens | Use when |
| --- | --- | --- |
| `single_job` | all stages for a target run in one job/process | small tests or non-DFT workflows |
| `dft_staged` | cheap initialization stays together; DFT stages become dependent jobs | normal production DFT screens |
| `fully_staged` | every stage is its own dependent job | debugging or unusual resource tuning |

!!! info "`dft_staged` is the production default"

    For DFT workflows, `dft_staged` usually gives the right balance: cheap
    conformer generation and filtering happen together, while long DFT stages
    get separate scheduler jobs and resources.

## Submit The Same Workflow To Slurm

After the one-target smoke test, submit the same workflow object. This submits
all targets in the workflow:

```python
from frust.cluster import ClusterConfig

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/screen_ts",
)

result = wf.submit(
    out_dir="runs/screen_ts",
    cluster=cluster,
    execution="dft_staged",
)

result
```

With this exact call, FRUST uses these defaults:

| Setting | Default behavior |
| --- | --- |
| `targets` omitted | submit every target from `wf.targets()` |
| `stage_resources` omitted | use `Resources(cpus=4, mem_gb=20, timeout_min=720)` for every submitted job group |
| `execution="dft_staged"` | submit one `init` job, then dependent DFT jobs for each target |
| `out_dir="runs/screen_ts"` | write one subdirectory per target under `runs/screen_ts/` |

Representative result:

```python
JobSubmissionResult(
    job_ids=[...],
    tags=["TS1__furan__TMP__r0", "TS1__furan__TMP__r1", "..."],
    save_dirs=["runs/screen_ts/TS1__furan__TMP__r0", "..."],
    mode="screen_ts:dft_staged",
    backend="slurm",
)
```

For production DFT work, override the stage resources that need more time,
memory, or cores:

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

The `stage_resources` keys are workflow stage-group names. Missing keys still
fall back to `Resources(cpus=4, mem_gb=20, timeout_min=720)`. Use
`wf.show_stages(execution="dft_staged")` and read the `group` column before
choosing overrides. For a screen TS workflow in `dft_staged` mode, the common
keys are:

| key | Contains |
| --- | --- |
| `init` | TS guess generation, constrained GFNFF, xTB SP, xTB opt, DFT pre-SP, DFT pre-opt |
| `hess` | ORCA Hessian/frequency input Hessian stage |
| `optts` | ORCA `OptTS` using the previous Hessian |
| `freq` | final frequency check |
| `solv` | final solvent single point |

!!! info "When `execution` is omitted"

    DFT workflows default to `execution="dft_staged"`. Non-DFT workflows default
    to `execution="single_job"`.

## Collect Finished Outputs

When the jobs are done, collect the deepest parquet from each target directory:

```python
merged = wf.collect(
    "runs/screen_ts",
    output="screen_ts_merged.parquet",
    require_normal_termination=True,
)

ft.show_steps(merged)
```

`wf.collect(...)` preserves merged dataframe attrs, so `ft.show_steps(...)`
still works after combining many target outputs.

## The Same Pattern For Molecules

The same method-plan and execution ideas also work for molecular-state
workflows:

```python
wf = ft.workflows.mols(
    csv_path="molecules.csv",
    split="per_rpos",
    select_mols=["ligand", "int2", "mol2"],
    method="r2scan-3c",
    n_confs=None,
    top_n=10,
    dft=True,
)

wf.targets()[:5]
```

`split="per_rpos"` gives one target per generated molecular state and reactive
position. That is useful when DFT is expensive and you want those targets to
land as separate cluster chains.

## Where The Older APIs Fit

`ft.workflows` is the recommended high-level API for new local-to-cluster work.
The older layers are still useful:

| API | Best use |
| --- | --- |
| `ft.workflows` | one object that can run locally, submit to Slurm, and collect results |
| `ft.pipes` | quick local helper functions and legacy-style scripts |
| `ft.Stepper` | explicit dataframe-by-dataframe calculator control |
| `ft.cluster.submit_chain(...)` | legacy transformer/template chain submissions |
| `ft.cluster.submit_screen_chain(...)` | existing screen-chain helper; use `ft.workflows.screen_ts(...)` for new method-plan work |

Use the smallest layer that gives you the control you need. For new production
screens where local testing and cluster submission should share the same code,
start with `ft.workflows`.
