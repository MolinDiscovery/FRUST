# Catalyst And Substrate Screens

This tutorial follows one catalyst screen from component CSV to an inspectable
workflow object. The dedicated [Catalyst Screens](../catalyst-screens/overview.md)
section contains the full input, TS-guess, and execution reference.

The path is:

```text
component CSV
    -> normalized components
    -> substrate-catalyst systems
    -> tsguess2 dataframes
    -> inspect one target
    -> composed catalyst_screen workflow
    -> local smoke test
    -> staged cluster run
```

## 1. Read The Example Screen

FRUST includes a compact documentation input at
[`docs/examples/screen.csv`](../examples/screen.csv):

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,,pyrrole
substrate,COC1=CC=CO1,methoxyfuran,"3,5",furan
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

```python
import frust as ft

components = ft.screen.read("docs/examples/screen.csv")
components[["role", "compound_name", "rpos"]]
```

| role | compound_name | rpos |
| --- | --- | --- |
| substrate | `n_methyl_pyrrole` |  |
| substrate | `methoxyfuran` | `3,5` |
| catalyst | `tmp_bcat` |  |

Blank substrate `rpos` means "use symmetry-unique aromatic C-H positions." A
single integer or a comma/semicolon-separated value selects explicit RDKit atom
indices. Draw labels from the exact SMILES before entering them:

```python
ft.DrawUniqueChGrid(components.loc[components["role"].eq("substrate"), "smiles"].tolist())
```

## 2. Expand Components Into Systems

```python
systems = ft.screen.expand(components)
systems[["system_name", "substrate_name", "catalyst_name", "rpos"]]
```

| system_name | substrate_name | catalyst_name | rpos |
| --- | --- | --- | --- |
| `n_methyl_pyrrole__tmp_bcat` | `n_methyl_pyrrole` | `tmp_bcat` |  |
| `methoxyfuran__tmp_bcat` | `methoxyfuran` | `tmp_bcat` | `3,5` |

This explicit table is useful before any geometry work. With multiple catalysts,
`ft.screen.expand(...)` creates every substrate-catalyst pair and prefixes extra
metadata with `substrate_` or `catalyst_`.

## 3. Generate One-Conformer TS Guesses

Use one conformer while checking the wiring:

```python
ts_guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
)

ts_guesses.keys()
```

The default backend is `tsguess2`. It builds a connected TS-like graph, assigns
v2 roles, anchors those roles to the built-in coordinate specification, and
embeds the conformer.

```python
ts1 = ts_guesses["TS1"]
ts1[["structure_type", "rpos", "cid", "tsguess_backend", "ts_spec_id"]].head()
```

| structure_type | rpos | cid | tsguess_backend | ts_spec_id |
| --- | ---: | ---: | --- | --- |
| `TS1` | 2 | 0 | `tsguess2` | `TS1::tsguess2-v2::wb97xd3-631g::gas::r1` |
| `TS1` | 3 | 0 | `tsguess2` | `TS1::tsguess2-v2::wb97xd3-631g::gas::r1` |
| `TS1` | 3 | 0 | `tsguess2` | `TS1::tsguess2-v2::wb97xd3-631g::gas::r1` |
| `TS1` | 5 | 0 | `tsguess2` | `TS1::tsguess2-v2::wb97xd3-631g::gas::r1` |

The role sets differ by TS family:

| TS family | Stored roles |
| --- | --- |
| `TS1` | `cat_B`, `cat_N`, `substrate_C`, `transfer_H` |
| `TS2` | `cat_B`, `cat_N`, `substrate_C`, `B_transfer_H`, `N_transfer_H` |
| `TS3` | `cat_B`, `pin_B`, `substrate_C`, `transfer_H` |
| `TS4` | `cat_B`, `pin_B`, `substrate_C`, `transfer_H` |

See [TS Guess DataFrames](../catalyst-screens/ts-guesses.md) for the graph
construction, v2 role meanings, constraints, and backend compatibility notes.

## 4. Inspect Before Calculating

Plot one row from every family:

```python
for ts_type in ["TS1", "TS2", "TS3", "TS4"]:
    ft.plot_row(ts_guesses[ts_type], 0)
```

Then inspect the row-level constraint model:

```python
row = ts_guesses["TS2"].iloc[0]

sorted(row["constraint_roles"])
row["constraint_spec"]
row["ts_core_metrics"]
```

`constraint_roles` maps chemical roles to indices in this row.
`constraint_spec` defines the distances and angles used by constrained
optimizers. `connectivity_bonds` is only the stored/drawn graph and should not
be used to infer constraints.

!!! warning "A well-embedded core is not a validated TS"

    `ts_core_metrics` measures agreement with the built-in starting
    specification. Validate the final optimized structure and imaginary mode
    before using a barrier.

## 5. Build The Production Workflow

For local-to-cluster work, use `ft.workflows.catalyst_screen(...)`. It composes
screening, optional ranking, and final validation into one workflow object that
also calculates the molecular references needed for barrier equations.

```python
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

`scope="barriers"` calculates TS1--TS4 plus the ligand, dimer, HBpin, and H2
references required by the barrier equations. `scope="full_cycle"` additionally
calculates catalyst, int1, int2, INT3, and borylated product states for a
balanced catalytic-cycle profile.

`wf.plan()` is calculation-free. It shows which structures will be calculated
and which approved references can be reused:

```python
wf.plan()[["branch", "state_id", "system_name", "rpos", "action"]]
```

| branch | state_id | system_name | rpos | action |
| --- | --- | --- | ---: | --- |
| `transition_states` | `TS1` | `n_methyl_pyrrole__tmp_bcat` | 2 | `calculate` |
| `transition_states` | `TS1` | `n_methyl_pyrrole__tmp_bcat` | 3 | `calculate` |
| `transition_states` | `TS2` | `n_methyl_pyrrole__tmp_bcat` | 2 | `calculate` |
| ... | ... | ... | ... | ... |
| `references` | `ligand` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |
| `references` | `dimer` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |
| `references` | `HBpin-mol` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |
| `references` | `HH` | `n_methyl_pyrrole__tmp_bcat` |  | `calculate` |

Inspect the active calculation and scheduler groups before running:

```python
wf.show_stages(execution="dft_staged")[
    ["branch", "group", "stage", "engine", "options", "constraint", "lowest"]
]
```

By default, `catalyst_screen(...)` inserts PRISM pruning after TS-guess
generation. Install `prism-pruner` where the workflow runs, customize it with
`prune_initial={...}`, or set `prune_initial=False` when every initial conformer
must be retained.

!!! tip "TS-only screens use `screen_ts`"

    When you only need transition states without molecular references, use
    `ft.workflows.screen_ts(...)` instead. It accepts `screening` and `method`
    but skips the reference branches.

## 6. Run A Local Smoke Test

Before a production submission, run the workflow locally with the same
chemistry and method plan. `catalyst_screen` runs all branches together, so
start with a small input or a cheap level:

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

`level="low_cost"` reports electronic barriers from the g-xTB screen without
DFT or UMA refinement, Hessian, `OptTS`, frequency, or solvent stages.
`wf.run(...)` returns a `ScreenRun` — the same portable analysis interface you
get after a cluster submission. `run.states()` lists every calculated state
with energies and quality flags; `run.barriers()` returns the barrier table.

Successful targets are compacted by default to their final parquet plus
`timing.json`. Use `target_retention="all"` when debugging and you want every
intermediate checkpoint.

!!! tip "Single-target smoke tests use `screen_ts`"

    `catalyst_screen` runs all branches together. To test one TS target in
    isolation, use `ft.workflows.screen_ts(...)` with `wf.run(targets=[0], ...)`.

## 7. Submit The Same Workflow

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
        "hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "optts": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "solv": Resources(cpus=24, mem_gb=20, timeout_min=3600),
    },
)
```

The finalization job writes a portable run bundle with per-branch
`merged.parquet` files, `run_report.json`, and a snapshot of reused references.
Open the bundle with `ft.screen.open_run(...)`:

```python
run = ft.screen.open_run(result.run_dir)
run.summary()
run.barriers()[["substrate_name", "ts_type", "rpos", "delta_e_kcal_mol", "quality_status"]]
```

Continue with:

- [UMA Method Choices](../catalyst-screens/uma-screening.md) for UMA screening,
  ranking, and validation combinations.
- [End-To-End Calculation And Analysis](../catalyst-screens/end-to-end.md) for
  reference libraries, dimer selection, and portable run bundles.
- [Workflow Method Plans](../workflows/workflow-methods.md) for calculator
  presets and stage replacement.
- [Transition States](../troubleshooting/transition-states.md) for final TS
  validation.
