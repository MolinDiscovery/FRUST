# End-To-End Calculation And Analysis

Use one composed workflow when a TS screen should arrive together with the
references needed to interpret it:

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    screening="gxtb-default",
    level="full",
    method="r2scan-3c",
    ranking_solvation="method",
    scope="barriers",
)

wf.plan()[["branch", "state_id", "system_name", "rpos"]]
```

| branch | state_id | system_name | rpos |
| --- | --- | --- | ---: |
| transition_states | TS1 | pyrrole__NMe | 2 |
| transition_states | TS2 | pyrrole__NMe | 2 |
| references | ligand | pyrrole__NMe |  |
| references | dimer | pyrrole__NMe |  |
| references | dimer_bh_bridged | pyrrole__NMe |  |
| references | dimer_eight_membered | pyrrole__NMe |  |
| references | HBpin-mol | pyrrole__NMe |  |
| references | HH | pyrrole__NMe |  |

`plan()` shows the structures and the reasons they are required before any
embedding or calculation. Multiple sites can share the same molecular target.

## Select The Dimer Reference

The default `dimer_reference="lowest"` calculates all three aminoborane dimer
topologies and selects one reference per catalyst:

| state id | connectivity |
| --- | --- |
| `dimer` | reciprocal B-H-B bridges (the existing FRUST dimer) |
| `dimer_bh_bridged` | one N-B contact and one B-H-B bridge |
| `dimer_eight_membered` | two reciprocal N-B contacts |

Full runs rank qualified conformers by Gibbs energy. `low_cost` and
`dft_ranked` runs rank them by electronic energy. The selected conformer's
electronic and Gibbs energies stay paired and are reused for every substrate
and reactive position belonging to that catalyst.

```python
run = ft.screen.open_run("results")
run.dimer_references()[
    [
        "catalyst_name",
        "state_id",
        "relative_energy_kcal_mol",
        "selected",
        "selection_quality_status",
    ]
]
```

!!! warning "Lowest means all candidates were checked"

    If any requested topology is missing or invalid, FRUST does not silently
    choose among the remaining structures. The selection and dependent
    barriers are marked `incomplete` or `invalid`. Use an explicit
    `dimer_reference="dimer"`, `"dimer_bh_bridged"`, or
    `"dimer_eight_membered"` only when that topology is intentionally fixed.

## Choose The Result Level

| level | geometry | energy used for screening analysis | result |
| --- | --- | --- | --- |
| `low_cost` | g-xTB or UMA, per screening plan | same screening potential | ΔE |
| `uma_ranked` | g-xTB optimized geometry | UMA SP reranking | ΔE |
| `dft_ranked` | g-xTB or UMA, per screening plan | DFT SP | ΔE |
| `full` | final UMA or DFT optimization | matching final-method energy plus frequencies | ΔE and ΔG |

`uma_ranked` is available only when UMA SP reranks g-xTB optimized geometries.
For `full`, TSs and all required references use the **same final method**.
The [UMA method choices](uma-screening.md) show which paths end at UMA and
which end at ωB97.

The default `ranking_solvation="method"` applies the method's analysis solvent
to every DFT ranking SP on a screened structure. For the current presets this normally
means SMD chloroform. Use `ranking_solvation="gas"` to opt out, or provide a
different SMD solvent name.

```python
ranked = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="dft_ranked",
    method="r2scan-3c",
)

ranked.show_stages()[["branch", "stage", "solvent"]]
```

The selected level applies to TSs and every molecular dependency. A ranked TS
is therefore combined only with equally ranked ligand, dimer, HBpin, and H2
energies.

## Screen With UMA Before ωB97 Validation

For a small input table, both UMA environments, the reranking and full UMA
choices, use the [UMA method choices](uma-screening.md). The abbreviated
full-run example below shows how that choice fits into the composed workflow.

The following full run selects conformers with ALPB-corrected UMA, then sends
the retained structures directly to ωB97 validation:

```python
uma_screen = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    include_dft_rank_sp=False,
)

uma_screen.show_stages()[["branch", "stage", "engine", "solvent"]]
```

| Stage | Meaning |
| --- | --- |
| `xtb_preopt` | Constrained GFN-FF preoptimization for TSs and INT3 |
| `uma_sp` | UMA single-point energies, including the selected ALPB correction |
| `uma_sp_filter` | Keep the lowest `top_n` conformers by UMA single-point energy |
| `uma_opt` | Constrained UMA optimization for TSs and INT3 |
| `dft_preopt` onward | ωB97 optimization, TS/frequency checks, and final chloroform SP |

The same `screening="uma-gas"` option removes the ALPB correction. The
`low_cost` result uses `uma_opt-EE` and `uma_opt-oc`; `dft_ranked` adds the
ωB97 ranking single point. For a `full` run, set
`include_dft_rank_sp=True` if you also want that ranking point and its
separate `dft_ranked` result tier. The default for a full UMA-screened ωB97 run is `False`;
the existing g-xTB default remains `True`.

For the default UMA full run, `run.available_analysis_levels()` returns
`("low_cost", "full")`. With `include_dft_rank_sp=True`, it returns
`("low_cost", "dft_ranked", "full")`. A direct `level="dft_ranked"` request
always performs the ranking single point.

The initial TS guess and its constraints are chosen separately from the UMA
potential. By default, ωB97 validation selects the ωB97 gas guess profile.
To use the reviewed UMA **gas** profile instead:

```python
uma_screen = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    screening="uma-alpb-chloroform",
    method="wb97xd3-631g",
    level="full",
    spec_profile="omol-uma-s-1p2p1/gas",
    spec_match="exact",
)
```

!!! note "Guess profile and calculation solvent are independent"

    The example uses gas-phase UMA reference geometry and constraints to
    construct a guess, then calculates the screening stages with the UMA plus
    ALPB(chloroform) potential. There is currently no reviewed UMA ALPB
    geometry profile.

## Preview Structures Before Submission

Preview one representative system across the important workflow branches before
starting any calculators:

```python
targets = wf.targets()
sample_system = targets[0].target.system.system_name

wanted_states = {
    "TS1",
    "TS3",
    "dimer",
    "dimer_bh_bridged",
    "dimer_eight_membered",
    "int1",
    "int2",
    "INT3",
}

preview_indices = [
    index
    for index, item in enumerate(targets)
    if item.target.system.system_name == sample_system
    and item.target.state_id in wanted_states
]

preview = wf.preview(
    targets=preview_indices,
    n_confs=1,
    n_cores=4,
)

preview[["state_id", "system_name", "catalyst_name", "custom_name"]]
```

Inspect the generated structures in an interactive 3D grid:

```python
ft.plot_mols(
    preview,
    legends=preview["custom_name"].tolist(),
    columns=2,
    cell_size=(520, 440),
    show_labels=True,
    linked=False,
)
```

For transition states and `INT3`, inspect the role assignments and constraints
alongside the structures:

```python
preview.loc[
    preview["constraint_roles"].notna(),
    ["custom_name", "constraint_roles", "constraint_spec"],
]
```

`preview()` uses the same structure builders as `run()` and `submit()`, but stops
after graph construction and 3D embedding. It does not run pruning, xTB, g-xTB,
ORCA, or scheduler submission. With `scope="barriers"`, cycle-only states are
simply absent from the selected targets; with `scope="full_cycle"`, the same
snippet also previews `int1`, `int2`, and `INT3`.

!!! tip "Choose chemically difficult examples"

    For a large enumerated screen, replace `sample_system` with a system known
    to exercise unusual chemistry, such as a catalyst substituted directly on
    boron or one containing additional nitrogen atoms. Previewing a small,
    deliberately difficult subset is more informative than drawing the first
    several nearly identical targets.

## Submit The Complete Run

```python
from frust.cluster import ClusterConfig, Resources

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs",
)

resources = {
    "init": Resources(cpus=12, mem_gb=12, timeout_min=7200),
    "dft_opt": Resources(cpus=12, mem_gb=12, timeout_min=7200),
    "dft_hessian": Resources(cpus=12, mem_gb=24, timeout_min=7200),
    "dft_ts_opt": Resources(cpus=12, mem_gb=12, timeout_min=7200),
    "dft_freq": Resources(cpus=12, mem_gb=24, timeout_min=7200),
    "dft_solv_sp": Resources(cpus=12, mem_gb=12, timeout_min=7200),
}

submission = wf.submit(
    out_dir="results",
    cluster=cluster,
    execution="dft_staged",
    stage_resources=resources,
)
```

!!! note "ORCA memory fraction in composed screens"

    `CatalystScreenWorkflow.submit()` currently uses the underlying child
    workflow default `orca_memory_fraction=0.8` and does not expose an override.
    Slurm receives the complete `Resources.mem_gb` request, while ORCA receives
    80% of it. Thus, `mem_gb=20` allocates 20 GB to the job and approximately
    16 GB to ORCA. The setting is irrelevant for `level="low_cost"` and applies
    to ORCA calculations in `dft_ranked` and `full` runs.

The TS and minimum workflows retain their own chemically homogeneous stage
graphs. Their collectors run independently, followed by one lightweight
finalizer that creates the portable analysis tables.

!!! note "One method per run"

    Use separate run directories for `r2scan-3c` and `r2scan-3c-solv`. This
    keeps each manifest and reference fingerprint unambiguous.

## Analyze Anywhere

Copy the complete `results/` directory from the cluster, then open it without
the original workflow object, CSV, or scheduler:

```python
run = ft.screen.open_run("results")

run.summary()
barriers = run.barriers()
barriers
```

| substrate_name | catalyst_name | rpos | ts_type | delta_e_kcal_mol | delta_g_corrected_kcal_mol | quality_status |
| --- | --- | ---: | --- | ---: | ---: | --- |
| pyrrole | NMe | 2 | TS1 | 23.0 | 21.4 | review |
| pyrrole | NMe | 2 | TS2 | 19.1 | 17.8 | ready |
| pyrrole | NMe | 2 | TS3 | 28.0 | 26.1 | ready |
| pyrrole | NMe | 2 | TS4 | 26.2 | 24.7 | invalid |

A full run also preserves the independently selected winner at each cheaper
level. This gives the screening electronic barrier, the DFT-ranked electronic
barrier when ranking is enabled, and the final DFT electronic and Gibbs
barriers from one submission:

```python
run.available_analysis_levels()  # example with DFT ranking enabled
# ('low_cost', 'dft_ranked', 'full')

comparison = run.compare_barriers()
comparison[[
    "ts_type",
    "delta_e_low_cost_kcal_mol",
    "delta_e_dft_ranked_kcal_mol",
    "delta_e_full_kcal_mol",
    "delta_g_full_kcal_mol",
]]
```

Use `run.barriers(level="dft_ranked")` to inspect one tier in the usual tidy
format. Gibbs columns are intentionally empty at `low_cost` and `dft_ranked`;
frequencies are only calculated at `full`. With
`dimer_reference="lowest"`, FRUST selects the lowest qualified dimer again at
each level, so a screening comparison does not silently reuse the full-DFT
dimer choice.

!!! note "These are exact checkpoints"

    FRUST saves the winning structure immediately after each ranking stage.
    It does not reconstruct a screening result from the conformer that later
    wins full DFT refinement.

The run directory is self-describing:

```text
results/
├── manifest.json
├── run_report.json
├── calculations/
│   ├── transition_states/
│   │   ├── merged.parquet
│   │   └── tiers/
│   │       ├── low_cost/merged.parquet
│   │       └── dft_ranked/merged.parquet
│   └── references/
│       ├── merged.parquet
│       └── tiers/
│           ├── low_cost/merged.parquet
│           └── dft_ranked/merged.parquet
└── analysis/
    ├── states.parquet
    ├── dimer_references.parquet
    ├── barriers.parquet
    ├── states_by_level.parquet
    ├── dimer_references_by_level.parquet
    ├── barriers_by_level.parquet
    ├── profiles.parquet       # full_cycle only
    ├── profiles_by_level.parquet  # full_cycle only
    ├── reviews.csv
    └── report.json
```

`states.parquet` always shows the selected electronic energy, geometry and
energy stages, method, basis, solvent, result ID, and quality status. Full runs
also contain frequency Gibbs energy, thermal correction, assembled `G`, and
vibration classification. `barriers.parquet` contains `delta_e_kcal_mol` for
every level and Gibbs columns only for full runs.
`calculations/references/merged.parquet` contains the campaign's retained
reference candidates. Each barrier records the chosen rows in
`reference_result_ids`. `run_report.json` summarizes collection and
finalization. Its `terminal_results` section lists missing target IDs and
reference targets with no qualified candidate.

| Minimum vibration result | Run quality |
| --- | --- |
| Zero imaginary modes | `ready` when the other quality checks pass |
| One imaginary mode with magnitude below 50 cm⁻¹ | `review` |
| Stronger or multiple imaginary modes | `invalid` |

When a dimer is `review` or `invalid`, FRUST still calculates numerical barriers
from the available energies. Each dependent row inherits the dimer status and
records `dependency_review:dimer` or `dependency_invalid:dimer` in
`quality_issues`. Full-cycle profile rows inherit the same baseline quality.

Reusing an existing run directory is allowed only when its scientific run
signature matches. A changed input, method, correction, or conformer protocol
requires a new output directory.

!!! note "Earlier run formats"

    Use the FRUST version that created an earlier catalyst-screen bundle to
    open or resume it. Start new calculations in a fresh directory.

For gas-frequency calculations followed by a solvent single point, FRUST uses

```text
G = E_solv + (G_freq - E_freq)
```

For solvent-inclusive frequency calculations, it uses `G = G_freq`. The
resolved expression and all contributing columns are recorded in the run.

## Review Every TS Mode Once

TS-mode review applies to `level="full"`. The two electronic-only levels have
no frequencies and return an empty review queue.

```python
queue = run.review_queue()
queue[["result_id", "state_id", "system_name", "rpos", "n_imag", "vibration_flags"]]
```

Inspect the structure and mode, then persist the decision:

```python
result_id = queue.iloc[0]["result_id"]
run.plot_vibration(result_id)
run.set_review(
    result_id,
    "approved",
    note="Imaginary mode follows the intended B-H/C-H transfer coordinate.",
)
```

Reviews are keyed to the actual result content. Recalculation or a changed
geometry produces a new result ID and therefore requires a new review.

## Full Catalytic Cycle

```python
wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="full",
    method="r2scan-3c",
    scope="full_cycle",
)
```

This adds catalyst, `int1`, `int2`, `HBpin-ligand`, and `INT3` calculations.
FRUST balances every profile state to the same overall composition:

| profile state | absolute combination |
| --- | --- |
| Dimer | 1/2 selected dimer + ligand + HBpin |
| Cat | catalyst + ligand + HBpin |
| TS1, int1, TS2 | state + HBpin |
| int2 | int2 + H2 + HBpin |
| TS3, INT3, TS4 | state + H2 |
| Product | 1/2 selected dimer + HBpin-ligand + H2 |

The Dimer row is zero. The literal `-1.89 kcal/mol` Gibbs correction is applied
only to TS1 and TS3 and is recorded separately in `manifest.json`. Electronic
profiles never receive that correction.

Consequently, the Product energy is the balanced borylation energy
`E(HBpin-ligand) + E(H2) - E(ligand) - E(HBpin)`. Catalyst activation is not
added to the Product a second time.

```python
profile = run.profile(system_name="pyrrole__NMe", rpos=2)
run.plot_profile(system_name="pyrrole__NMe", rpos=2)
```

When the run calculated multiple dimer topologies, select any available dimer
directly for an exploratory profile:

```python
run.plot_profile(
    system_name="pyrrole__NMe",
    rpos=2,
    dimer_reference="dimer_bh_bridged",
)
```

Accepted explicit choices are `"dimer"`, `"dimer_bh_bridged"`, and
`"dimer_eight_membered"`; `"lowest"` repeats automatic selection. This
rebuilds the profile in memory from stored results. It does not calculate a
missing dimer or change the run's manifest and saved analysis tables.

States with missing or invalid dependencies remain visible in the tables but
are omitted from plots by default.
