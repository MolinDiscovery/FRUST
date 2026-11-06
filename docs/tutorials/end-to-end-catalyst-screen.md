# End-To-End Catalyst-Screen Tutorial

This tutorial starts with one substrate and one catalyst and ends with a
portable barrier table, reviewed transition states, and reusable reference
calculations.

```text
screen.csv
    -> inspect the complete calculation plan
    -> submit TSs and molecular references together
    -> open the portable result bundle
    -> review TS vibrations
    -> inspect reference structures and their quality
    -> trace each barrier to its saved reference results
```

The first example uses `scope="barriers"`. It calculates TS1--TS4 and the four
molecular references required by the barrier equations: free substrate
(`ligand` in the current state names), dimer, HBpin, and H2.

## 1. Create A Small Screen

Create `screen.csv`:

```csv
role,smiles,compound_name,rpos
substrate,CN1C=CC=C1,n_methyl_pyrrole,2
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,
```

Here, `rpos=2` is the RDKit atom index of the aromatic C-H position to test.
Check atom labels from the exact substrate SMILES before starting a large
screen:

```python
import frust as ft

ft.DrawUniqueChGrid(["CN1C=CC=C1"])
```

!!! tip "Start with one system and one reactive position"

    First verify the structures, scheduler resources, and analysis with a
    small screen. Add substrates, catalysts, and reactive positions only after
    that path works.

## 2. Construct The Complete Workflow

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
```

The three method blocks are independent:

| Block | Controls | This example |
| --- | --- | --- |
| `screening` | Inexpensive geometry generation and optimization | `"gxtb-default"` (GFN-FF + g-xTB) |
| `ranking` | Optional single-point selection on screened geometries | omitted (implicit DFT ranking SP) |
| `method` | Final validation calculator and thermochemistry | `"r2scan-3c"` (ORCA composite DFT) |

Add a `ranking` block to select validation inputs with a different method:

```python
ranked_wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    screening="gxtb-default",
    ranking="uma-alpb-chloroform",
    ranking_top_n=1,
    method="r2scan-3c",
    level="full",
    scope="barriers",
)
```

Here g-xTB generates geometries, UMA + ALPB(chloroform) selects the best one,
and ORCA `r2scan-3c` performs the final validation. See
[UMA Method Choices](../catalyst-screens/uma-screening.md) for all UMA and
hybrid combinations, including full UMA runs and `wb97_comparison` workflows.

The default `dimer_reference="lowest"` adds `dimer`,
`dimer_bh_bridged`, and `dimer_eight_membered` to the reference plan. FRUST
selects one qualified topology per catalyst using Gibbs energy at `full` level
and electronic energy at `low_cost` or `dft_ranked` level. The exact selected
result row supplies both energies used downstream. After the run, inspect the
decision with `run.dimer_references()`.

!!! warning "Strict topology comparison"

    A missing or invalid candidate prevents a `lowest` selection; dependent
    barriers remain flagged. Pass an explicit topology such as
    `dimer_reference="dimer_bh_bridged"` only when the reference should be
    fixed instead of compared.

The three calculation levels answer different questions:

| level | final geometry | analysis energy | available result |
| --- | --- | --- | --- |
| `low_cost` | g-xTB | g-xTB | ΔE |
| `dft_ranked` | g-xTB | solvent DFT SP | ΔE |
| `full` | DFT | final DFT electronic energy plus frequencies | ΔE and ΔG |

`ranking_solvation="method"` is the default. It gives the DFT ranking SP the
same solvent as the method's final analysis energy; the current production
presets resolve this to SMD chloroform. Use `ranking_solvation="gas"` for an
explicit gas-phase ranking SP, or provide another SMD solvent name such as
`"toluene"`.

For example, a quick solvent-ranked screen is:

```python
ranked = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="dft_ranked",
    method="r2scan-3c",
)
```

FRUST performs the same solvent DFT SP for every equation term: TS, ligand,
dimer, HBpin, and H2. It will not combine DFT-ranked TS energies with g-xTB
reference energies.

The `r2scan-3c` preset performs gas-phase frequencies followed by a solvent
single point. Its recorded molecular free-energy expression is:

```text
G = E_solv + (G_freq - E_freq)
```

The solvent-inclusive `r2scan-3c-solv` preset instead uses `G = G_freq`. Keep
different methods in separate run directories; method fingerprints prevent
their references from being mixed.

Inspect the recorded recipe:

```python
wf.method.thermochemistry.to_dict()
wf.method.fingerprint()
```

No structures or calculators have run at this point.

## 3. Inspect What Will Be Calculated

```python
plan = wf.plan()
plan[["branch", "state_id", "system_name", "rpos"]]
```

For this input, the compact plan looks like this.

| branch | state_id | system_name | rpos |
| --- | --- | --- | ---: |
| transition_states | TS1 | n_methyl_pyrrole__tmp_bcat | 2 |
| transition_states | TS2 | n_methyl_pyrrole__tmp_bcat | 2 |
| transition_states | TS3 | n_methyl_pyrrole__tmp_bcat | 2 |
| transition_states | TS4 | n_methyl_pyrrole__tmp_bcat | 2 |
| references | ligand | n_methyl_pyrrole__tmp_bcat |  |
| references | dimer | n_methyl_pyrrole__tmp_bcat |  |
| references | dimer_bh_bridged | n_methyl_pyrrole__tmp_bcat |  |
| references | dimer_eight_membered | n_methyl_pyrrole__tmp_bcat |  |
| references | HBpin-mol | n_methyl_pyrrole__tmp_bcat |  |
| references | HH | n_methyl_pyrrole__tmp_bcat |  |

`ligand` is the free substrate in the established FRUST state naming, while
`HH` is H2. The three dimer rows are the candidate topologies compared by
`dimer_reference="lowest"`. FRUST plans each required molecular reference once
for the campaign, even if several barriers depend on it.

Inspect the calculation stages and their cluster resource-group names:

```python
stages = wf.show_stages(execution="dft_staged")
stages[["branch", "group", "stage", "engine", "options"]]
```

Preview one generated structure before submitting calculations:

```python
preview = wf.preview(targets=[0], n_confs=1)
ft.plot_mols(preview)
```

Both `plan()` and `show_stages()` are calculation-free. `preview()` generates
and embeds only the selected structure; it does not run xTB, g-xTB, or ORCA.

## 4. Submit TSs And References Together

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
    artifact_policy="compact",
)

submission.finalization_job_id
submission.child_submissions
```

For hundreds or thousands of systems, `artifact_policy="compact"` keeps
portable scientific results and uses scratch for routine calculator output.
Branch collection can release obsolete checkpoints from successful targets
while preserving the final and tier inputs needed for finalization or retry.
Full finalization releases the per-target copies and bulk Submitit files.

A full TS row in the collected screening result is compact:

| Data kept | Example column |
| --- | --- |
| identity | `structure_id`, `system_name`, `state_id`, `rpos`, `cid` |
| selected structure | `atoms`, `dft_ts_opt-oc`, `connectivity_bonds` |
| TS definition | `constraint_roles`, `constraint_spec` |
| energies | `dft_freq-EE`, `dft_freq-GE`, `dft_solv_sp-EE` |
| frequencies | `dft_freq-frequencies_cm1` |
| completion | `dft_ts_opt-NT`, `dft_freq-NT`, `dft_solv_sp-NT` |

The frequency column contains every frequency value. Full TS results also
retain their final displacement vectors for animation with `ft.plot_vibs(...)`.

After a successful screening run, the portable bundle looks like:

```text
results/
├── .submitit/
│   ├── ownership.json
│   └── control/              # small finalizer record
├── calculations/
│   ├── transition_states/merged.parquet
│   └── references/merged.parquet
├── analysis/
├── manifest.json
└── run_report.json
```

During execution, target and collector logs live in `results/.submitit/jobs/`,
not in `ClusterConfig.log_dir`. Stderr is merged into stdout. The finalizer
deletes `jobs/` only when every required collection, reference, and analysis
artifact validates. Failed runs retain scheduler logs and failed-stage evidence.
No separate scratch filesystem is required.

The finalizer cannot safely delete its own active control files. Remove that
small remainder after the job has exited with:

```python
ft.screen.cleanup_submitit("results")
```

Use `allow_incomplete=True` only for deliberate cleanup after debugging a
failed run. FRUST refuses unowned or out-of-run paths.

!!! note "ORCA receives 80% of the requested job memory"

    The composed `catalyst_screen` submission currently uses the child
    workflow default `orca_memory_fraction=0.8`; it does not expose that
    setting as an argument. For example, `Resources(mem_gb=20)` requests 20 GB
    from Slurm and makes approximately 16 GB available to ORCA, leaving the
    remainder for Python and job overhead. This has no effect on
    `level="low_cost"`, because that level does not run ORCA. It applies to the
    ORCA stages in `dft_ranked` and `full` runs.

FRUST submits chemically homogeneous child workflows:

```text
transition-state jobs ──> TS collector ───────┐
                                               ├─> portable-analysis finalizer
minimum-reference jobs ─> reference collector ┘
```

The finalizer uses an `afterany` dependency. It therefore records a failed or
partial `run_report.json` even when an upstream branch fails. It refuses to
mark the run successful—or remove logs—when a required collection report is
missing or incomplete.

!!! warning "Use a deliberate results directory"

    A run directory carries a scientific signature. Reusing it with the same
    input and protocol supports recovery, but changing the method, input,
    corrections, or conformer protocol raises instead of mixing calculations.

## 5. Open The Portable Run

After the finalization job completes:

```python
run = ft.screen.open_run("results")
run.summary()
```

The four quality states mean:

| quality status | Meaning |
| --- | --- |
| `ready` | Automatic checks passed and no manual decision remains. |
| `review` | The result is usable for inspection but needs TS-mode review or has one weak imaginary minimum mode (`abs(frequency) < 50 cm^-1`). |
| `invalid` | Termination, vibration, composition, or manual-review checks failed. |
| `incomplete` | A required result or free-energy component is missing. |

Inspect the molecular-state audit table:

```python
states = run.states()
states[[
    "state_id",
    "system_name",
    "rpos",
    "electronic_energy_hartree",
    "free_energy_hartree",  # present only for level="full"
    "energy_method",
    "solvent",
    "n_imag",
    "review_status",
    "quality_status",
    "quality_issues",
]]
```

This table retains flagged rows. Nothing is silently removed from the
scientific audit trail.

For example, a dimer with one small imaginary frequency remains available:

```python
states.query("state_id == 'dimer'")[[
    "n_imag",
    "imaginary_frequencies_cm1",
    "vibration_flags",
    "quality_status",
]]
```

| n_imag | imaginary_frequencies_cm1 | vibration_flags | quality_status |
| ---: | ---: | --- | --- |
| 1 | -8.14 | weak_minimum_imag | review |

!!! note "Retention is not acceptance"

    FRUST keeps completed `review` and `invalid` calculations so their energies
    and evidence remain auditable. Their status is propagated to dependent
    barriers and profiles; it does not mean the structure has passed scientific
    validation.

## 6. Read The Barrier Table

```python
barriers = run.barriers()
barriers[[
    "substrate_name",
    "catalyst_name",
    "rpos",
    "ts_type",
    "delta_e_kcal_mol",
    "delta_g_kcal_mol",
    "g_correction_kcal_mol",
    "delta_g_corrected_kcal_mol",
    "quality_status",
]]
```

Because this was submitted with `level="full"`, the same portable run also
contains the independently selected low-cost and DFT-ranked screening results:

```python
comparison = run.compare_barriers()
comparison[[
    "substrate_name",
    "catalyst_name",
    "rpos",
    "ts_type",
    "delta_e_low_cost_kcal_mol",
    "delta_e_dft_ranked_kcal_mol",
    "delta_e_full_kcal_mol",
    "delta_g_full_kcal_mol",
]]
```

The three electronic columns can differ because each level selects its own
lowest conformer. When `dimer_reference="lowest"`, each level also chooses its
own lowest qualified dimer topology. Inspect one tier with the same tidy API:

```python
screening_barriers = run.barriers(level="dft_ranked")
full_barriers = run.barriers(level="full")
```

`delta_g_kcal_mol` and `delta_g_corrected_kcal_mol` are empty in the screening
table because low-cost and DFT-ranked stages do not calculate frequencies.
They are populated only for `level="full"`.

!!! note "One full submission, three analysis tiers"

    The lower-tier rows are saved when their ranking stages finish. They are
    not approximations reconstructed from the final full-DFT conformer.

FRUST evaluates the same stoichiometric expressions for electronic and Gibbs
energies. Electronic barriers are available at every level:

```text
TS1/TS2 ΔE = (2 × (E_TS - E_ligand) - E_dimer) / 2
TS3/TS4 ΔE = (2 × (E_TS - E_ligand - E_HBpin + E_H2) - E_dimer) / 2
```

The full level additionally evaluates:

```text
TS1 = -1.89 + (2 × (G_TS - G_ligand) - G_dimer) / 2
TS2 =         (2 × (G_TS - G_ligand) - G_dimer) / 2
TS3 = -1.89 + (2 × (G_TS - G_ligand - G_HBpin + G_H2) - G_dimer) / 2
TS4 =         (2 × (G_TS - G_ligand - G_HBpin + G_H2) - G_dimer) / 2
```

All energy differences are converted from Hartree to kcal/mol. The literal
`-1.89 kcal/mol` correction is Gibbs-only: it is recorded separately and is
never silently added to `delta_e_kcal_mol`.

## 7. Review Transition-State Modes

This section applies only to `level="full"`. Electronic-only runs have no
frequency calculation, so `run.review_queue()` is empty and ΔE quality is
based on calculation completion and protocol consistency instead.

List results awaiting manual review:

```python
queue = run.review_queue()
queue[[
    "result_id",
    "state_id",
    "system_name",
    "rpos",
    "imaginary_frequencies_cm1",
    "vibration_flags",
]]
```

Inspect the imaginary mode:

```python
result_id = queue.iloc[0]["result_id"]
run.plot_vibration(result_id, mode=0)
```

Persist the scientific decision:

```python
run.set_review(
    result_id,
    "approved",
    reviewer="JMN",
    note="Imaginary mode follows the intended B-H/C-H transfer coordinate.",
)
```

Use `"rejected"` when the mode does not describe the intended reaction. The
barrier remains visible but receives `quality_status="invalid"`.

Reviews are tied to the result geometry, energy, frequencies, and normal-mode
content. A changed recalculation receives a new result ID and must be reviewed
again.

## 8. Inspect Molecular References

Find the exact free-substrate and dimer results used by a TS1 barrier in the
run's state table.

```python
import json

ts1 = run.barriers().query("ts_type == 'TS1'").iloc[0]
reference_ids = json.loads(ts1["reference_result_ids"]).values()
run.states().loc[
    lambda states: states["result_id"].isin(reference_ids),
    ["state_id", "cid", "free_energy_hartree", "quality_status", "quality_issues"],
]
```

A full reference with one imaginary mode below 50 cm⁻¹ in magnitude receives
`review` status. Stronger or multiple imaginary modes make the result
`invalid`. Both remain visible so you can trace their effect on the barriers.
The `terminal_results` section of `run_report.json` lists missing reference
targets and targets with no qualified candidate.

## 9. Move And Reopen The Results

Copy the whole `results/` directory from the cluster. Analysis does not require
the original CSV, workflow object, or scheduler:

```python
downloaded = ft.screen.open_run("downloaded/results")

downloaded.summary()
downloaded.barriers()
downloaded.review_queue()
```

The important portable artifacts are:

```text
results/
├── manifest.json
├── run_report.json
├── calculations/
│   ├── transition_states/merged.parquet
│   └── references/
│       ├── merged.parquet
│       └── tiers/<level>/merged.parquet
└── analysis/
    ├── states.parquet
    ├── barriers.parquet
    ├── reviews.csv
    └── report.json
```

## 10. Extend The Run To A Full Cycle

Use a new output directory and change the scope:

```python
cycle_wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="full",
    method="r2scan-3c",
    scope="full_cycle",
)

cycle_wf.plan()[["branch", "state_id", "options"]]
```

This adds the isolated catalyst, `int1`, `int2`, `HBpin-ligand`, and `INT3`.
The catalyst is not required by the four barrier equations, which is why it is
added by `full_cycle` rather than the default barrier-only scope.

Submit to a different directory:

```python
cycle_submission = cycle_wf.submit(
    out_dir="results_full_cycle",
    cluster=cluster,
    execution="dft_staged",
    stage_resources=resources,
)
```

After finalization:

```python
cycle = ft.screen.open_run("results_full_cycle")

profile = cycle.profile(
    system_name="n_methyl_pyrrole__tmp_bcat",
    rpos=2,
    include_invalid=True,
)
profile[[
    "profile_state",
    "relative_e_kcal_mol",
    "relative_g_corrected_kcal_mol",
    "quality_status",
]]

cycle.plot_profile(
    system_name="n_methyl_pyrrole__tmp_bcat",
    rpos=2,
)
```

If all dimer topologies were calculated, compare the same profile against a
specific stored reference without editing or copying the result bundle:

```python
cycle.plot_profile(
    system_name="n_methyl_pyrrole__tmp_bcat",
    rpos=2,
    dimer_reference="dimer_eight_membered",
)
```

The same argument works with `cycle.profile(...)`. FRUST rebuilds this view in
memory; the recorded `dimer_reference="lowest"` result remains unchanged.

Every profile row is balanced to the same total atomic composition. Missing or
invalid dependencies remain in the table, while plots omit invalid and
incomplete states by default.

## Common Problems

| Symptom | Meaning | Next action |
| --- | --- | --- |
| Barrier is `incomplete` | At least one equation term is missing | Inspect `run.states()` and child collection reports |
| Barrier is `invalid` | A dependency failed termination, vibration, composition, or review checks | Read `quality_issues` and inspect the raw result |
| Dimer and barriers are `review` | The dimer has one weak imaginary mode below 50 cm^-1 | Inspect the dimer mode and decide whether to reoptimize |
| TS remains `review` after approval | A low-frequency flag remains or a different result ID was approved | Compare the queue and approved `result_id` |
| `FileExistsError` for `out_dir` | Existing manifest has a different scientific signature | Choose a new directory for the changed run |

The minimal executable companion is
[`examples/end_to_end_catalyst_screen.ipynb`](https://github.com/MolinDiscovery/FRUST/blob/main/examples/end_to_end_catalyst_screen.ipynb).
For a field-by-field description of the run bundle, see
[End-To-End Calculation And Analysis](../catalyst-screens/end-to-end.md).

Continue with:

- [UMA Method Choices](../catalyst-screens/uma-screening.md) for UMA screening,
  ranking, and validation combinations.
- [Workflow Method Plans](../workflows/workflow-methods.md) for calculator
  presets and stage replacement.
- [Slurm Arrays](../cluster/arrays.md) for array-based cluster submission.
- [Transition States](../troubleshooting/transition-states.md) for final TS
  validation.
