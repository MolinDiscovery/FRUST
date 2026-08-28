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
    -> approve reference structures once
    -> reuse those references in later screens
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

## 2. Choose A Reference Library

References are ordinary scientific files rather than an opaque cache. Choose a
shared directory visible from the cluster:

```bash
export FRUST_REFERENCE_STORE=/groups/kemi/jmni/grow/frust_reference_library
```

Alternatively, pass the path explicitly when constructing the workflow. The
environment variable is convenient because the same notebook can be moved
between run directories without editing it.

The library will eventually look like:

```text
frust_reference_library/
├── index.parquet
├── reviews.csv
└── entries/
    ├── screening/
    │   ├── low_cost/
    │   └── dft_ranked/
    └── references/
        └── full/
            └── r2scan_3c/
                └── dimer/
                    └── tmp_bcat/
                        └── ref_<content-id>/
```

The compatibility key represents the molecule, method, and calculation
protocol. The reference ID also fingerprints the resulting geometry, energies,
and vibrations. Consequently, a changed recalculation can coexist with the old
entry instead of overwriting it.

## 3. Construct The Complete Workflow

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

## 4. Inspect What Will Be Calculated

```python
plan = wf.plan()
plan[["branch", "state_id", "system_name", "rpos", "action"]]
```

On the first run, the compact plan is similar to:

| branch | state_id | system_name | rpos | action |
| --- | --- | --- | ---: | --- |
| transition_states | TS1 | n_methyl_pyrrole__tmp_bcat | 2 | calculate |
| transition_states | TS2 | n_methyl_pyrrole__tmp_bcat | 2 | calculate |
| transition_states | TS3 | n_methyl_pyrrole__tmp_bcat | 2 | calculate |
| transition_states | TS4 | n_methyl_pyrrole__tmp_bcat | 2 | calculate |
| references | ligand | n_methyl_pyrrole__tmp_bcat |  | calculate |
| references | dimer | n_methyl_pyrrole__tmp_bcat |  | calculate |
| references | HBpin-mol | n_methyl_pyrrole__tmp_bcat |  | calculate |
| references | HH | n_methyl_pyrrole__tmp_bcat |  | calculate |

`ligand` is the free substrate in the established FRUST state naming, while
`HH` is H2. An approved compatible reference changes only `action` from
`calculate` to `reuse`.

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

## 5. Submit TSs And References Together

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

submission.finalization_job_id
submission.child_submissions
```

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

The finalizer uses an `afterany` dependency. It therefore produces a report and
partial analysis even when an upstream branch contains failed calculations.

!!! warning "Use a deliberate results directory"

    A run directory carries a scientific signature. Reusing it with the same
    input and protocol supports recovery, but changing the method, input,
    corrections, or conformer protocol raises instead of mixing calculations.

## 6. Open The Portable Run

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

## 7. Read The Barrier Table

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

## 8. Review Transition-State Modes

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

## 9. Inspect And Approve Molecular References

Full references are checked for normal termination, complete thermochemistry,
and their vibration pattern before publication. A minimum with zero imaginary
frequencies receives `validation_status="auto_valid"`. A minimum with exactly
one weak imaginary frequency (`abs(frequency) < 50 cm^-1`) is published with
`validation_status="review"`, while stronger or multiple imaginary frequencies
are retained in the run but not published to the reference library.

Neither `auto_valid` nor `review` full references are reused by the default
`reuse_policy="approved"` until you inspect and approve them. A review-quality
reference remains ineligible for `reuse_policy="auto_valid"`, even after manual
approval.

`low_cost` and `dft_ranked` entries are stored separately as screening
artifacts. Exact protocol matches can be reused automatically because they are
not presented as approved DFT minima or thermochemical references. Their XYZ,
result parquet, metadata, and retained calculator evidence remain inspectable.

```python
library.search(calculation_level="dft_ranked")[[
    "reference_id",
    "state_id",
    "electronic_energy_hartree",
]]
```

```python
import os

library = ft.screen.open_reference_library(
    os.environ["FRUST_REFERENCE_STORE"]
)

reference_queue = library.review_queue()
reference_queue[[
    "reference_id",
    "state_id",
    "compound_name",
    "formula",
    "method",
    "validation_status",
    "free_energy_hartree",
]]
```

Open one entry and inspect all of its evidence:

```python
reference_id = reference_queue.iloc[0]["reference_id"]
reference = library.get(reference_id)

reference.summary()
reference.view()
reference.plot_vibrations(mode=0)
reference.xyz_path()
reference.files()
reference.dataframe()
```

`reference.files()` exposes retained ORCA inputs and outputs. `result.parquet`
contains the complete canonical FRUST row, while `metadata.json` records the
method plan, thermochemistry, validation, provenance, and checksums.

Approve the entry only after inspection:

```python
reference.approve(
    reviewer="JMN",
    note="Correct tmp_bcat dimer minimum and expected connectivity.",
)
```

If inspection fails, use `reference.reject(...)` *instead of* the approval
call. Rejection preserves the entry for auditing but excludes it from reuse;
for example, the review note could be `"Collapsed to the wrong dimer
geometry."`.

!!! note "Approval affects later runs"

    A newly calculated reference is already part of the run that produced it.
    Approval controls whether future workflows may reuse that shared entry.
    Inspect a `validation_status="review"` entry particularly carefully before
    approving it; reoptimization is often preferable when the mode is chemically
    meaningful rather than a weak peripheral motion.

## 10. Confirm Reuse Before The Next Submission

After approving the references, construct the next workflow normally:

```python
next_wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="full",
    method="r2scan-3c",
    scope="barriers",
)

next_wf.plan().query("branch == 'references'")[[
    "state_id",
    "action",
    "reference_id",
]]
```

The compatible approved rows now show `action="reuse"`. Every reused entry is
copied into the new run bundle, including its XYZ, result parquet, metadata,
review, checksums, and retained calculator files. The completed run therefore
does not depend on the shared library remaining available.

## 11. Move And Reopen The Results

Copy the whole `results/` directory from the cluster. Analysis does not require
the original CSV, workflow object, scheduler, or shared reference library:

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
│   ├── transition_states/
│   └── references/
│       ├── computed.parquet
│       ├── reused.parquet
│       ├── merged.parquet
│       ├── publication_report.json
│       ├── index.parquet
│       ├── reviews.csv
│       └── entries/
└── analysis/
    ├── states.parquet
    ├── barriers.parquet
    ├── reviews.csv
    └── report.json
```

## 12. Extend The Run To A Full Cycle

Use a new output directory and change the scope:

```python
cycle_wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    level="full",
    method="r2scan-3c",
    scope="full_cycle",
)

cycle_wf.plan()[["branch", "state_id", "action"]]
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

Every profile row is balanced to the same total atomic composition. Missing or
invalid dependencies remain in the table, while plots omit invalid and
incomplete states by default.

## Common Problems

| Symptom | Meaning | Next action |
| --- | --- | --- |
| Reference plan still says `calculate` | No compatible approved entry exists | Check method/protocol fingerprints and approve the intended entry |
| Barrier is `incomplete` | At least one equation term is missing | Inspect `run.states()` and child collection reports |
| Barrier is `invalid` | A dependency failed termination, vibration, composition, or review checks | Read `quality_issues` and inspect the raw result |
| Dimer and barriers are `review` | The dimer has one weak imaginary mode below 50 cm^-1 | Inspect the dimer mode; approve the library entry only if scientifically acceptable, otherwise reoptimize |
| Reference row exists but was not published | Automatic reference validation rejected it | Read `calculations/references/publication_report.json` for the explicit reason |
| TS remains `review` after approval | A low-frequency flag remains or a different result ID was approved | Compare the queue and approved `result_id` |
| `FileExistsError` for `out_dir` | Existing manifest has a different scientific signature | Choose a new directory for the changed run |
| Reference checksum failure | An entry file changed after publication | Preserve it for diagnosis and publish a clean recalculation |

The minimal executable companion is
[`examples/end_to_end_catalyst_screen.ipynb`](https://github.com/MolinDiscovery/FRUST/blob/main/examples/end_to_end_catalyst_screen.ipynb).
For a field-by-field description of the run bundle and reference library, see
[End-To-End Calculation And Analysis](../catalyst-screens/end-to-end.md).
