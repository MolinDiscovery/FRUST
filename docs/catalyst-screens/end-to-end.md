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

wf.plan()[["branch", "state_id", "system_name", "rpos", "action"]]
```

| branch | state_id | system_name | rpos | action |
| --- | --- | --- | ---: | --- |
| transition_states | TS1 | pyrrole__NMe | 2 | calculate |
| transition_states | TS2 | pyrrole__NMe | 2 | calculate |
| references | ligand | pyrrole__NMe |  | reuse |
| references | dimer | pyrrole__NMe |  | calculate |
| references | HBpin-mol | pyrrole__NMe |  | reuse |
| references | HH | pyrrole__NMe |  | reuse |

`plan()` is calculation-free. It shows which structures will be calculated and
which approved scientific references can be reused.

## Choose The Result Level

| level | geometry | energy used for screening analysis | result |
| --- | --- | --- | --- |
| `low_cost` | g-xTB | g-xTB | ΔE |
| `dft_ranked` | g-xTB | DFT SP | ΔE |
| `full` | DFT | final DFT energy plus frequencies | ΔE and ΔG |

The default `ranking_solvation="method"` applies the method's analysis solvent
to every DFT SP on a g-xTB structure. For the current presets this normally
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

The TS and minimum workflows retain their own chemically homogeneous stage
graphs. Their collectors run independently, followed by one lightweight
finalizer that creates the portable analysis tables.

!!! note "One method per run"

    Use separate run directories for `r2scan-3c` and `r2scan-3c-solv`. This
    keeps each manifest and reference fingerprint unambiguous.

## Analyze Anywhere

Copy the complete `results/` directory from the cluster, then open it without
the original workflow object, CSV, scheduler, or shared reference library:

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

The run directory is self-describing:

```text
results/
├── manifest.json
├── run_report.json
├── calculations/
│   ├── transition_states/
│   │   └── merged.parquet
│   └── references/
│       ├── computed.parquet
│       ├── reused.parquet
│       ├── merged.parquet
│       ├── index.parquet
│       ├── reviews.csv
│       └── entries/
└── analysis/
    ├── states.parquet
    ├── barriers.parquet
    ├── profiles.parquet       # full_cycle only
    ├── reviews.csv
    └── report.json
```

`states.parquet` always shows the selected electronic energy, geometry and
energy stages, method, basis, solvent, result ID, and quality status. Full runs
also contain frequency Gibbs energy, thermal correction, assembled `G`, and
vibration classification. `barriers.parquet` contains `delta_e_kcal_mol` for
every level and Gibbs columns only for full runs.
Within `calculations/references/`, `computed.parquet` contains references made
for this run, `reused.parquet` contains snapshots from the shared library, and
`merged.parquet` is their analysis-ready union. `entries/` is the inspectable
scientific evidence behind those rows. `run_report.json` summarizes child-job
collection and finalization; it does not contain a second copy of the energies.

Reusing an existing run directory is allowed only when its scientific run
signature matches. A changed input, method, correction, or conformer protocol
raises instead of silently mixing old and new outputs.

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

## Inspectable Reference Library

Set a shared library once on the cluster:

```bash
export FRUST_REFERENCE_STORE=/groups/kemi/jmni/grow/frust_reference_library
```

The default production reuse policy is `"approved"`. A new reference is
automatically checked for termination, required thermochemistry, and zero
imaginary frequencies, but it must be inspected once before future runs reuse
it:

```python
library = ft.screen.open_reference_library(
    "/groups/kemi/jmni/grow/frust_reference_library"
)

library.review_queue()
reference_id = library.review_queue().iloc[0]["reference_id"]
reference = library.get(reference_id)

reference.summary()
reference.view()
reference.plot_vibrations()
reference.xyz_path()
reference.files()

reference.approve(note="Correct NMe dimer minimum")
```

Each immutable entry contains `metadata.json` plus its checksum, the full FRUST
result parquet, `optimized.xyz`, and retained final-stage calculator inputs and
outputs. The compatibility key represents molecule + method + protocol; the
reference ID additionally fingerprints the resulting geometry, energies, and
vibrations. A reused entry is copied into the run, so completed work never
depends on the continued existence or contents of the shared library.

!!! warning "Reject instead of replacing"

    `reference.reject(...)` preserves the audit trail and prevents future
    reuse. A recalculation becomes a new immutable entry with a new reference
ID; old runs retain their original snapshots.

The store separates automatically reusable screening artifacts from reviewed
thermochemical references:

```text
entries/
├── screening/
│   ├── low_cost/
│   └── dft_ranked/
└── references/
    └── full/
```

Use `reuse_policy="auto_valid"` only when automatic minimum checks are an
acceptable substitute for manual structure approval.

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
| Dimer | 1/2 dimer + ligand + HBpin |
| Cat | catalyst + ligand + HBpin |
| TS1, int1, TS2 | state + HBpin |
| int2 | int2 + H2 + HBpin |
| TS3, INT3, TS4 | state + H2 |
| Product | catalyst + HBpin-ligand + H2 |

The Dimer row is zero. The literal `-1.89 kcal/mol` Gibbs correction is applied
only to TS1 and TS3 and is recorded separately in `manifest.json`. Electronic
profiles never receive that correction.

```python
profile = run.profile(system_name="pyrrole__NMe", rpos=2)
run.plot_profile(system_name="pyrrole__NMe", rpos=2)
```

States with missing or invalid dependencies remain visible in the tables but
are omitted from plots by default.
