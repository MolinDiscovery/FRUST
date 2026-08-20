# End-To-End Calculation And Analysis

Use one composed workflow when a TS screen should arrive together with the
references needed to interpret it:

```python
import frust as ft

wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
    method="r2scan-3c",
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

| substrate_name | catalyst_name | rpos | ts_type | barrier_kcal_mol | quality_status |
| --- | --- | ---: | --- | ---: | --- |
| pyrrole | NMe | 2 | TS1 | 21.4 | review |
| pyrrole | NMe | 2 | TS2 | 17.8 | ready |
| pyrrole | NMe | 2 | TS3 | 26.1 | ready |
| pyrrole | NMe | 2 | TS4 | 24.7 | invalid |

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

`states.parquet` shows the electronic energy, frequency Gibbs energy, thermal
correction, assembled `G`, vibration classification, result ID, and quality
status. `barriers.parquet` is the compact table normally used for screening.
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
export FRUST_REFERENCE_STORE=/groups/kemi/jmni/frust_reference_library
```

The default production reuse policy is `"approved"`. A new reference is
automatically checked for termination, required thermochemistry, and zero
imaginary frequencies, but it must be inspected once before future runs reuse
it:

```python
library = ft.screen.open_reference_library(
    "/groups/kemi/jmni/frust_reference_library"
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

Use `reuse_policy="auto_valid"` only when automatic minimum checks are an
acceptable substitute for manual structure approval.

## Full Catalytic Cycle

```python
wf = ft.workflows.catalyst_screen(
    csv_path="screen.csv",
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

The Dimer row is zero. The literal `-1.89 kcal/mol` correction is applied only
to TS1 and TS3 and is recorded in `manifest.json`.

```python
profile = run.profile(system_name="pyrrole__NMe", rpos=2)
run.plot_profile(system_name="pyrrole__NMe", rpos=2)
```

States with missing or invalid dependencies remain visible in the tables but
are omitted from plots by default.
