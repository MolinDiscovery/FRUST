# Deriving TS Geometry Profiles

`tsguess2` separates the chemistry that defines a TS family from the numerical
geometry associated with a calculation method:

```text
CoreTopology
  method-independent roles and distance/angle definitions
        +
StateGeometrySpec
  method/environment coordinates and constraint values
        |
        v
resolved TSGuess2Spec
        |
        v
constraint_roles + constraint_spec + ts_spec_id on every generated row
```

This page describes the maintainer workflow for turning reviewed optimized
structures into a new method-aware geometry profile. It does not describe how
to calculate the reference structures themselves.

## Where The Pieces Live

| Artifact | Responsibility |
| --- | --- |
| `frust/tsguess2/topologies.py` | Stable role names and named distance/angle definitions for each state |
| `frust/tsguess2/profiles/*.py` | Reviewed method/environment geometry profiles |
| `frust/tsguess2/profiles/__init__.py` | Profile registry, selection, and same-method environment fallback |
| `scripts/extract_tsguess2_profile.py` | Deterministic extraction of reviewable candidates from optimized parquets |

The extractor deliberately creates JSON candidates. It does **not** edit the
profile registry or make a candidate selectable in production.

## Start From Final Workflow Results

A complete reference set normally has one merged parquet for transition states
and one for constrained minima:

```text
runs/21_r2scan3c_specs_TSs_nosolv/
  merged.parquet       # TS1--TS4
  collection_report.json

runs/21_r2scan3c_specs_INT3_nosolv/
  merged.parquet       # INT3
  collection_report.json
```

The extractor reads these final columns:

| State | Coordinates | Vibrations |
| --- | --- | --- |
| `TS1`--`TS4` | `dft_ts_opt-oc` | `dft_freq-vibs` |
| `INT3` | `dft_opt-oc` | `dft_freq-vibs` |

It obtains atom indices from `constraint_roles`, takes role coordinates from
the final optimized structure, and recalculates every named internal coordinate
defined by `CORE_TOPOLOGIES`.

!!! warning "Do not copy `constraint_spec` from the source row"

    `constraint_spec` describes the constraints used to construct and
    preoptimize that source guess. It may belong to an older profile. The new
    profile values must be measured from the final `dft_ts_opt-oc` or
    `dft_opt-oc` coordinates. The extractor performs that measurement.

The final solvent single-point stage does not change geometry. For a gas-phase
profile, a later `dft_solv_sp` column may be present in the merged table, but
the extractor still uses the gas-phase optimization coordinates.

## Review Frequencies And Modes First

Check normal termination and summarize the final frequencies:

```python
import pandas as pd
import frust as ft

ts = pd.read_parquet(
    "runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet"
)

ft.summarize_ts_vibrations(
    ts,
    col="dft_freq-vibs",
    show_pos_freqs=False,
)
```

Each transition state should have exactly one chemically meaningful imaginary
mode. Frequency count alone is insufficient, so inspect the animation before
marking a state reviewed:

```python
ts3_index = ts.index[ts["state_id"].eq("TS3")][0]
ft.plot_vibs(ts, row_index=ts3_index, vId=0)
```

`INT3` is a constrained minimum and should have no imaginary frequencies.

!!! warning "Mode review is a scientific approval"

    `--mode-reviewed` records that a person inspected the imaginary mode. Do
    not pass it merely because the frequency count is correct.

## Extract A Gas-Phase Candidate

Run the extractor from the repository root:

```bash
conda run -n UMA python scripts/extract_tsguess2_profile.py \
  runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet \
  runs/21_r2scan3c_specs_INT3_nosolv/merged.parquet \
  --method r2scan-3c \
  --output profile_candidates/r2scan3c_gas.json
```

Omitting `--basis`, `--solvation-model`, and `--solvent` identifies the profile
as `r2scan-3c/gas`. Composite methods such as r2SCAN-3c do not need a separate
basis value.

Before mode review, TS candidates are emitted as `quarantined`. Once TS1--TS4
have been inspected, record that review explicitly:

```bash
conda run -n UMA python scripts/extract_tsguess2_profile.py \
  runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet \
  runs/21_r2scan3c_specs_INT3_nosolv/merged.parquet \
  --method r2scan-3c \
  --mode-reviewed TS1 TS2 TS3 TS4 \
  --output profile_candidates/r2scan3c_gas.json
```

Reviewed states with the expected frequency count are emitted as `candidate`,
not `active`.

## Extract A Solvated Candidate

Provide both the solvation model and solvent:

```bash
conda run -n UMA python scripts/extract_tsguess2_profile.py \
  runs/r2scan3c_specs_TSs_solv/merged.parquet \
  runs/r2scan3c_specs_INT3_solv/merged.parquet \
  --method r2scan-3c \
  --solvation-model smd \
  --solvent chloroform \
  --mode-reviewed TS1 TS2 TS3 TS4 \
  --output profile_candidates/r2scan3c_smd_chloroform.json
```

`GeometryKey` requires the solvation model and solvent together. Supplying only
one is rejected.

## Understand The Candidate Artifact

The JSON is an audit and review boundary:

```json
{
  "profile": "r2scan-3c/gas",
  "candidates": {
    "TS3": {
      "status": "candidate",
      "role_coordinates": {},
      "constraint_values": {},
      "reference": {
        "coordinates_column": "dft_ts_opt-oc",
        "vibrations_column": "dft_freq-vibs",
        "negative_frequencies": [-82.84],
        "mode_reviewed": true,
        "source_sha256": "..."
      }
    }
  }
}
```

The real coordinate and constraint mappings are populated with extracted
values. The abbreviated empty mappings above only show the artifact shape.

| Field | Meaning |
| --- | --- |
| `profile` | Canonical method/environment identifier |
| `status` | `candidate` after successful numerical and visual review; otherwise `quarantined` |
| `role_coordinates` | Final Cartesian coordinates for topology-required chemical roles |
| `constraint_values` | Distances and angles recomputed from the final coordinates |
| `negative_frequencies` | All final frequencies below zero in cm⁻¹ |
| `source_sha256` | Checksum of the merged source parquet |

## Promote Reviewed Candidates

Promotion is intentionally a separate code-review step:

1. Review the JSON values, source paths, checksums, frequency counts, and mode
   decisions.
2. Add a profile module such as
   `frust/tsguess2/profiles/r2scan3c_gas.py`.
3. Convert each approved JSON entry into a `StateGeometrySpec` with a
   `ReferenceRecord`.
4. Register the module in `frust/tsguess2/profiles/__init__.py`.
5. Mark only approved production geometries `active`.
6. Verify exact resolution and generated core geometry.

A profile module starts with the method/environment key:

```python
from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec


GEOMETRY_KEY = GeometryKey(method="r2scan-3c")

GEOMETRIES = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={...},
        constraint_values={...},
        reference=ReferenceRecord(...),
        status="active",
    ),
}
```

Revisions are immutable provenance identifiers. If a reference geometry is
replaced, add a new revision instead of silently changing the meaning of an
existing `ts_spec_id`.

## Verify Selection

After registration, inspect the public catalog:

```python
import frust as ft

profiles = ft.show_spec_profiles()
profiles.loc[
    profiles["profile"].eq("r2scan-3c/gas"),
    ["state", "status", "spec_id", "negative_frequencies", "mode_reviewed"],
]
```

Then verify that exact resolution no longer uses an environment fallback:

```python
from frust.tsguess2 import resolve_profile_spec

selection = resolve_profile_spec(
    "TS3",
    "r2scan-3c/gas",
    match="exact",
)

selection.match
```

Expected output:

```text
'exact'
```

Run the focused fast tests after changing a profile:

```bash
conda run -n UMA python -m pytest \
  tests/test_method_aware_ts_specs.py \
  tests/test_workflow_methods.py
```

These tests do not replace visual inspection of the reference modes or the
generated TS guesses.
