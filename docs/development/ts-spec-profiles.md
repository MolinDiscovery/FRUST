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

## The Six-Step Workflow

Use this sequence for every new or replacement profile:

| Step | Action | Required result |
| ---: | --- | --- |
| 1 | Validate the final workflow artifacts | One successful row per required state, with final coordinates and frequencies |
| 2 | Review frequencies, structures, and every TS imaginary mode | Exactly one correct reactive mode per TS; no imaginary modes for constrained minima |
| 3 | Run and review the extractor | Deterministic JSON with approved states marked `candidate` |
| 4 | Convert approved candidates into a profile module | One `StateGeometrySpec` per approved state with complete provenance |
| 5 | Register the profile | The profile is visible to the resolver without changing another method family |
| 6 | Verify resolution, constraints, embedding, tests, and docs | Exact selection and internally consistent generated rows |

Do not combine extraction and activation into one unreviewed operation. The JSON
between steps 3 and 4 is the scientific review boundary.

## Step 1: Validate The Final Workflow Artifacts

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

The filenames are not part of the extractor contract. Copied artifacts such as
`r2scan3c_specs_TSs_nosolv.parquet` and
`r2scan3c_specs_int_nosolv.parquet` work equally well; substitute their paths
in the commands below. The important requirement is one final row per state.

First inspect both collection reports:

```bash
jq '{n_targets, n_collected, n_failures, n_missing, n_errored}' \
  runs/21_r2scan3c_specs_TSs_nosolv/collection_report.json

jq '{n_targets, n_collected, n_failures, n_missing, n_errored}' \
  runs/21_r2scan3c_specs_INT3_nosolv/collection_report.json
```

For a complete TS1--TS4 plus INT3 reference set, the first report should have
four collected targets and the second should have one. Both reports should have
zero failures, missing targets, and errors.

Then validate the final dataframe rows:

```python
from pathlib import Path

import numpy as np
import pandas as pd


ts_path = Path("runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet")
int3_path = Path("runs/21_r2scan3c_specs_INT3_nosolv/merged.parquet")

ts = pd.read_parquet(ts_path)
int3 = pd.read_parquet(int3_path)

assert set(ts["state_id"]) == {"TS1", "TS2", "TS3", "TS4"}
assert list(int3["state_id"]) == ["INT3"]
assert ts["state_id"].is_unique
assert int3["state_id"].is_unique

assert ts["dft_ts_opt-NT"].eq(True).all()
assert ts["dft_freq-NT"].eq(True).all()
assert int3["dft_opt-NT"].eq(True).all()
assert int3["dft_freq-NT"].eq(True).all()

for _, row in ts.iterrows():
    assert np.isfinite(np.asarray(row["dft_ts_opt-oc"], dtype=float)).all()
for _, row in int3.iterrows():
    assert np.isfinite(np.asarray(row["dft_opt-oc"], dtype=float)).all()
```

The extractor reads these final columns:

| State | Coordinates | Vibrations |
| --- | --- | --- |
| `TS1`--`TS4` | `dft_ts_opt-oc` | `dft_freq-vibs` |
| `INT3` | `dft_opt-oc` | `dft_freq-vibs` |

The final solvent single-point stage does not change geometry. A gas-phase
result may contain `dft_solv_sp-*` columns, but the gas profile still comes from
the gas-phase optimization coordinates above.

!!! warning "Do not copy `constraint_spec` from the source row"

    `constraint_spec` records the profile used to build and preoptimize the
    source guess. It may belong to an older method. The extractor instead maps
    final atoms through `constraint_roles` and recalculates every distance and
    angle defined by `CORE_TOPOLOGIES` from the final optimized coordinates.

Stop here if a state is missing, duplicated, non-terminating, or lacks final
coordinates. Do not fill missing states from a different calculation without
making that mixed provenance explicit.

## Step 2: Review Frequencies, Structures, And Modes

Summarize the final frequencies:

```python
import frust as ft


ft.summarize_ts_vibrations(
    ts,
    col="dft_freq-vibs",
    show_pos_freqs=False,
)
```

The numerical acceptance rule is:

| State kind | Required negative frequencies |
| --- | ---: |
| Transition state | Exactly one |
| Constrained minimum such as `INT3` | Zero |

Frequency count is only the first check. Inspect the final optimized structures
and animate the imaginary mode of **every** TS:

```python
ft.plot_mols(
    ts,
    include_coords=["dft_ts_opt-oc"],
    coord_indices=[0],
    linked=False,
)
```

```python
for state in ("TS1", "TS2", "TS3", "TS4"):
    row_index = ts.index[ts["state_id"].eq(state)][0]
    vibrations = ts.at[row_index, "dft_freq-vibs"]
    negative_mode_ids = [
        mode_id
        for mode_id, mode in enumerate(vibrations)
        if float(mode["frequency"]) < 0
    ]
    assert len(negative_mode_ids) == 1
    print(state)
    ft.plot_vibs(
        ts,
        row_index=row_index,
        vId=negative_mode_ids[0],
        linked=False,
    )
```

Do not assume that the imaginary vibration is always stored as mode zero. The
loop finds its actual position in each row before opening the animation. If a
notebook frontend only renders the final viewer from a loop, run one state per
cell using the same `negative_mode_ids[0]` selection.

Check the constrained minimum independently:

```python
int3_vibrations = int3.iloc[0]["dft_freq-vibs"]
int3_negative = [
    float(mode["frequency"])
    for mode in int3_vibrations
    if float(mode["frequency"]) < 0
]
assert int3_negative == []
```

Use the role map to judge the motion:

| State | Reactive roles that should participate visibly |
| --- | --- |
| `TS1` | `cat_N`, `transfer_H`, `substrate_C`, with the catalyst core remaining chemically sensible |
| `TS2` | `cat_B`, `B_transfer_H`, `cat_N`, `N_transfer_H` |
| `TS3` | `cat_B`, `transfer_H`, `pin_B`, `substrate_C`; the mode should follow the intended TS3 core rather than peripheral catalyst motion |
| `TS4` | `cat_B`, `transfer_H`, `pin_B`, `substrate_C`, with motion appropriate to the distinct TS4 geometry |

Also inspect the final `INT3` structure and confirm that the lowest positive
modes are not symptoms of a malformed structure.

!!! warning "Mode review is a scientific approval"

    `--mode-reviewed TS3` means a person checked that TS3's imaginary mode is
    the intended reaction coordinate. It does not mean merely that TS3 has one
    negative frequency.

Record the decision outside the animation, for example:

| State | Frequency count | Mode/structure reviewed | Decision |
| --- | --- | --- | --- |
| TS1 | one negative | yes | approve or reject |
| TS2 | one negative | yes | approve or reject |
| TS3 | one negative | yes | approve or reject |
| TS4 | one negative | yes | approve or reject |
| INT3 | no negatives | structure reviewed | approve or reject |

Only approved TS states should be passed to `--mode-reviewed`.

## Step 3: Extract And Review Candidate JSON

### 3.1 Run The Extractor Before Recording Review

From the repository root, create the output directory and run the extractor
without `--mode-reviewed` first:

```bash
mkdir -p profile_candidates

conda run -n UMA python scripts/extract_tsguess2_profile.py \
  runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet \
  runs/21_r2scan3c_specs_INT3_nosolv/merged.parquet \
  --method r2scan-3c \
  --output profile_candidates/r2scan3c_gas.json
```

Omitting `--basis`, `--solvation-model`, and `--solvent` produces the profile
identifier `r2scan-3c/gas`. Composite methods such as r2SCAN-3c do not have a
separate basis value.

Before review, TS entries should be `quarantined`. This confirms that the
extractor is not silently treating frequency count as mode approval.

### 3.2 Inspect The Audit Summary

```bash
jq '{
  profile,
  states: (.candidates | to_entries | map({
    state: .key,
    status: .value.status,
    negative_frequencies: .value.reference.negative_frequencies,
    mode_reviewed: .value.reference.mode_reviewed,
    source_sha256: .value.reference.source_sha256
  }))
}' profile_candidates/r2scan3c_gas.json
```

Check all of the following:

- the profile identifier names the intended method and environment;
- every expected state appears exactly once;
- source paths point to the reviewed parquets;
- all TSs have exactly one expected negative frequency;
- `INT3` has none;
- TS mode-review flags still reflect the decision from step 2;
- TS rows share the checksum of the TS source, while INT3 records its own
  source checksum.

The candidate also contains:

| Field | Meaning |
| --- | --- |
| `role_coordinates` | Final Cartesian coordinates for topology-required roles |
| `constraint_values` | Distances and angles recomputed from those final coordinates |
| `frequency_count_valid` | Whether the TS/minimum has the expected number of negative modes |
| `source_sha256` | Checksum of the exact merged source parquet |

Inspect one complete candidate rather than only the summary:

```bash
jq '.candidates.TS3' profile_candidates/r2scan3c_gas.json
```

### 3.3 Record The Completed Mode Review

After step 2 is complete, rerun with only the approved TS states:

```bash
conda run -n UMA python scripts/extract_tsguess2_profile.py \
  runs/21_r2scan3c_specs_TSs_nosolv/merged.parquet \
  runs/21_r2scan3c_specs_INT3_nosolv/merged.parquet \
  --method r2scan-3c \
  --mode-reviewed TS1 TS2 TS3 TS4 \
  --output profile_candidates/r2scan3c_gas.json
```

Approved states with valid frequency counts are now `candidate`. The extractor
never emits `active`; activation happens in reviewed source code in step 4.

Run the same extraction twice and compare the files if determinism is in doubt:

```bash
cp profile_candidates/r2scan3c_gas.json /tmp/r2scan3c_gas.first.json

# Run the same extractor command again, then compare:
diff -u \
  /tmp/r2scan3c_gas.first.json \
  profile_candidates/r2scan3c_gas.json
```

No diff is expected.

### Solvated Profiles

Provide both solvation fields for a solvated reference:

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

`GeometryKey` rejects a solvation model without a solvent, or a solvent without
a solvation model.

## Step 4: Write The Profile Module

Create one module per method/environment combination. For the running example:

```text
frust/tsguess2/profiles/r2scan3c_gas.py
```

### 4.1 Map JSON Fields To Python Models

| Candidate JSON | Profile module |
| --- | --- |
| top-level `profile` | `GEOMETRY_KEY.profile_id` |
| `role_coordinates` | `StateGeometrySpec.role_coordinates` |
| `constraint_values` | `StateGeometrySpec.constraint_values` |
| `reference.negative_frequencies` | `ReferenceRecord.negative_frequencies` as a tuple |
| `reference.mode_reviewed` | `ReferenceRecord.mode_reviewed` |
| `reference.source_sha256` | Source checksum constant used by `ReferenceRecord` |
| approved `candidate` status | explicit `status="active"` after review |

JSON coordinate lists become Python tuples. Keep all extracted role and
constraint names unchanged; they must match `CORE_TOPOLOGIES` exactly.

### 4.2 Add Provenance Helpers

Start the new module with the method/environment and source checksums:

```python
"""Built-in gas-phase r2SCAN-3c ``tsguess2`` geometries."""

from __future__ import annotations

from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec


GEOMETRY_KEY = GeometryKey(method="r2scan-3c")
TS_SOURCE_SHA256 = "<reference.source_sha256 from a TS candidate>"
INT3_SOURCE_SHA256 = "<reference.source_sha256 from the INT3 candidate>"


def _reference(
    *,
    state: str,
    negative_frequencies: tuple[float, ...],
) -> ReferenceRecord:
    """Return provenance for one reviewed gas-phase reference row."""
    return ReferenceRecord(
        substrate_name="1-methylpyrrole",
        catalyst_name="NMe",
        method="r2SCAN-3c",
        basis=None,
        solvation_model=None,
        solvent=None,
        coordinates_column=(
            "dft_ts_opt-oc" if state.startswith("TS") else "dft_opt-oc"
        ),
        vibrations_column="dft_freq-vibs",
        negative_frequencies=negative_frequencies,
        mode_reviewed=True,
        source_sha256=(
            TS_SOURCE_SHA256 if state.startswith("TS") else INT3_SOURCE_SHA256
        ),
        notes="Extracted from reviewed gas-phase r2SCAN-3c reference calculations.",
    )
```

For a solvated profile, construct `GeometryKey` and `ReferenceRecord` with the
same reviewed environment:

```python
GEOMETRY_KEY = GeometryKey(
    method="r2scan-3c",
    solvation_model="smd",
    solvent="chloroform",
)
```

### 4.3 Add Every Approved State

Copy the values from each JSON candidate into one `StateGeometrySpec`. This is
a complete example for one TS1 candidate; use the values from your own JSON:

```python
GEOMETRIES: dict[str, StateGeometrySpec] = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.210596, 1.289281, -1.309434),
            "cat_N": (-0.720317, -1.294644, -0.349124),
            "substrate_C": (1.456758, 0.206029, -0.914079),
            "transfer_H": (0.539469, -0.625080, -0.660208),
        },
        constraint_values={
            "catN_transferH": 1.4601883341,
            "transferH_substrateC": 1.2635710364,
            "catB_substrateC": 1.6978398634,
            "catB_catN": 2.9095468230,
        },
        reference=_reference(
            state="TS1",
            negative_frequencies=(-205.46,),
        ),
        status="active",
    ),
}
```

Repeat the block for every approved state. A complete TS1--TS4 plus INT3 module
contains five dictionary entries.

Use `revision=1` for the first geometry in a state/profile slot. When replacing
an existing reference, increment its revision so the new values receive a new
`ts_spec_id`; never change numerical values while retaining the old revision.

Before registration, confirm that the module imports and every topology can be
combined:

```bash
conda run -n UMA python -c \
  'from frust.tsguess2.profiles.r2scan3c_gas import GEOMETRIES; print(sorted(GEOMETRIES))'
```

Expected output for the complete example:

```text
['INT3', 'TS1', 'TS2', 'TS3', 'TS4']
```

## Step 5: Register The Profile

Open `frust/tsguess2/profiles/__init__.py` and make three changes.

### 5.1 Import The New Module

```python
from frust.tsguess2.profiles.r2scan3c_gas import (
    GEOMETRIES as R2SCAN3C_GAS_GEOMETRIES,
    GEOMETRY_KEY as R2SCAN3C_GAS_KEY,
)
```

### 5.2 Expose Its Geometry Key

Include the imported key in `PROFILE_KEYS`:

```python
PROFILE_KEYS: dict[str, GeometryKey] = {
    key.profile_id: key
    for key in (
        WB97_GAS_KEY,
        WB97_SMD_KEY,
        R2SCAN3C_GAS_KEY,
        R2SCAN3C_SMD_KEY,
    )
}
```

If the registry previously declared a placeholder such as
`R2SCAN3C_GAS_KEY = GeometryKey(...)`, remove the placeholder and use the key
imported from the profile module. There should be one source of truth.

### 5.3 Register Its Geometries

```python
_GEOMETRIES: dict[str, dict[str, StateGeometrySpec]] = {
    WB97_GAS_KEY.profile_id: WB97_GAS_GEOMETRIES,
    R2SCAN3C_GAS_KEY.profile_id: R2SCAN3C_GAS_GEOMETRIES,
    R2SCAN3C_SMD_KEY.profile_id: R2SCAN3C_SMD_GEOMETRIES,
}
```

Do not remove quarantined entries for another environment. For example, adding
a reviewed gas TS3 does not make an invalid SMD TS3 active. Under
`prefer-exact`, the SMD request may fall back to the reviewed gas geometry;
under `exact`, it must still stop.

At this point the profile becomes selectable. Continue immediately to step 6;
successful import alone is not sufficient validation.

## Step 6: Verify The Registered Profile

### 6.1 Inspect The Public Catalog

```python
import frust as ft


profiles = ft.show_spec_profiles()
gas = profiles.loc[
    profiles["profile"].eq("r2scan-3c/gas"),
    [
        "state",
        "status",
        "spec_id",
        "negative_frequencies",
        "mode_reviewed",
        "source_sha256",
    ],
]
gas
```

Confirm that every approved state is `active`, has the expected revision,
frequency tuple, review flag, and source checksum.

### 6.2 Require Exact Resolution

```python
from frust.tsguess2 import resolve_profile_spec


for state in ("TS1", "TS2", "TS3", "TS4", "INT3"):
    selection = resolve_profile_spec(
        state,
        "r2scan-3c/gas",
        match="exact",
    )
    assert selection.match == "exact"
    assert selection.resolved_profile == "r2scan-3c/gas"
    assert selection.spec.spec_id.endswith("::r1")
```

If this fails, the profile is incomplete, inactive, or not registered under the
expected key.

### 6.3 Verify Distances And Angles Against Coordinates

The profile regression test must independently recalculate each constraint from
`role_coordinates` and compare it with the stored value. The project test for
this is `test_r2scan_gas_constraints_match_stored_role_coordinates` in
`tests/test_method_aware_ts_specs.py`.

Run it directly:

```bash
conda run -n UMA python -m pytest \
  tests/test_method_aware_ts_specs.py::test_r2scan_gas_constraints_match_stored_role_coordinates
```

This catches transcription errors when converting JSON arrays and numbers into
the Python profile module.

### 6.4 Generate And Inspect One Guess Per State

```python
import frust as ft


components = ft.screen.read("docs/examples/screen.csv")
systems = ft.screen.expand(components)
guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
    spec_profile="r2scan-3c/gas",
    spec_match="exact",
)

for state, frame in guesses.items():
    assert frame["ts_spec_id"].str.contains("r2scan-3c::gas").all()
    print(state, frame.iloc[0]["ts_spec_id"])
    ft.plot_row(frame, 0)
```

Inspect the reactive core and `ts_core_metrics`. Exact profile resolution proves
which numbers were selected; it does not prove that RDKit embedded a chemically
sensible guess.

### 6.5 Verify Workflow Selection And Run Fast Tests

```python
workflow = ft.workflows.screen_ts(
    dataframe=components,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method="r2scan-3c",
    spec_match="exact",
)

assert workflow.resolved_spec_profile == "r2scan-3c/gas"
assert len(workflow.targets()) > 0
```

Run the focused fast suite:

```bash
conda run -n UMA python -m pytest \
  tests/test_method_aware_ts_specs.py \
  tests/test_workflow_methods.py \
  tests/test_cluster_screen_chain.py
```

Then run the complete default suite and strict documentation build:

```bash
conda run -n UMA python -m pytest
conda run -n UMA mkdocs build --strict
git diff --check
```

The default test configuration excludes tests marked `slow`. Run slow geometry
tests separately only when the change or release process requires them.

## Completion Checklist

A profile is ready only when every item is true:

- [ ] Collection reports are complete and error-free.
- [ ] Final optimization and frequency stages terminated normally.
- [ ] Every TS has exactly one visually confirmed reactive imaginary mode.
- [ ] Every constrained minimum has no imaginary frequencies and a sensible
      final structure.
- [ ] The reviewed extractor output is deterministic and every promoted state
      is `candidate`.
- [ ] Profile coordinates, constraints, frequencies, review flags, and hashes
      match the candidate JSON.
- [ ] The profile module uses a new immutable revision when replacing values.
- [ ] The registry exposes the profile and retains unrelated quarantines.
- [ ] Exact resolution succeeds for every approved state.
- [ ] Distances and angles reproduce the stored role coordinates.
- [ ] At least one generated guess per state has been visually inspected.
- [ ] Focused tests, the complete default suite, documentation build, and diff
      checks pass.

These checks serve different purposes. Automated tests protect data flow and
geometry consistency; they do not replace scientific review of the reference
structures and vibrational modes.
