# Catalyst And Substrate Screens

This workflow starts from one CSV that contains both substrates and catalysts.
FRUST turns it into explicit substrate-catalyst systems, then builds grouped TS
guess dataframes for `TS1`, `TS2`, `TS3`, and `TS4`.

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,2,pyrroles
substrate,COc1ccccc1,anisole,"2,4",anisoles
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

The mental model is:

```text
component rows -> substrate-catalyst systems -> grouped TS guess dataframes
```

## Read The Screen

```python
import frust as ft

components = ft.screen.read("screen.csv")
components
```

Output:

| role | smiles | compound_name | rpos | series |
| --- | --- | --- | --- | --- |
| substrate | `CN1C=CC=C1` | `n_methyl_pyrrole` | `2` | `pyrroles` |
| substrate | `COc1ccccc1` | `anisole` | `2,4` | `anisoles` |
| catalyst | `CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B` | `tmp_bcat` |  | `baseline` |

`role` accepts `substrate`, `sub`, `catalyst`, and `cat`. If
`compound_name` is missing, FRUST creates stable names such as
`substrate_000` and `catalyst_000`.

!!! note "Catalyst rows do not use `rpos`"

    `rpos` belongs to the substrate. In non-strict mode, catalyst `rpos` values
    are ignored with a warning. Use `ft.screen.read("screen.csv", strict=True)`
    when you want accidental catalyst `rpos` entries to fail.

## Expand Systems

```python
systems = ft.screen.expand(components)
systems[["system_name", "substrate_name", "catalyst_name", "rpos"]]
```

Output:

| system_name | substrate_name | catalyst_name | rpos |
| --- | --- | --- | --- |
| `n_methyl_pyrrole__tmp_bcat` | `n_methyl_pyrrole` | `tmp_bcat` | `2` |
| `anisole__tmp_bcat` | `anisole` | `tmp_bcat` | `2,4` |

The expanded table is deliberately inspectable. If a screen has 11 substrates
and 3 catalysts, this step produces 33 systems before reactive-position and
conformer expansion.

Extra columns are preserved with prefixes. In the example above, substrate
metadata becomes `substrate_series` and catalyst metadata becomes
`catalyst_series`.

## Build TS Guesses

```python
ts_guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
)

ts_guesses.keys()
```

Output:

```text
dict_keys(['TS1', 'TS2', 'TS3', 'TS4'])
```

Each TS type is returned as its own dataframe so constrained calculations do not
mix TS families:

```python
ts1 = ts_guesses["TS1"]
ts1[
    [
        "custom_name",
        "system_name",
        "structure_type",
        "rpos",
        "cid",
        "ts_spec_id",
    ]
]
```

Representative output:

| custom_name | system_name | structure_type | rpos | cid | ts_spec_id |
| --- | --- | --- | ---: | ---: | --- |
| `TS1(n_methyl_pyrrole__tmp_bcat_rpos(2))` | `n_methyl_pyrrole__tmp_bcat` | `TS1` | 2 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `TS1(anisole__tmp_bcat_rpos(2))` | `anisole__tmp_bcat` | `TS1` | 2 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `TS1(anisole__tmp_bcat_rpos(4))` | `anisole__tmp_bcat` | `TS1` | 4 | 0 | `TS1::builtin::methylpyrrole_v1` |

The generated dataframe carries the full TS-core constraint model:

```python
row = ts1.iloc[0]

row["constraint_roles"]
```

Representative output:

```python
{
    "cat_B": 16,
    "cat_N": 9,
    "substrate_C": 43,
    "transfer_H": 53,
}
```

```python
row["constraint_spec"][:2]
```

Representative output:

```python
[
    {"kind": "distance", "roles": ["cat_B", "transfer_H"], "value": 2.07696},
    {"kind": "distance", "roles": ["cat_N", "transfer_H"], "value": 1.51270},
]
```

`constraint_roles` tells FRUST which atom index plays each chemical role.
`constraint_spec` tells FRUST what distances and angles should be constrained.
This is the new row-level replacement for hard-coding TS geometry inside
`Stepper`.

Inspect one guess before launching calculations:

```python
ft.plot_row(ts_guesses["TS4"], 0)
```

The TS guess stores disconnected catalyst, substrate, and reagent fragments in
one coordinate set. FRUST places the reactive role atoms on the built-in TS
template, then rotates singly anchored fragments to reduce obvious
inter-fragment clashes while preserving the template core. The row also stores
`connectivity_bonds`, so FRUST visualizers use the assembled covalent graph
instead of guessing bonds from short TS-contact distances.

!!! warning "Current v1 chemistry scope"

    This first screen workflow supports neutral B-aryl-N catalysts and aromatic
    substrate C-H positions. If the catalyst has no unique B-aryl-N scaffold, TS
    generation fails instead of guessing.

## Run A Constrained Step

Use each grouped dataframe with `Stepper`:

```python
step = ft.Stepper(debug=True, save_output_dir=False)

ts1_screen = step.xtb(
    ts_guesses["TS1"],
    name="xtb_preopt",
    options={"gfnff": None, "opt": None},
    constraint=True,
)
```

With the new screen-generated rows, `constraint=True` is row-first:

1. If the row has `constraint_roles` and `constraint_spec`, `Stepper` renders
   those constraints.
2. Otherwise, `Stepper` falls back to the legacy `step_type + constraint_atoms`
   behavior used by older `create_ts_per_rpos(...)` workflows.

This means new variable-catalyst screens do not need `Stepper(step_type="TS1")`
just to know what the constraints mean. The dataframe row is self-describing.

## Inspect Core Geometry

Each generated row also stores role-based diagnostics:

```python
row["ts_core_metrics"][:2]
```

Representative output:

```python
[
    {
        "kind": "distance",
        "roles": ["cat_B", "transfer_H"],
        "reference": 2.07696,
        "measured": 2.07696,
        "delta": 0.0,
    },
    {
        "kind": "distance",
        "roles": ["cat_N", "transfer_H"],
        "reference": 1.5127,
        "measured": 1.5127,
        "delta": 0.0,
    },
]
```

These diagnostics are intended for future calibrated-template workflows. After a
full DFT TS optimization, FRUST can compare the optimized reactive-core
distances and angles against the generated TS guess and decide whether a
chemistry class needs a calibrated spec.
