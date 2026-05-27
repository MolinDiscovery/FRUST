# Catalyst And Substrate Screens

The catalyst-screen workflow starts from a CSV that contains substrates and
catalysts. FRUST expands that component table into explicit substrate-catalyst
systems, then constructs transition-state guess dataframes for `TS1`, `TS2`,
`TS3`, and `TS4`.

The whole workflow is:

```python
import frust as ft

components = ft.screen.read("screen.csv")
systems = ft.screen.expand(components)
ts_guesses = ft.screen.create_ts_guesses(
    systems,
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    n_confs=1,
)
```

The important mental model is:

```text
component rows
    -> substrate-catalyst systems
    -> one row per TS type, reactive position, and conformer
    -> row-level constrained xTB/DFT calculations
```

FRUST does not infer an entire reaction mechanism from arbitrary SMILES. It
instantiates a small set of built-in transition-state motifs that are currently
based on the TMP/methylpyrrole template family. The new screen machinery makes
those motifs usable with different substrates and closely related catalysts.

## Start From A Component CSV

Create one CSV that lists both substrates and catalysts:

```csv
role,smiles,compound_name,rpos,series
substrate,CN1C=CC=C1,n_methyl_pyrrole,,pyrrole
substrate,COC1=CC=CO1,methoxyfuran,"3,5",furan
catalyst,CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B,tmp_bcat,,baseline
```

Read it with `ft.screen.read(...)`:

```python
components = ft.screen.read("screen.csv")
components
```

Output:

| role | smiles | compound_name | rpos | series |
| --- | --- | --- | --- | --- |
| substrate | `CN1C=CC=C1` | `n_methyl_pyrrole` |  | `pyrrole` |
| substrate | `COC1=CC=CO1` | `methoxyfuran` | `3,5` | `furan` |
| catalyst | `CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B` | `tmp_bcat` |  | `baseline` |

Only `role` and `smiles` are required. `compound_name` gives stable readable
names. If it is omitted, FRUST creates names such as `substrate_000` and
`catalyst_000`.

`role` accepts `substrate`, `sub`, `catalyst`, and `cat`.

!!! note "Catalyst rows do not use `rpos`"

    `rpos` belongs to substrate rows. In non-strict mode, catalyst `rpos`
    values are ignored with a warning. Use
    `ft.screen.read("screen.csv", strict=True)` when accidental catalyst
    `rpos` entries should fail.

### Choose Reactive Positions

`rpos` is the RDKit atom index of an aromatic C-H atom in the substrate. If
`rpos` is blank, FRUST uses the symmetry-unique aromatic C-H positions it can
find for that substrate.

For the example above:

| substrate | `rpos` input | generated `rpos` values |
| --- | --- | --- |
| `n_methyl_pyrrole` | blank | `2`, `3` |
| `methoxyfuran` | `3,5` | `3`, `5` |

Use a drawing before choosing manual `rpos` values:

```python
ft.DrawUniqueChGrid(
    ["CN1C=CC=C1", "COC1=CC=CO1"],
)
```

The drawing labels the relevant aromatic C-H atom indices. This matters because
small SMILES changes can change atom numbering.

!!! warning "Validate `rpos` from the exact SMILES"

    `COC1=CC=CO1` has aromatic C-H positions `(3, 4, 5)` in this SMILES.
    A value such as `2` is invalid even if it looks chemically reasonable in a
    hand drawing, because `rpos` follows the parsed atom indices.

## Expand Components Into Systems

`ft.screen.expand(...)` creates explicit substrate-catalyst pairs:

```python
systems = ft.screen.expand(components)
systems[
    [
        "system_name",
        "substrate_name",
        "catalyst_name",
        "rpos",
        "substrate_series",
        "catalyst_series",
    ]
]
```

Output:

| system_name | substrate_name | catalyst_name | rpos | substrate_series | catalyst_series |
| --- | --- | --- | --- | --- | --- |
| `n_methyl_pyrrole__tmp_bcat` | `n_methyl_pyrrole` | `tmp_bcat` |  | `pyrrole` | `baseline` |
| `methoxyfuran__tmp_bcat` | `methoxyfuran` | `tmp_bcat` | `3,5` | `furan` | `baseline` |

The expansion is deliberately explicit. If a screen has 11 substrates and 3
catalysts, this step gives 33 substrate-catalyst systems before `rpos`,
TS-type, and conformer expansion.

Extra metadata columns are preserved with prefixes. In the table above,
`series` became `substrate_series` and `catalyst_series`.

## Build TS Guess Dataframes

Generate TS guesses from the expanded systems:

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

Each TS family is returned as its own dataframe:

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

| system_name | structure_type | rpos | cid | ts_spec_id |
| --- | --- | ---: | ---: | --- |
| `n_methyl_pyrrole__tmp_bcat` | `TS1` | 2 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `n_methyl_pyrrole__tmp_bcat` | `TS1` | 3 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `methoxyfuran__tmp_bcat` | `TS1` | 3 | 0 | `TS1::builtin::methylpyrrole_v1` |
| `methoxyfuran__tmp_bcat` | `TS1` | 5 | 0 | `TS1::builtin::methylpyrrole_v1` |

The full row also contains readable names such as:

```text
TS1(n_methyl_pyrrole__tmp_bcat_rpos(2))
TS1(methoxyfuran__tmp_bcat_rpos(5))
```

The same four `rpos` rows are generated for `TS2`, `TS3`, and `TS4`.

The row count is:

```text
rows per TS dataframe = systems x reactive positions x conformers
```

In this example, one catalyst and two substrates become two systems. The first
substrate contributes two symmetry-unique positions, and the second contributes
two explicit positions, so each TS dataframe has four rows.

## What The Chemistry Model Contains

The screen module only decides what combinations should exist. The chemistry is
constructed by the TS guess machinery.

Each built-in TS template is described in terms of named chemical roles rather
than fixed atom indices. The most important roles are:

| role | Meaning |
| --- | --- |
| `cat_B` | The boron atom in the catalyst B-aryl-N scaffold |
| `cat_N` | The nitrogen atom in the same catalyst scaffold |
| `substrate_C` | The substrate aromatic carbon selected by `rpos` |
| `transfer_H` | The hydrogen involved in the main TS hydrogen-transfer motif |
| `cat_H` | A retained catalyst B-H hydride used in TS2-TS4 |
| `n_transfer_H` | The H atom placed toward catalyst N in the TS2 H-H motif |
| `pin_B` | The boron atom in the built-in HBpin fragment for TS3/TS4 |

This role language is the key abstraction. The template says where `cat_B`,
`transfer_H`, `substrate_C`, and the other roles should be in space. The code
then finds the corresponding atoms in the actual catalyst/substrate/fragments.

### TS1-TS4 At A Glance

- `TS1`: C-H activation motif from the methylpyrrole template.
  Main roles: `cat_B`, `cat_N`, `substrate_C`, `transfer_H`.
- `TS2`: H-H motif involving catalyst B/N and one retained B-H.
  Main roles: `cat_B`, `cat_N`, `cat_H`, `transfer_H`,
  `n_transfer_H`, `substrate_C`.
- `TS3`: HBpin-associated TMP motif where the HBpin B-H becomes
  `transfer_H`. Main roles: `cat_B`, `cat_N`, `cat_H`, `pin_B`,
  `transfer_H`, `substrate_C`.
- `TS4`: HBpin-associated TMP motif where a catalyst B-H becomes
  `transfer_H`. Main roles: `cat_B`, `cat_N`, `cat_H`, `transfer_H`,
  `pin_B`, `substrate_C`.

These are transition-state guesses, not optimized transition states. They are
intended to put the chemically important atoms close enough to the calibrated
template geometry that constrained low-level pre-optimization and later DFT
refinement can take over.

!!! note "TS3 and TS4 deliberately differ in hydride bookkeeping"

    In `TS3`, `transfer_H` is the B-H hydrogen on the HBpin fragment, and one
    catalyst B-H is retained as `cat_H`.

    In `TS4`, the HBpin B-H is removed, two catalyst B-H atoms are retained,
    and one of them becomes `transfer_H`.

## How FRUST Constructs A TS Guess

For every system, `rpos`, TS type, and conformer, FRUST does the following:

1. Parse catalyst and substrate SMILES and add explicit hydrogens.
2. Find exactly one catalyst B-aryl-N scaffold and assign `cat_B` and `cat_N`.
3. Find the substrate atom selected by `rpos`, remove its attached H, and
   assign the adjusted atom index to `substrate_C`.
4. Add the TS-specific fragment: H for `TS1`, H2 for `TS2`, and HBpin for
   `TS3`/`TS4`.
5. Assign TS-specific hydride roles, such as `cat_H`, `transfer_H`,
   `n_transfer_H`, and `pin_B`.
6. Place role atoms on built-in template coordinates.
7. Add softer frame anchors for the neighboring substrate atoms, and for
   HBpin oxygen atoms in `TS3`/`TS4`.
8. Embed and rigidly place the disconnected fragments, snapping only the hard
   role atoms exactly to the template.
9. Rotate under-anchored fragments to reduce unexpected close contacts.
10. Store atoms, coordinates, role maps, constraints, diagnostics, and plotting
    connectivity in the dataframe row.

The practical result is that the catalyst and substrate can change while the
reactive core remains template-like.

### Hard Anchors And Soft Frame Anchors

Hard anchors are the role atoms that define the TS core. For example, `cat_B`,
`cat_N`, `transfer_H`, `pin_B`, and `substrate_C` are snapped to their role
coordinates when they are part of a TS type.

Soft frame anchors guide the orientation of larger fragments without pretending
that every atom in the fragment belongs to the reactive core:

- `substrate frame`: uses `substrate_C` plus its two heavy neighbors. This
  keeps the aromatic ring face and bond directions aligned with the template.
- `HBpin frame`: uses `pin_B` plus its two oxygen neighbors. This prevents
  arbitrary pinacol rotation in `TS3` and `TS4`.

This distinction is important. If only a single atom from a fragment were
anchored, RDKit could place the rest of that fragment in a chemically wrong
orientation. If too many atoms were force-snapped, the code would over-constrain
substrates and catalysts that are only similar to the template. The current
approach fixes the reactive role atoms and gently controls the surrounding
fragment frame.

## Inspect The Row-Level Constraint Model

Every generated TS row carries the information needed to run constrained
optimization.

```python
row = ts_guesses["TS3"].iloc[0]

row["constraint_roles"]
```

Representative output:

```python
{
    "cat_B": 16,
    "cat_N": 9,
    "substrate_C": 42,
    "cat_H": 39,
    "pin_B": 56,
    "transfer_H": 67,
}
```

`constraint_roles` is the bridge between chemistry and atom indices. It says
which atom in this particular assembled molecule plays each named role.

The actual constrained distances and angles are stored separately:

```python
row["constraint_spec"]
```

Representative output for `TS3`:

| kind | roles | value |
| --- | --- | ---: |
| distance | `['transfer_H', 'cat_B']` | 1.376 |
| distance | `['transfer_H', 'pin_B']` | 1.264 |
| distance | `['transfer_H', 'substrate_C']` | 2.477 |
| distance | `['cat_B', 'substrate_C']` | 1.616 |
| distance | `['pin_B', 'substrate_C']` | 2.180 |
| distance | `['pin_B', 'cat_B']` | 2.007 |
| angle | `['cat_B', 'transfer_H', 'pin_B']` | 98.89 |
| angle | `['cat_B', 'substrate_C', 'pin_B']` | 61.75 |

This separation is what lets the same template work across different atom
numbering, different substrates, and different catalysts.

!!! warning "Display bonds are not constraints"

    `connectivity_bonds` is stored so FRUST can draw the assembled TS guess with
    the intended covalent graph. It is not the source of constrained geometry.

    For example, `TS3` and `TS4` keep a `cat_B-pin_B` distance constraint, but
    they do not store a visual B-B bond. The constrained calculation still sees
    the B-B distance because it comes from `constraint_spec`.

## Inspect The Geometry Before Running Calculations

Always inspect a few rows before launching a large screen:

```python
ft.plot_row(ts_guesses["TS1"], 0)
ft.plot_row(ts_guesses["TS2"], 0)
ft.plot_row(ts_guesses["TS3"], 0)
ft.plot_row(ts_guesses["TS4"], 0)
```

For a full dataframe:

```python
ft.plot_mols(ts_guesses["TS4"], range(0, min(6, len(ts_guesses["TS4"]))))
```

Look for these chemistry features:

- `TS1`: the transferred H sits between substrate C and catalyst N, with
  catalyst B close to substrate C.
- `TS2`: the H-H motif is outside the catalyst/substrate framework, not buried
  in the molecule.
- `TS3`: the HBpin B-H, catalyst B, and substrate C form the expected compact
  TMP-like core.
- `TS4`: HBpin is placed against the substrate C, and the B-B contact is a
  constrained distance, not a visual bond.

The guesses do not need to look like final DFT structures, but the reactive
core should be recognizable. If the core is qualitatively wrong, do not start a
large calculation batch.

## Run Constrained Pre-Optimization

Use each grouped TS dataframe with `Stepper`:

```python
step = ft.Stepper(n_cores=8, save_output_dir=False)

ts4_preopt = step.xtb(
    ts_guesses["TS4"],
    name="xtb_preopt",
    options={"gfnff": None, "opt": None},
    constraint=True,
)
```

With screen-generated rows, `constraint=True` is row-first:

1. If the row has `constraint_roles` and `constraint_spec`, `Stepper` renders
   those role-based constraints.
2. If those columns are absent, `Stepper` falls back to the older
   `step_type + constraint_atoms` behavior.

This means variable-catalyst screens do not need `Stepper(step_type="TS4")`
just to know what TS4 constraints mean. The row carries its own constraint
model.

After pre-optimization, keep the lowest row per structure:

```python
ts4_lowest = ft.lowest_energy_rows(ts4_preopt)
ft.plot_mols(ts4_lowest, range(0, len(ts4_lowest)))
```

The optimized coordinate column will usually be named from the stage, for
example `xtb_preopt-oc`.

## Use Core Metrics To Judge Template Quality

Generated rows also include `ts_core_metrics`:

```python
row = ts_guesses["TS3"].iloc[0]
row["ts_core_metrics"][:4]
```

Representative output:

| kind | roles | reference | measured | delta |
| --- | --- | ---: | ---: | ---: |
| distance | `['transfer_H', 'cat_B']` | 1.376 | 1.382 | 0.006 |
| distance | `['transfer_H', 'pin_B']` | 1.264 | 1.265 | 0.001 |
| distance | `['transfer_H', 'substrate_C']` | 2.477 | 2.482 | 0.005 |
| distance | `['cat_B', 'substrate_C']` | 1.616 | 1.616 | 0.000 |

The `reference` value is the built-in template constraint. The `measured` value
is what the embedded guess actually has. The `delta` is the difference.

These metrics are useful at two stages:

- Immediately after embedding: catch construction failures or poor fragment
  placement before calculations.
- After DFT TS optimization: measure how far the optimized TS moved away from
  the template.

The second use is important for future calibrated-template work. If optimized
TS structures consistently stay close to the generated template for one
chemistry class, the built-in template is likely transferable. If a class gives
large shifts in key distances or angles, that class may need a new calibrated
template or an automatic full TS optimization route.

!!! note "Template quality is chemical, not just numerical"

    A distance change around `0.02 Angstrom` after high-level optimization
    usually suggests that the template captured the core well. A change around
    `0.2 Angstrom` in a key forming/breaking bond is a warning sign that the
    substrate or catalyst electronics may have changed the TS enough that the
    template is no longer a good guess.

## What Can Fail

Most failures should be treated as useful chemistry or input diagnostics rather
than nuisance errors.

- Invalid `rpos`: the requested atom is not an aromatic C-H in the parsed
  SMILES. Draw atom labels and use the exact SMILES from the CSV.
- No unique B-aryl-N scaffold: the catalyst is outside the current v1 scaffold
  model or is ambiguous. Inspect the catalyst SMILES and decide whether a new
  matcher is needed.
- Missing B-H hydrogens: a TS type requires catalyst hydrides that the catalyst
  cannot provide. Check catalyst valence/charge and whether the TS motif
  applies.
- Strange TS2 H-H placement: the H-H motif is underconstrained or clashes after
  embedding. Inspect `TS2` before optimization and add regression coverage if
  it recurs.
- TS3/TS4 HBpin orientation looks wrong: the HBpin frame or hydride assignment
  may not match the intended motif. Compare against TMP-like reference
  structures and `ts_core_metrics`.

## Current Scope And Extension Rules

The first screen implementation is intentionally conservative:

Supported now:

- Neutral B-aryl-N catalysts with one recognizable scaffold.
- Aromatic substrate C-H positions.
- TS1-TS4 built from built-in methylpyrrole/TMP template geometry.
- Row-level xTB/ORCA constraints from `constraint_spec`.

Not automatic yet:

- Arbitrary catalyst topologies.
- Non-aromatic C-H activation sites.
- Automatic template calibration from optimized TS structures.
- Mechanism discovery from SMILES alone.

When extending this workflow, preserve the role-based design:

- Add or modify templates by chemical roles, not fixed atom indices.
- Keep `constraint_roles` and `constraint_spec` as the source of optimizer
  constraints.
- Treat `connectivity_bonds` as display/storage connectivity only.
- Add regression tests for role assignment, hydride bookkeeping, frame
  orientation, expected stored bonds, and core metric deltas.

The goal is that FRUST can screen related catalysts and substrates while still
making it clear when the chemistry has moved outside the trusted template
space.
