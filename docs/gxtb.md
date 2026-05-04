# g-xTB With FRUST

FRUST exposes g-xTB v2 through `Stepper.gxtb(...)`. This is a dedicated path
separate from `Stepper.xtb(...)`: FRUST calls Tooltoad's `gxtb_calculate(...)`,
and Tooltoad runs the special g-xTB `xtb` binary with `--gxtb` added
automatically.

## Setup

Set `GXTB_EXE` to the g-xTB v2 `xtb` executable. Do not point it at the normal
`XTB_EXE` unless that binary advertises `--gxtb`.

```bash
export GXTB_EXE=/Users/jacobmolinnielsen/Library/g-xtb/xtb-6.7.1-gxtb-210426-macos-arm64/bin/xtb
```

The local setup also stores this in `~/.env`, which Tooltoad loads when it
starts:

```bash
GXTB_ROOT=/Users/jacobmolinnielsen/Library/g-xtb/xtb-6.7.1-gxtb-210426-macos-arm64
GXTB_EXE=/Users/jacobmolinnielsen/Library/g-xtb/xtb-6.7.1-gxtb-210426-macos-arm64/bin/xtb
```

Quick check:

```bash
"$GXTB_EXE" --help | grep -- --gxtb
"$GXTB_EXE" --help | grep -- --grad
```

## Basic API

Use `Stepper.gxtb(...)` on a normal FRUST dataframe with `atoms` and a coordinate
column such as `coords_embedded` or a prior `*-oc` column.

```python
from frust.stepper import Stepper

step = Stepper(
    step_type="MOLS",
    n_cores=8,
    save_output_dir=False,
)

df_gxtb = step.gxtb(
    df,
    options={"grad": None},
    n_cores=8,
)
```

This runs the equivalent of:

```bash
xtb mol.xyz --gxtb --grad
```

`Stepper.gxtb(...)` takes the shared xTB-style arguments:

```python
step.gxtb(
    df,
    name="gxtb",
    options=None,
    detailed_inp_str="",
    constraint=False,
    save_step=False,
    lowest=None,
    n_cores=None,
)
```

## Common Calculations

Single point:

```python
df_sp = step.gxtb(df)
```

Gradient:

```python
df_grad = step.gxtb(
    df,
    options={"grad": None},
)
```

Geometry optimization:

```python
df_opt = step.gxtb(
    df,
    options={"opt": None},
)
```

Numerical Hessian on the input geometry:

```python
df_hess = step.gxtb(
    df,
    options={"hess": None},
)
```

Optimize, then Hessian:

```python
df_ohess = step.gxtb(
    df,
    options={"ohess": None},
)
```

Transition-state optimization:

```python
df_ts = step.gxtb(
    df,
    options={"opt": "ts"},
)
```

Mode-following transition-state optimization:

```python
df_ts = step.gxtb(
    df,
    options={
        "opt": "ts",
        "modef": 1,
    },
)
```

Charge and spin are handled by Tooltoad's normal xTB contract. If the input row
or calling layer passes charge/multiplicity through the backend, Tooltoad maps
them to xTB's `--chrg` and `--uhf` arguments.

## Output Columns

FRUST maps Tooltoad result keys into stage-prefixed dataframe columns.

Default single-point prefix:

```text
gxtb-EE
gxtb-NT
```

Optimization prefix:

```text
gxtb-opt-EE
gxtb-opt-NT
gxtb-opt-oc
```

Custom names override the prefix:

```python
df_named = step.gxtb(
    df,
    name="gxtb_preopt",
    options={"opt": None},
)
```

Typical columns:

```text
gxtb_preopt-EE
gxtb_preopt-NT
gxtb_preopt-oc
```

Useful suffixes:

- `-EE`: electronic energy.
- `-NT`: normal termination boolean.
- `-oc`: optimized coordinates.
- `-GE`: Gibbs energy when thermochemistry is available.
- `-vibs`: vibrations when Hessian output is parsed.
- `-error`: row-level exception text if the backend fails.

## Chaining

Use g-xTB exactly like the other FRUST stepper stages. FRUST will pick the most
recent coordinate column when moving to the next stage.

```python
step = Stepper(step_type="MOLS", n_cores=8, save_output_dir=False)

df = step.gxtb(
    df,
    name="gxtb_preopt",
    options={"opt": None},
)

df = step.orca(
    df,
    name="hf_sp",
    options={"HF": None, "STO-3G": None, "SP": None},
)
```

To keep only the lowest-energy conformers per structure group:

```python
df_low = step.gxtb(
    df,
    options={"opt": None},
    lowest=5,
)
```

`lowest=` uses the same shared FRUST behavior as `Stepper.xtb(...)` and
`Stepper.orca(...)`.

## Constraints

FRUST's built-in constrained xTB setup is also available for g-xTB. This is
mainly useful for the existing FRUST step types such as `TS1`, `TS2`, `TS3`,
`TS4`, and `INT3`.

```python
step = Stepper(step_type="TS1", n_cores=8)

df_ts1 = step.gxtb(
    df,
    options={"opt": "ts"},
    constraint=True,
)
```

The dataframe must contain the constraint metadata expected by the step type,
typically `constraint_atoms`.

For custom xTB control input, pass `detailed_inp_str`:

```python
df_custom = step.gxtb(
    df,
    options={"grad": None},
    detailed_inp_str="""
$constrain
force constant=50
distance: 1, 2, 1.50
$end
""",
)
```

FRUST appends its generated constraint block after the custom input when both
`detailed_inp_str` and `constraint=True` are used.

## Saving Calculation Files

Use `save_step=True` when you want to keep calculation directories.

```python
step = Stepper(
    step_type="MOLS",
    n_cores=8,
    save_output_dir="FRUST_results",
)

df_saved = step.gxtb(
    df,
    options={"opt": None},
    save_step=True,
)
```

The saved directory can include files such as:

```text
mol.xyz
xtbopt.xyz
xtbopt.log
charges
wbo
xtbtopo.mol
```

The exact file set depends on the options used and what the g-xTB binary writes.

## Direct Tooltoad Use

FRUST uses Tooltoad internally, but the calculator can also be called directly:

```python
from tooltoad.gxtb import gxtb_calculate

result = gxtb_calculate(
    atoms=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    options={"grad": None},
    n_cores=1,
)

print(result["normal_termination"])
print(result["electronic_energy"])
print(result["grad"].shape)
```

## Notes And Limitations

- `GXTB_EXE` is required only when `gxtb_calculate(...)` is called.
- FRUST does not automatically reuse `XTB_EXE`; the normal local xTB binary may
  not support `--gxtb`.
- Tooltoad always adds `--gxtb`, so do not include `"gxtb": None` yourself.
- TS optimization is requested as `options={"opt": "ts"}`, not as a separate
  `optts` option.
- The installed upstream README notes that not all xTB features are supported by
  g-xTB yet.
- The macOS g-xTB README warns about parallel numerical Hessians. Prefer
  `n_cores=1` for cautious local Hessian checks on macOS.

## Troubleshooting

Check that Tooltoad sees the executable:

```bash
python - <<'PY'
from tooltoad.gxtb import _resolve_gxtb_cmd
print(_resolve_gxtb_cmd())
PY
```

Check that the executable is actually g-xTB-capable:

```bash
"$GXTB_EXE" --help | grep -- --gxtb
```

If FRUST returns `gxtb-NT=False`, inspect the `gxtb-error` column first. If the
backend ran but xTB itself failed, rerun with `save_step=True` and inspect the
saved xTB output files.
