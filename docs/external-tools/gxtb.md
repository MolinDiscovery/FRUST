# g-xTB With FRUST

FRUST exposes g-xTB v2 through `Stepper.gxtb(...)`. This is a dedicated path
separate from `Stepper.xtb(...)`: FRUST calls Tooltoad's `gxtb_calculate(...)`,
and Tooltoad runs the special g-xTB `xtb` binary with `--gxtb` added
automatically.

## Setup

Put `GXTB_EXE` in `~/.env` and point it to the g-xTB v2 `xtb` executable. Do
not point it at the normal `XTB_EXE` unless that binary advertises `--gxtb`.

Install or unpack the latest g-xTB v2 release from Grimme lab:

```text
https://github.com/grimme-lab/g-xtb/releases
```

!!! note
    g-xTB v2 is still a preliminary release and may receive frequent patch
    releases. Prefer a versioned install directory with a stable symlink, so
    `GXTB_EXE` does not need to change every time you update from, for example,
    `2.0.0` to `2.0.1`.

FRUST uses the newer g-xTB v2 `xtb` binary, not older g-xTB workflows. For
direct `Stepper.gxtb(...)`, Tooltoad calls this binary with `--gxtb`. For
ORCA-driven g-xTB, the MolinDiscovery OET fork provides `oet_gxtb`; ORCA calls
that OET wrapper, and the wrapper calls `GXTB_EXE` so g-xTB 2.0 can provide
external energies and gradients back to ORCA.

```bash
GXTB_ROOT=/path/to/g-xtb
GXTB_EXE=${GXTB_ROOT}/bin/xtb
```

See [External tool setup](../getting-started/external-tool-setup.md) for the
shared `.env` setup used by FRUST and Tooltoad, including a
[g-xTB update workflow](../getting-started/external-tool-setup.md#update-g-xtb).

Quick check:

```bash
"$GXTB_EXE" --help | grep -- --gxtb
"$GXTB_EXE" --help | grep -- --grad
```

## Basic API

Start by building a FRUST dataframe, then pass it to `Stepper.gxtb(...)`.
For a quick tool check, a single SMILES string is enough:

```python
from frust.stepper import Stepper

step = Stepper(
    n_cores=8,
    save_output_dir=False,
)

df = step.build_initial_df("CCO", name="ethanol")

df_gxtb = step.gxtb(
    df,
    options={"grad": None},
    n_cores=8,
)
```

For larger workflows, `Stepper.gxtb(...)` consumes the same dataframe shape:
`atoms` plus a coordinate column such as `coords_embedded` or a prior `*-oc`
column.

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

Transition-state optimization is not handled by direct `Stepper.gxtb(...)`.
Use ORCA-driven g-xTB instead, so ORCA owns the `OptTS` search and g-xTB only
provides energies and gradients. For difficult TS searches, first compute a
g-xTB Hessian directly and pass it into ORCA with `use_last_hess=True`:

```python
df = step.gxtb(
    df,
    name="gxtb-hess",
    options={"hess": None},
    save_step=True,
)

df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
    use_last_hess=True,
    save_step=True,
)
```

For a simpler first attempt, you can run ORCA `OptTS` without a supplied
Hessian and let ORCA build its own initial guess:

```python
df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
)
```

Use `NumFreq`, not `Freq`, when you want frequencies with ORCA-driven g-xTB.
`Freq` uses ORCA's analytic frequency machinery, which is not compatible with
the external g-xTB method. `NumFreq` uses finite differences of the external
gradients.

Mode-specific TS setup should also go through ORCA input blocks:

```python
df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None, "NumFreq": None},
    gxtb=True,
    xtra_inp_str="""
%geom
  TS_Mode {B 0 1} end
end
""",
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

## Workflow Method Plans

For new local-to-cluster workflows, put g-xTB into the method plan instead of
editing every calculation script by hand.

```python
import frust as ft

method = (
    ft.workflows.methods.preset("r2scan-3c")
    .replace(
        xtb_sp=ft.workflows.methods.gxtb(job="sp"),
        xtb_opt=ft.workflows.methods.gxtb(job="opt"),
    )
)

wf = ft.workflows.screen_ts(
    csv_path="screen.csv",
    ts_types=["TS1", "TS2", "TS3", "TS4"],
    method=method,
    n_confs=None,
    top_n=10,
    dft=True,
)
```

This changes only the selected xTB-like filtering stages. The ORCA DFT stages
still come from the `r2scan-3c` preset.

!!! note "g-xTB stages are not GFN2 stages"

    Use `ft.workflows.methods.gxtb(job="sp")` or `gxtb(job="opt")`. Do not
    write `engine="gxtb", options={"gfn": 2}`; `gfn=2` is a normal xTB option,
    not a g-xTB method selection.

## ORCA-Driven TS Optimization

For transition states, use ORCA's external optimization interface:

```python
step = Stepper(step_type="TS1", n_cores=8, save_output_dir=False)

df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
)
```

FRUST automatically inserts `ExtOpt` and an OET `%method` block pointing to
`oet_gxtb`. OET then calls the `GXTB_EXE` binary as `xtb --gxtb --grad` for
each ORCA gradient request. This is the correct route for `OptTS`, `NEB-TS`,
and other ORCA optimizer workflows.

You can check which g-xTB executable FRUST used from the step metadata:

```python
df_ts.attrs["frust_steps"]["gxtb-OptTS"]["calculator"]["executables"]["gxtb"]
```

Example:

```python
{
    "path": "/cluster/apps/g-xtb/bin/xtb",
    "configured": "/cluster/apps/g-xtb/bin/xtb",
    "source": "GXTB_EXE",
    "resolved": True,
}
```

`source` is `"GXTB_EXE"` when FRUST used the environment variable and
`"gxtb_exe"` when you passed `gxtb_exe=...` directly to `step.orca(...)`.
For compatibility, ORCA-driven g-xTB stages also keep the flat aliases
`gxtb_exe` and `gxtb_exe_source` on the stage metadata.

For TS searches that need a better starting Hessian, use a two-step workflow:

```python
df = step.gxtb(
    df,
    name="gxtb-hess",
    options={"hess": None},
    save_step=True,
)

df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None},
    gxtb=True,
    use_last_hess=True,
    save_step=True,
)
```

`Stepper.gxtb(..., options={"hess": None})` asks the g-xTB binary for a
numerical Cartesian Hessian. Tooltoad converts that xTB-style `hessian` file to
an ORCA text `.hess` file and stores it in the dataframe, usually as
`gxtb-hess-input.hess`. `use_last_hess=True` tells FRUST to write the latest
`*.hess` dataframe column to ORCA as `private_input.hess` and add:

```orca
%geom
  inhess Read
  InHessName "private_input.hess"
end
```

Use `NumFreq` when you want ORCA to verify the optimized geometry with
finite-difference frequencies:

```python
df_ts = step.orca(
    df,
    name="gxtb-OptTS",
    options={"OptTS": None, "NumFreq": None},
    gxtb=True,
)
```

Do not add `%geom Calc_Hess true` for this external g-xTB route. With ORCA 6.1,
that makes ORCA enter an internal Hessian/property-integral path that is not
compatible with the external g-xTB method. Use the `use_last_hess=True` route
above when you want an initial g-xTB Hessian.

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
- Direct `Stepper.gxtb(...)` is not the TS optimization route. Use
  `Stepper.orca(..., options={"OptTS": None}, gxtb=True)` for ORCA-driven TS
  searches.
- ORCA `Freq` is not compatible with `gxtb=True` because g-xTB is supplied as
  an external method through `ExtOpt`. Use `NumFreq` for finite-difference
  frequencies.
- ORCA `%geom Calc_Hess true` is not compatible with `gxtb=True`. Use
  `Stepper.gxtb(..., options={"hess": None})` followed by
  `Stepper.orca(..., gxtb=True, use_last_hess=True)` when you want an initial
  g-xTB Hessian.
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
