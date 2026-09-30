# UMA With FRUST

For a complete catalyst screen, start with
[UMA Screening And ωB97 Validation](../catalyst-screens/uma-screening.md).
This page covers the lower-level `Stepper.orca(...)` controls and the ORCA
input that carries them to ORCA-External-Tools (OET).

The examples use `omol@uma-s-1p2p1` with the MolinDiscovery OET fork at
`1b4fcda` and an OET runtime with `fairchem-core` 2.23.0. Specify the model
explicitly: a task-only `uma="omol"` still defaults to the older `uma-s-1p1`
model in the low-level API.

## Requirements

FRUST expects ORCA-External-Tools to be installed and available through
`OET_TOOLS`. Put this in `~/.env`:

```bash
OET_TOOLS=/path/to/orca-external-tools
```

The path is resolved only when UMA/OET functionality is used.

The expected OET 2 executables are:

```text
<OET_TOOLS>/bin/oet_server
<OET_TOOLS>/bin/oet_client
<OET_TOOLS>/bin/oet_uma
```

See [External tool setup](../getting-started/external-tool-setup.md) for
installation, `.env` setup, and smoke tests.

## Basic API

Use an explicit model through the public Stepper API:

```python
import frust as ft

step = ft.Stepper(n_cores=4, memory_gb=16)
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
)
```

The `uma` argument accepts a task alone or a task plus model:

```python
uma="omol"                # low-level default: uma-s-1p1
uma="omol@uma-s-1p2p1"   # model used by the UMA screening presets
```

The explicit model becomes these OET arguments:

```text
-t omol -m uma-s-1p2p1
```

## Full UMA Arguments

`Stepper.orca(...)` exposes these UMA-specific arguments:

```python
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_server=True,
    uma_device="cpu",
    uma_cache_dir=None,
    uma_offline=False,
    uma_server_cores=None,
    uma_memory_per_thread_mib=500,
    uma_keep_logs="on_failure",
    uma_log_dir=None,
    uma_xtb_alpb=None,  # or "chloroform"
    uma_inference_settings="batch",
)
```

Pin the model as shown; the other defaults are a reasonable starting point.

## Gas Or ALPB(chloroform)

Omit `uma_xtb_alpb` for gas-phase UMA. To add the tested solvent correction:

```python
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_xtb_alpb="chloroform",
    uma_inference_settings="batch",
)
```

OET evaluates the same geometry, charge, multiplicity, and atom order in all
three terms:

```text
E = E_UMA + E_GFN2-xTB,ALPB(chloroform) - E_GFN2-xTB,gas
∇E = ∇E_UMA + ∇E_GFN2-xTB,ALPB(chloroform) - ∇E_GFN2-xTB,gas
```

The saved ORCA input exposes the correction in `%method`:

```orca
%method
  ProgExt "/path/to/orca-external-tools/bin/oet_client"
  Ext_Params "-b 127.0.0.1:54403 -t omol -m uma-s-1p2p1 -d cpu --xtb-alpb chloroform --inference-settings batch"
end
```

`uma_xtb_exe="/path/to/xtb"` selects a particular normal xTB executable
when the OET environment does not resolve the intended one. This argument
requires `uma_xtb_alpb`. The correction is an OET option passed through ORCA;
it is not ORCA's SMD solvent model. In the built-in catalyst workflow,
`screening="uma-alpb-chloroform"` applies it to both `uma_sp` and `uma_opt`.

## Server Mode

Server mode is the default:

```python
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
)
```

With `uma_server=True`, FRUST starts:

```bash
<OET_TOOLS>/bin/oet_server uma --bind 127.0.0.1:<free_port>
```

FRUST then waits for:

```text
http://127.0.0.1:<free_port>/healthz
```

and injects an ORCA block like:

```orca
%method
   ProgExt "/path/to/orca-external-tools/bin/oet_client"
Ext_Params "-b 127.0.0.1:54403 -t omol -m uma-s-1p2p1 -d cpu"
end
%output
Print[P_EXT_OUT] 1
Print[P_EXT_GRAD] 1
end
```

After ORCA finishes, FRUST shuts down the full UMA server process group unless
the call is inside a workflow stage group that will reuse it. A workflow starts
one server lazily inside each executing job group, reuses it across UMA stages
in that group, and stops it when the group exits. A numerical-frequency stage
in the same group can reuse that server. Separately submitted groups have
separate server lifetimes.

Server mode is used locally and in submitted jobs. With the Slurm submitit
backend, FRUST is already running inside the allocated job, so the server and
`oet_client` calls stay on that compute node through a `127.0.0.1` bind.

## Standalone Mode

Use standalone mode only when you explicitly do not want a server:

```python
df = step.orca(
    df,
    name="uma-opt-standalone",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_server=False,
)
```

This injects:

```orca
%method
   ProgExt "/path/to/orca-external-tools/bin/oet_uma"
Ext_Params "-t omol -m uma-s-1p2p1 -d cpu"
end
%output
Print[P_EXT_OUT] 1
Print[P_EXT_GRAD] 1
end
```

Standalone mode starts a new UMA process for each external call, so it is
usually slower for optimizations.

## Device, Cache, And Offline Mode

CPU is the default:

```python
df = step.orca(df, options={"ExtOpt": None, "Opt": None}, uma="omol@uma-s-1p2p1")
```

CUDA can be requested with:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_device="cuda",
)
```

The device is passed to OET as `-d cpu` or `-d cuda`.

To use a specific FairChem cache directory:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_cache_dir="/path/to/fairchem-cache",
)
```

This adds:

```text
-c /path/to/fairchem-cache
```

To request offline mode:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_offline=True,
)
```

This adds:

```text
-o True
```

## Cores And Memory

The ORCA call uses `Stepper.n_cores` unless overridden with `n_cores=...`:

```python
step = ft.Stepper(n_cores=8, memory_gb=30)

df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
)
```

By default the UMA server receives the same core count as this ORCA call.
Override the UMA server budget separately with:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    n_cores=8,
    uma_server_cores=4,
    uma_memory_per_thread_mib=750,
)
```

This starts the server with:

```text
--nthreads 4 --memory-per-thread 750
```

## Server Logs

Server logs are controlled with `uma_keep_logs`:

```python
uma_keep_logs="on_failure"  # default
uma_keep_logs=True          # same as "always"
uma_keep_logs="always"
uma_keep_logs=False         # same as "never"
uma_keep_logs="never"
```

The default keeps logs only if the UMA-backed ORCA step fails:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_keep_logs="on_failure",
)
```

If preserved and `uma_log_dir` is not set, logs go to:

```text
UMA-logs/
```

To choose a log directory:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_keep_logs="always",
    uma_log_dir="dev/uma-logs",
)
```

Each log starts with the launcher command and useful cluster context:

```text
[launcher] bind=127.0.0.1:54403 server_cores=10 memory_per_thread_mib=500 ...
```

## Common Workflows

Single-point style external call:

```python
df = step.orca(
    df,
    name="uma_sp",
    options={"ExtOpt": None},
    uma="omol@uma-s-1p2p1",
)
```

Geometry optimization:

```python
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    save_step=True,
)
```

Transition-state optimization:

```python
df = step.orca(
    df,
    name="uma-OptTS",
    options={"ExtOpt": None, "OptTS": None},
    uma="omol@uma-s-1p2p1",
    save_step=True,
)
```

Numerical frequency check:

```python
df = step.orca(
    df,
    name="uma-OptTS-NumFreq",
    options={"ExtOpt": None, "OptTS": None, "NumFreq": None},
    uma="omol@uma-s-1p2p1",
    save_step=True,
)
```

## Constraints And Hessians

FRUST's usual ORCA constraint handling still applies:

```python
df = step.orca(
    df,
    name="uma-constrained-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    constraint=True,
)
```

`use_last_hess=True` also works with UMA if the dataframe already contains a
previous `*.hess` column:

```python
df = step.orca(
    df,
    name="uma-OptTS-readhess",
    options={"ExtOpt": None, "OptTS": None},
    uma="omol@uma-s-1p2p1",
    use_last_hess=True,
)
```

FRUST writes the latest `*.hess` dataframe column to ORCA as
`private_input.hess` and adds:

```orca
%geom
  inhess Read
  InHessName "private_input.hess"
end
```

## Output Columns

FRUST uses the same output-column convention as other `Stepper` engines.

For:

```python
df = step.orca(
    df,
    name="uma_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
)
```

you should expect columns such as:

```text
uma_opt-EE
uma_opt-NT
uma_opt-oc
```

where:

```text
EE = electronic energy
NT = normal termination
oc = optimized coordinates
```

Frequency jobs may also add vibration and Gibbs-energy columns depending on
what ORCA returns.

The step metadata is stored in:

```python
df.attrs["frust_steps"]["uma_opt"]
```

and includes:

```python
{
    "engine": "orca",
    "options": {"ExtOpt": None, "Opt": None},
    "uma": "omol@uma-s-1p2p1",
    "uma_task": "omol",
    "uma_model": "uma-s-1p2p1",
    "uma_server": True,
}
```

For a corrected step, the metadata also records
`"uma_xtb_alpb": "chloroform"`; calculator provenance records the xTB
method, ALPB model, and solvent. Inspect the actual row and stage metadata
with `ft.show_steps(df)` and `df.attrs["frust_steps"]["uma_opt"]`.

## Limitations

- `uma` and `gxtb=True` are mutually exclusive in one `Stepper.orca(...)` call.
- `uma` must be a non-empty string.
- `uma="@uma-s-1p2p1"` is invalid because the task is missing.
- `uma="omol@"` is invalid because the model is missing.
- Server mode binds to `127.0.0.1`; it is intended for the current process and
  current compute node, not for a shared network service.
- If the server fails to start, FRUST raises an error pointing to the preserved
  UMA server log.
