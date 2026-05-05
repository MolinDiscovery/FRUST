# UMA With FRUST

This page documents the current FRUST API for running UMA through ORCA and
ORCA-External-Tools 2.

## Requirements

FRUST expects ORCA-External-Tools to be installed and available through one of
these environment variables:

```bash
OET_TOOLS=/Users/jacobmolinnielsen/Library/orca-external-tools
```

The old name still works:

```bash
UMA_TOOLS=/Users/jacobmolinnielsen/Library/orca-external-tools
```

If both are set, `OET_TOOLS` wins. The path is resolved only when UMA/OET
functionality is used.

The expected OET 2 executables are:

```text
<OET_TOOLS>/bin/oet_server
<OET_TOOLS>/bin/oet_client
<OET_TOOLS>/bin/oet_uma
```

## Basic API

Use UMA through `Stepper.orca(...)`:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
)
```

The `uma` argument accepts either a task only or a task plus model:

```python
uma="omol"
uma="omol@uma-s-1p1"
```

These are equivalent:

```python
df = step.orca(df, options={"ExtOpt": None, "Opt": None}, uma="omol")

df = step.orca(df, options={"ExtOpt": None, "Opt": None}, uma="omol@uma-s-1p1")
```

Internally they become OET arguments:

```text
-t omol -m uma-s-1p1
```

## Full UMA Arguments

`Stepper.orca(...)` exposes these UMA-specific arguments:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p1",
    uma_server=True,
    uma_device="cpu",
    uma_cache_dir=None,
    uma_offline=False,
    uma_server_cores=None,
    uma_memory_per_thread_mib=500,
    uma_keep_logs="on_failure",
    uma_log_dir=None,
)
```

The defaults are usually the right starting point.

## Server Mode

Server mode is the default:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
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
ProgExt "/Users/jacobmolinnielsen/Library/orca-external-tools/bin/oet_client"
Ext_Params "-b 127.0.0.1:54403 -t omol -m uma-s-1p1 -d cpu"
end
%output
Print[P_EXT_OUT] 1
Print[P_EXT_GRAD] 1
end
```

After ORCA finishes, FRUST shuts down the full UMA server process group.

Server mode is used locally and on clusters. On Slurm, FRUST is already running
inside the allocated job, so starting the server from FRUST means the server is
started on the compute node that owns the calculation.

## Standalone Mode

Use standalone mode only when you explicitly do not want a server:

```python
df = step.orca(
    df,
    name="uma-opt-standalone",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    uma_server=False,
)
```

This injects:

```orca
%method
ProgExt "/Users/jacobmolinnielsen/Library/orca-external-tools/bin/oet_uma"
Ext_Params "-t omol -m uma-s-1p1 -d cpu"
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
df = step.orca(df, options={"ExtOpt": None, "Opt": None}, uma="omol")
```

CUDA can be requested with:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    uma_device="cuda",
)
```

The device is passed to OET as `-d cpu` or `-d cuda`.

To use a specific FairChem cache directory:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
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
    uma="omol",
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
step = Stepper(step_type="none", n_cores=8, memory_gb=30)

df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
)
```

By default the UMA server receives the same core count as this ORCA call.
Override the UMA server budget separately with:

```python
df = step.orca(
    df,
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
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
    uma="omol",
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
    uma="omol",
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
    name="uma-sp",
    options={"ExtOpt": None, "SP": None},
    uma="omol",
)
```

Geometry optimization:

```python
df = step.orca(
    df,
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
    save_step=True,
)
```

Transition-state optimization:

```python
df = step.orca(
    df,
    name="uma-OptTS",
    options={"ExtOpt": None, "OptTS": None},
    uma="omol@uma-s-1p1",
    save_step=True,
)
```

Numerical frequency check:

```python
df = step.orca(
    df,
    name="uma-OptTS-NumFreq",
    options={"ExtOpt": None, "OptTS": None, "NumFreq": None},
    uma="omol@uma-s-1p1",
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
    uma="omol",
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
    uma="omol",
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
    name="uma-opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol",
)
```

you should expect columns such as:

```text
uma-opt-EE
uma-opt-NT
uma-opt-oc
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
df.attrs["frust_steps"]["uma-opt"]
```

and includes:

```python
{
    "engine": "orca",
    "options": {"ExtOpt": None, "Opt": None},
    "uma": "omol",
    "uma_task": "omol",
    "uma_model": "uma-s-1p1",
    "uma_server": True,
}
```

## Limitations

- `uma` and `gxtb=True` are mutually exclusive in one `Stepper.orca(...)` call.
- `uma` must be a non-empty string.
- `uma="@uma-s-1p1"` is invalid because the task is missing.
- `uma="omol@"` is invalid because the model is missing.
- Server mode binds to `127.0.0.1`; it is intended for the current process and
  current compute node, not for a shared network service.
- If the server fails to start, FRUST raises an error pointing to the preserved
  UMA server log.

