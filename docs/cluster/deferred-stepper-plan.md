# Deferred Stepper Plan

## Summary

The goal is to make FRUST feel more like a calculator toolbox while still
supporting cluster execution. The current `Stepper` API executes immediately:

```python
df = step.gxtb(df, name="gxtb-opt", options={"opt": None})
df = step.orca(df, name="DFT-SP", options={...})
```

That works well locally, but it does not map cleanly onto Slurm because each
line expects a real dataframe back immediately. A cluster submission is
asynchronous, so returning a dataframe from `step.gxtb(...)` would be
misleading.

The proposed solution is to let `Stepper` support a deferred mode. In deferred
mode, calculator methods record stages instead of executing them. The user then
chooses whether to run the recorded stages locally or submit them to the
cluster.

## Desired User Experience

Local immediate mode should keep working exactly as it does today:

```python
step = Stepper("TS1", n_cores=10)

df = step.gxtb(
    df,
    name="gxtb-opt",
    options={"opt": None},
    n_cores=1,
    constraint=True,
)

df = step.orca(
    df,
    name="DFT-SP-solvent",
    options={
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    },
    xtra_inp_str="""%CPCM
SMD TRUE
SMDSOLVENT "chloroform"
end""",
)
```

Deferred mode should make the same calculation describable without requiring a
custom submitit wrapper:

```python
from frust.cluster import ClusterConfig, Resources
from frust.stepper import Stepper

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/RMSD-PP-GFN2-gxTB",
)

step = Stepper("TS1", n_cores=10).defer()

step.gxtb(
    name="gxtb-opt",
    options={"opt": None},
    n_cores=1,
    constraint=True,
)

step.orca(
    name="DFT-SP-solvent",
    options={
        "wB97X-D3": None,
        "6-31+G**": None,
        "TightSCF": None,
        "SP": None,
        "NoSym": None,
    },
    xtra_inp_str="""%CPCM
SMD TRUE
SMDSOLVENT "chloroform"
end""",
)

job = step.submit(
    df,
    cluster=cluster,
    resources=Resources(cpus=10, mem_gb=48, timeout_min=14400),
    output_parquet="DFT-SP-SOLV(GFN2-RMSD-PP-gxTB(opt)).parquet",
    output_base="DFT-SP-SOLV-GFN2-RMSD-PP-gxTB",
    job_name_from="custom_name",
)
```

The local equivalent should be:

```python
df_out = step.run(df)
df_out.to_parquet("DFT-SP-SOLV(GFN2-RMSD-PP-gxTB(opt)).parquet")
```

## Core Design

`Stepper` gains an explicit deferred mode:

```python
step = Stepper("TS1").defer()
```

In deferred mode:

- `step.xtb(...)`, `step.gxtb(...)`, and `step.orca(...)` do not require a
  dataframe argument.
- Each method records a stage in `step.stages`.
- Each recorded stage stores the method name, keyword arguments, and optional
  per-stage resource hints.
- `step.run(df)` executes all recorded stages locally and returns the final
  dataframe.
- `step.submit(df, ...)` submits one Slurm or local submitit job that executes
  `step.run(df)` remotely and writes the result parquet.

The stage list should be plain, pickle-friendly Python data:

```python
[
    {
        "method": "gxtb",
        "kwargs": {
            "name": "gxtb-opt",
            "options": {"opt": None},
            "n_cores": 1,
            "constraint": True,
        },
        "resources": None,
    },
    {
        "method": "orca",
        "kwargs": {
            "name": "DFT-SP-solvent",
            "options": {
                "wB97X-D3": None,
                "6-31+G**": None,
                "TightSCF": None,
                "SP": None,
                "NoSym": None,
            },
            "xtra_inp_str": "%CPCM\nSMD TRUE\nSMDSOLVENT \"chloroform\"\nend",
        },
        "resources": None,
    },
]
```

## API Sketch

### Immediate Mode

Existing behavior remains the default:

```python
step = Stepper("TS1")
df = step.gxtb(df, name="gxtb-opt", options={"opt": None})
```

### Deferred Mode

```python
step = Stepper("TS1").defer()
step.gxtb(name="gxtb-opt", options={"opt": None})
step.orca(name="DFT-SP", options={...})
df_out = step.run(df)
```

### Cluster Submission

```python
job = step.submit(
    df,
    cluster=ClusterConfig(backend="slurm", partition="kemi1"),
    resources=Resources(cpus=10, mem_gb=48, timeout_min=14400),
    output_parquet="result.parquet",
    output_base="runs/example",
    job_name_from="custom_name",
)
```

`submit(...)` should:

- create a submitit executor through the existing `frust.cluster` helpers;
- set Slurm resources from `Resources`;
- infer a job name from `job_name`, `job_name_from`, or a fallback;
- submit a pickle-friendly runner function;
- write the returned dataframe to `output_parquet` inside the submitted job;
- return a submitit job object or a small FRUST submission result object.

## Why This Is Better Than A Submitit Wrapper

A generic wrapper around a user function:

```python
submit_calculation(run_calc, df, ...)
```

does not add much over direct submitit. The user still has to write the wrapper
function, pass resources manually, manage output naming, and remember how to
write the parquet.

Deferred `Stepper` is more useful because FRUST owns the calculation recipe:

- stages are explicit and inspectable;
- local and cluster execution share the same recorded stages;
- job naming and parquet output can follow FRUST conventions;
- stage metadata and calculator provenance remain in the final dataframe;
- future per-stage Slurm dependencies can build on the same stage list.

## Compatibility Rules

- Do not change the current immediate `Stepper` API.
- Do not make `step.gxtb(df, ...)` behave asynchronously.
- Deferred mode must be explicit through `.defer()` or a similarly obvious
  constructor option.
- In deferred mode, passing a dataframe directly to `gxtb`, `xtb`, or `orca`
  should either execute immediately or raise a clear error. Prefer a clear
  error for the first implementation.
- Stages should store only pickle-friendly data.
- Avoid storing live callables, open files, active executors, or local closures
  inside the stage list.

## Implementation Plan

### 1. Add A Stage Record

Create a small internal dataclass, for example:

```python
@dataclass(frozen=True)
class StepperStage:
    method: str
    kwargs: dict[str, object]
    resources: Resources | None = None
```

This can live near `Stepper` initially. If it grows, move it to a dedicated
module.

### 2. Add Deferred Mode To `Stepper`

Add fields:

```python
self.deferred = False
self.stages = []
```

Add:

```python
def defer(self) -> "Stepper":
    clone = copy.copy(self)
    clone.deferred = True
    clone.stages = []
    return clone
```

Cloning avoids surprising mutation of an already-used immediate `Stepper`.

### 3. Record Stages In Deferred Mode

At the start of `xtb`, `gxtb`, and `orca`:

- if not deferred, keep current behavior;
- if deferred, require that no dataframe positional argument was passed;
- store the method name and kwargs;
- return `self` to allow chaining.

Example:

```python
step = (
    Stepper("TS1")
    .defer()
    .gxtb(name="gxtb-opt", options={"opt": None})
    .orca(name="DFT-SP", options={...})
)
```

### 4. Add `run(df)`

`run(df)` should:

- create an immediate `Stepper` with the same configuration;
- replay each recorded stage;
- return the final dataframe.

This method is useful for validating a deferred plan locally before submitting
to Slurm.

### 5. Add `submit(df, ...)`

`submit(df, ...)` should use the existing cluster building blocks:

- `ClusterConfig`
- `Resources`
- `create_executor`
- `update_executor`

The submitted callable should:

1. reconstruct or receive the deferred `Stepper`;
2. call `step.run(df)`;
3. write the resulting dataframe to `output_parquet`;
4. return the output parquet path.

### 6. Add Job Naming Helpers

Support:

```python
job_name="manual-name"
job_name_from="custom_name"
```

If `job_name_from` is provided, use:

```python
str(df[job_name_from].iloc[0])
```

Then sanitize using the existing cluster naming helper.

### 7. Add Optional Monitoring

Add a light option:

```python
monitor=True
```

If enabled, the submitted function tries:

```python
from nuse import start_monitoring
start_monitoring(filter_cgroup=True)
```

If `nuse` is unavailable, either warn or skip. Do not make monitoring a hard
dependency.

## Open Design Questions

### Should `step.submit(...)` submit one job or one job per stage?

Version 1 should submit one job for the whole recorded plan. It is simpler and
matches the debugging/research use case.

Version 2 could add:

```python
step.submit_chain(df, stage_resources={...})
```

That would submit each stage as a dependent Slurm job and pass parquet files
between stages.

### Should resources live on stages?

Version 1 can use one `Resources` object for the whole job.

Later:

```python
step.gxtb(..., resources=Resources(cpus=1, mem_gb=8, timeout_min=120))
step.orca(..., resources=Resources(cpus=10, mem_gb=48, timeout_min=14400))
```

This only matters once per-stage Slurm dependencies exist.

### Should `Stepper(..., cluster=cluster)` exist?

Maybe later, but not for version 1. It blurs construction and submission.

Prefer:

```python
step = Stepper("TS1").defer()
job = step.submit(df, cluster=cluster, resources=resources)
```

This keeps the asynchronous boundary explicit.

## Test Plan

Unit tests:

- immediate `Stepper` behavior remains unchanged;
- `.defer()` returns a deferred Stepper without mutating the original;
- deferred `xtb`, `gxtb`, and `orca` calls append stages;
- `run(df)` replays stages in order and returns the same output shape as
  immediate mode using mocked calculators;
- invalid deferred calls with a dataframe argument raise a clear error;
- `submit(...)` with `ClusterConfig(backend="local")` writes the output parquet;
- job naming from `custom_name` is sanitized correctly;
- `monitor=True` skips cleanly when `nuse` is unavailable.

Documentation tests or examples:

- show local immediate mode;
- show deferred local run;
- show deferred Slurm submission;
- explain that `submit(...)` returns a job, not a dataframe.

## Acceptance Criteria

- Existing local calculator workflows keep working.
- A user can define a deferred calculation once and either run it locally or
  submit it through Slurm.
- The user no longer needs to write a custom submitit wrapper for basic
  dataframe calculator jobs.
- The final parquet preserves normal FRUST result columns and
  `df.attrs["frust_steps"]` provenance.
- Cluster behavior uses the existing `frust.cluster` executor/config helpers.

