# FRUST Cluster Submission

Use the same workflow object for a local smoke test and a submitted production
run. Method-aware TS profiles, chemistry targets, and stage definitions stay
attached to that object.

```python
import frust as ft

cluster = ft.ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/r2scan-screen",
)

wf = ft.workflows.screen_ts(
    csv_path="screen.csv",
    ts_types=["TS1", "TS2", "TS4"],
    method="r2scan-3c-solv",
    n_confs=None,
    top_n=20,
)

wf.resolved_spec_profile
wf.show_stages()[["group", "stage", "engine"]]
```

Output:

```text
'r2scan-3c/smd-chloroform'
```

Run one target locally before submission:

```python
debug = wf.run(
    targets=[0],
    out_dir="debug/r2scan-screen",
    execution="dft_staged",
    n_cores=4,
)
```

Then submit the same target graph and method plan:

```python
result = wf.submit(
    out_dir="runs/r2scan-screen",
    cluster=cluster,
    execution="dft_staged",
    collect=True,
)
```

## Resource Groups

Inspect resource-group names with `wf.show_stages()` before overriding them:

```python
from frust.cluster import Resources

resources = {
    "init": Resources(cpus=8, mem_gb=24, timeout_min=720),
    "dft_hessian": Resources(cpus=12, mem_gb=64, timeout_min=1440),
    "dft_ts_opt": Resources(cpus=24, mem_gb=64, timeout_min=2880),
    "dft_freq": Resources(cpus=12, mem_gb=64, timeout_min=1440),
}

result = wf.submit(
    out_dir="runs/r2scan-screen",
    cluster=cluster,
    execution="dft_staged",
    stage_resources=resources,
)
```

## Lower-Level Screen Chain

`ft.cluster.submit_screen_chain(...)` remains available for scripts that use
the explicit staged screen module:

```python
result = ft.cluster.submit_screen_chain(
    csv_path="screen.csv",
    ts_types=["TS1", "TS2", "TS4"],
    out_dir="runs/staged-screen",
    cluster=cluster,
    composite_method="r2SCAN-3c",
    spec_profile="r2scan-3c/gas",
    spec_match="prefer-exact",
)
```

With `spec_profile="auto"`, the initialization stage derives the geometry
profile from `functional`/`basisset` or `composite_method`. Exact matching can
be required with `spec_match="exact"`.

!!! warning "No XYZ-template chain"

    The positional XYZ-template presets and `submit_chain(...)` entry point
    were removed. Use `ft.workflows.screen_ts(...).submit(...)` for modern TS
    work, or convert historical data explicitly with
    `ft.upgrade_legacy_constraints(...)`.

## Collect Results

Automatic collection writes the merged parquet and a compact report:

```python
merged = wf.collect("runs/r2scan-screen")
ft.show_steps(merged)
```

`result.job_ids`, `result.tags`, and `result.save_dirs` can be used to inspect
individual submitted targets.

## Local Submission Test

Use `backend="local"` to test submission wiring without Slurm:

```python
local = ft.ClusterConfig(backend="local", log_dir="logs/local-screen")
result = wf.submit(
    out_dir="runs/local-screen",
    cluster=local,
    execution="xtb_only",
    targets=[0],
)
```
