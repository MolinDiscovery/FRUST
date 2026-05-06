# FRUST Cluster Submission

This module provides a small submission layer on top of `submitit` for running FRUST workflows on either:

- a Slurm cluster
- a local executor for testing

The public interface is intentionally small:

```python
from frust.cluster import submit_jobs, submit_chain, ClusterConfig, Resources
```

If you install FRUST without cluster extras, install `submitit` first:

```bash
pip install -e ".[cluster]"
```

## What This Module Is For

Use `frust.cluster` when you want FRUST to submit jobs for you instead of calling the workflow functions directly in Python.

There are two modes:

- `submit_jobs(...)`
  Submit independent jobs, typically one high-level pipeline call per input or per generated TS structure.
- `submit_chain(...)`
  Submit dependent stage pipelines where later jobs wait for earlier jobs to finish successfully.

In practice:

- use `submit_jobs(...)` for `run_mols`, `run_ts_per_lig`, `run_ts_per_rpos`, and the related high-level `frust.pipes` workflows
- use `submit_chain(...)` for staged modules such as `frust.pipelines.run_ts_per_rpos`

## Core Configuration

Every submission uses two small config objects:

```python
from frust.cluster import ClusterConfig, Resources

cluster = ClusterConfig(
    backend="slurm",
    partition="kemi1",
    log_dir="logs/example",
)

resources = Resources(
    cpus=16,
    mem_gb=50,
    timeout_min=14400,
)
```

`ClusterConfig` controls where and how jobs are submitted.

- `backend="slurm"` uses `submitit.AutoExecutor`
- `backend="local"` uses `submitit.LocalExecutor`
- `partition` is only used for Slurm
- `log_dir` is where `submitit` writes logs
- `work_dir` is forwarded into FRUST workflows when they accept it
- `extra_slurm_parameters` can be used for account, qos, dependency tuning, and similar scheduler options

`Resources` controls:

- CPU count
- memory in GB
- timeout in minutes

## Tutorial 1: Submit Independent Molecular Jobs

This is the simplest entry point. `run_mols` takes a CSV with a `smiles` column and submits one independent workflow.

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_mols",
    out_dir="runs/mols_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/mols_example",
    ),
    resources=Resources(cpus=16, mem_gb=50, timeout_min=14400),
    debug=False,
    production=True,
    n_confs=None,
    save_output_dir=True,
    dft=False,
    select_mols="all",
)

result
```

This will:

- read the CSV
- validate that it contains a `smiles` column
- call `frust.pipes.run_mols`
- create parquet outputs under `runs/mols_example`
- return a `JobSubmissionResult`

## Tutorial 2: Submit `run_ts_per_lig`

Some pipelines need a TS template. In that case you must provide `ts_xyz`.

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_ts_per_lig",
    ts_xyz="structures/ts1.xyz",
    out_dir="runs/ts_per_lig_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/ts_per_lig_example",
    ),
    resources=Resources(cpus=16, mem_gb=50, timeout_min=14400),
    save_output_dir=True,
    dft=False,
)
```

## Tutorial 3: Submit `run_ts_per_rpos`

For `run_ts_per_rpos`, FRUST first expands the CSV into multiple `ts_struct` jobs using the TS template, then submits one independent job per generated TS structure.

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_ts_per_rpos",
    ts_xyz="structures/ts2.xyz",
    out_dir="runs/ts_per_rpos_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/ts_per_rpos_example",
    ),
    resources=Resources(cpus=16, mem_gb=50, timeout_min=14400),
    save_output_dir=True,
    dft=False,
)
```

This is the right choice when you want the high-level `run_ts_per_rpos(...)` workflow itself to run as separate submitted jobs.

## Tutorial 4: Submit a Dependent Stage Chain

If you want the staged pipeline in `frust.pipelines.run_ts_per_rpos`, use `submit_chain(...)` instead.

The simplest way is to use a built-in preset:

```python
from frust.cluster import submit_chain, ClusterConfig, Resources

result = submit_chain(
    csv_path="datasets/example.csv",
    preset="ts_per_rpos",
    ts_xyz="structures/ts2.xyz",
    out_dir="runs/ts_chain_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/ts_chain_example",
    ),
    stage_resources={
        "run_init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "run_hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "run_OptTS": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "run_freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "run_solv": Resources(cpus=24, mem_gb=20, timeout_min=3600),
        "run_cleanup": Resources(cpus=2, mem_gb=2, timeout_min=60),
    },
    debug=False,
    production=True,
    n_confs=None,
    save_output_dir=True,
)
```

This preset submits the following chain:

1. `run_init`
2. `run_hess`
3. `run_OptTS`
4. `run_freq`
5. `run_solv`
6. `run_cleanup`

Each stage is submitted with Slurm `afterok` dependencies when using the Slurm backend.

## Tutorial 5: Use the INT3 Chain Preset

FRUST also includes a built-in preset for the `int3` stage module.

```python
from frust.cluster import submit_chain, ClusterConfig, Resources

result = submit_chain(
    csv_path="datasets/example.csv",
    preset="int3_per_rpos",
    ts_xyz="structures/int3.xyz",
    out_dir="runs/int3_chain_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/int3_chain_example",
    ),
    stage_resources={
        "run_init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "run_Opt": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "run_freq": Resources(cpus=8, mem_gb=64, timeout_min=7200),
        "run_solv": Resources(cpus=24, mem_gb=20, timeout_min=3600),
        "run_cleanup": Resources(cpus=2, mem_gb=2, timeout_min=60),
    },
)
```

If you want to keep the same preset workflow but change the ORCA level of
theory, pass it directly into `submit_chain(...)`:

```python
from frust.cluster import submit_chain, ClusterConfig

result = submit_chain(
    csv_path="datasets/example.csv",
    preset="int3_per_rpos",
    ts_xyz="structures/int3.xyz",
    out_dir="runs/int3_chain_b3lyp",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/int3_chain_b3lyp",
    ),
    functional="B3LYP",
    basisset="def2-SVP",
    basisset_solv="def2-SVPD",
)
```

This keeps the same stage order and resource handling, but swaps the ORCA
keywords used by the preset stages.

## Tutorial 6: Use a Custom Chain

If you do not want to use a preset, you can specify the stage module and stage order directly.

```python
from frust.cluster import submit_chain, ClusterConfig, Resources

result = submit_chain(
    csv_path="datasets/example.csv",
    module_path="frust.pipelines.run_ts_per_rpos",
    stage_order=[
        "run_init",
        "run_hess",
        "run_OptTS",
        "run_freq",
        "run_solv",
        "run_cleanup",
    ],
    ts_xyz="structures/ts2.xyz",
    out_dir="runs/custom_chain_example",
    cluster=ClusterConfig(
        backend="slurm",
        partition="kemi1",
        log_dir="logs/custom_chain_example",
    ),
    stage_resources={
        "run_init": Resources(cpus=24, mem_gb=20, timeout_min=7200),
        "run_hess": Resources(cpus=8, mem_gb=64, timeout_min=7200),
    },
)
```

For custom chains:

- `module_path` and `stage_order` are both required
- missing stage resource entries fall back to `Resources(cpus=4, mem_gb=20, timeout_min=720)`
- stage names must exist as callables in the target module

## Local Testing

You can test the submission logic locally before using Slurm:

```python
from frust.cluster import submit_jobs, ClusterConfig, Resources

result = submit_jobs(
    csv_path="datasets/example.csv",
    pipeline="run_mols",
    out_dir="runs/local_test",
    cluster=ClusterConfig(
        backend="local",
        log_dir="logs/local_test",
    ),
    resources=Resources(cpus=1, mem_gb=2, timeout_min=10),
    debug=True,
    production=False,
    n_confs=1,
    save_output_dir=False,
)
```

This is useful for:

- checking that your CSV and template paths are correct
- confirming that the workflow wiring is valid
- testing the submission interface before using real cluster resources

## Returned Result Object

Both `submit_jobs(...)` and `submit_chain(...)` return a `JobSubmissionResult`:

```python
JobSubmissionResult(
    job_ids=[...],
    tags=[...],
    save_dirs=[...],
    mode="...",
    backend="slurm",
)
```

This is mainly useful for:

- printing or storing scheduler ids
- inspecting which tags were submitted
- locating output directories programmatically

## Current Presets

The currently supported built-in chain presets are:

- `ts_per_rpos`
- `int3_per_rpos`

## Common Errors

Missing `smiles` column:

- your CSV must contain a `smiles` column

Missing `ts_xyz`:

- `run_ts_per_lig`
- `run_ts_per_rpos`
- `run_ts_per_rpos_UMA`
- `run_ts_per_rpos_UMA_short`
- `run_orca_smoke_test`
- all `submit_chain(...)` calls

Mixed preset and custom arguments:

- use either `preset=...`
- or `module_path=...` together with `stage_order=[...]`

Unsupported pipeline name:

- `submit_jobs(...)` only supports the FRUST pipelines that are wired into the cluster interface

## Design Notes

This module is intentionally Python-first and CSV-first. The aim is to make common cluster use cases simple for researchers without requiring them to manually import `submitit`, discover stage functions, or build scheduler dependencies themselves.
