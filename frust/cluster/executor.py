from __future__ import annotations

from pathlib import Path

from frust.cluster.config import ClusterConfig, Resources


def _load_submitit():
    try:
        import submitit
    except ImportError as exc:
        raise ImportError(
            "frust.cluster requires 'submitit'. Install FRUST with the 'cluster' extra."
        ) from exc
    return submitit


def create_executor(cluster: ClusterConfig):
    submitit = _load_submitit()
    log_dir = Path(cluster.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if cluster.backend == "slurm":
        executor = submitit.AutoExecutor(folder=str(log_dir))
    elif cluster.backend == "local":
        executor = submitit.LocalExecutor(folder=str(log_dir))
    else:
        raise ValueError("cluster.backend must be either 'slurm' or 'local'")
    return executor


def update_executor(executor, cluster: ClusterConfig, resources: Resources, *, job_name: str):
    params = {
        "cpus_per_task": resources.cpus,
        "mem_gb": resources.mem_gb,
        "timeout_min": resources.timeout_min,
    }
    if cluster.backend == "slurm":
        params["slurm_job_name"] = job_name
        if cluster.partition is not None:
            params["slurm_partition"] = cluster.partition
        params["slurm_additional_parameters"] = dict(cluster.extra_slurm_parameters or {})
    executor.update_parameters(**params)


def update_executor_with_dependency(
    executor,
    cluster: ClusterConfig,
    resources: Resources,
    *,
    job_name: str,
    dependency_job_id: str | int | None,
):
    params = {
        "cpus_per_task": resources.cpus,
        "mem_gb": resources.mem_gb,
        "timeout_min": resources.timeout_min,
    }
    if cluster.backend == "slurm":
        params["slurm_job_name"] = job_name
        if cluster.partition is not None:
            params["slurm_partition"] = cluster.partition
        extra = dict(cluster.extra_slurm_parameters or {})
        if dependency_job_id is not None:
            extra["dependency"] = f"afterok:{dependency_job_id}"
        params["slurm_additional_parameters"] = extra
    executor.update_parameters(**params)

