from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from frust.cluster.config import ClusterConfig, Resources


def _load_submitit():
    """Import :mod:`submitit` with a FRUST-specific error message."""
    try:
        import submitit
    except ImportError as exc:
        raise ImportError(
            "frust.cluster requires 'submitit'. Install FRUST with the 'cluster' extra."
        ) from exc
    return submitit


def create_executor(cluster: ClusterConfig):
    """Create a submitit executor for the requested backend.

    Parameters
    ----------
    cluster : frust.cluster.config.ClusterConfig
        Cluster or local-executor configuration.

    Returns
    -------
    submitit.Executor
        Configured submitit executor instance.
    """
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
    """Apply non-dependent resource settings to an executor.

    Parameters
    ----------
    executor
        Submitit executor instance.
    cluster : frust.cluster.config.ClusterConfig
        Cluster or local-executor configuration.
    resources : frust.cluster.config.Resources
        CPU, memory, and timeout settings for the job.
    job_name : str
        Scheduler-visible job name.
    """
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
    """Apply resource settings and an optional dependency to an executor.

    Parameters
    ----------
    executor
        Submitit executor instance.
    cluster : frust.cluster.config.ClusterConfig
        Cluster or local-executor configuration.
    resources : frust.cluster.config.Resources
        CPU, memory, and timeout settings for the job.
    job_name : str
        Scheduler-visible job name.
    dependency_job_id : str or int or None
        Upstream job identifier used to build a Slurm ``afterok`` dependency.
    """
    dependency_job_ids = [] if dependency_job_id is None else [dependency_job_id]
    update_executor_with_dependencies(
        executor,
        cluster,
        resources,
        job_name=job_name,
        dependency_job_ids=dependency_job_ids,
        dependency_type="afterok",
    )


def update_executor_with_dependencies(
    executor,
    cluster: ClusterConfig,
    resources: Resources,
    *,
    job_name: str,
    dependency_job_ids: Iterable[str | int] | None,
    dependency_type: str = "afterok",
):
    """Apply resource settings and optional Slurm dependencies to an executor.

    Parameters
    ----------
    executor
        Submitit executor instance.
    cluster : frust.cluster.config.ClusterConfig
        Cluster or local-executor configuration.
    resources : frust.cluster.config.Resources
        CPU, memory, and timeout settings for the job.
    job_name : str
        Scheduler-visible job name.
    dependency_job_ids : iterable of str or int or None
        Upstream job identifiers. Empty or ``None`` means no dependency.
    dependency_type : {"afterok", "afterany"}, optional
        Slurm dependency condition. ``"afterok"`` starts the job only after all
        upstream jobs finish successfully. ``"afterany"`` starts the job after
        all upstream jobs finish in any state.

    Raises
    ------
    ValueError
        If ``dependency_type`` is not supported.
    """
    if dependency_type not in {"afterok", "afterany"}:
        raise ValueError("dependency_type must be 'afterok' or 'afterany'")

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
        dependency_ids = [str(job_id) for job_id in (dependency_job_ids or [])]
        if dependency_ids:
            extra["dependency"] = f"{dependency_type}:{':'.join(dependency_ids)}"
        params["slurm_additional_parameters"] = extra
    executor.update_parameters(**params)
