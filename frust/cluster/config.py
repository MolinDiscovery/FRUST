from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

"""Configuration models and built-in presets for FRUST cluster submission."""


class ChainPreset(StrEnum):
    """Named dependent-stage submission presets bundled with FRUST."""

    TS_PER_RPOS = "ts_per_rpos"
    INT3_PER_RPOS = "int3_per_rpos"
    SCREEN_TS_PER_RPOS = "screen_ts_per_rpos"


@dataclass(frozen=True)
class Resources:
    """Execution resources for a single submitted job.

    Parameters
    ----------
    cpus : int
        Number of CPU cores requested for the job.
    mem_gb : int or float
        Memory requested for the job in gigabytes.
    timeout_min : int
        Wall-clock timeout in minutes.
    """

    cpus: int
    mem_gb: int | float
    timeout_min: int


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster and executor settings shared across submitted jobs.

    Parameters
    ----------
    backend : {"slurm", "local"}, optional
        Execution backend. Use ``"slurm"`` for cluster submission through
        :mod:`submitit` or ``"local"`` for local testing. Defaults to
        ``"slurm"``.
    partition : str or None, optional
        Slurm partition name. Ignored for the local backend.
    log_dir : str or pathlib.Path, optional
        Directory in which submitit writes executor logs.
    work_dir : str or pathlib.Path or None, optional
        Optional scratch or work directory forwarded to FRUST pipelines when
        they accept a ``work_dir`` argument.
    extra_slurm_parameters : dict[str, str] or None, optional
        Additional scheduler parameters forwarded as
        ``slurm_additional_parameters``.
    """

    backend: str = "slurm"
    partition: str | None = None
    log_dir: str | Path = "logs"
    work_dir: str | Path | None = None
    extra_slurm_parameters: dict[str, str] | None = None


@dataclass(frozen=True)
class JobSubmissionResult:
    """Summary information returned after submission.

    Parameters
    ----------
    job_ids : list[str or int]
        Scheduler or executor job identifiers in submission order.
    tags : list[str]
        Sanitized job tags used for naming and logging.
    save_dirs : list[str]
        Output directories associated with the submitted jobs.
    mode : str
        Submitted workflow mode, such as a pipeline name or chain preset.
    backend : str
        Backend used for submission, typically ``"slurm"`` or ``"local"``.
    collection_job_id : str or int or None, optional
        Scheduler or executor job identifier for the automatic collection job,
        when one was submitted.
    collection_output : str or None, optional
        Path to the merged parquet written by the automatic collection job.
    collection_report : str or None, optional
        Path to the JSON report written by the automatic collection job.
    """

    job_ids: list[str | int]
    tags: list[str]
    save_dirs: list[str]
    mode: str
    backend: str
    collection_job_id: str | int | None = None
    collection_output: str | None = None
    collection_report: str | None = None


DEFAULT_CUSTOM_STAGE_RESOURCES = Resources(cpus=4, mem_gb=20, timeout_min=720)
DEFAULT_ORCA_MEMORY_FRACTION = 0.8


def orca_memory_gb(
    resources: Resources,
    fraction: float = DEFAULT_ORCA_MEMORY_FRACTION,
) -> float:
    """Return the memory budget forwarded to ORCA for a submitted job.

    Parameters
    ----------
    resources : Resources
        Full CPU, memory, and timeout allocation requested from the scheduler.
    fraction : float, optional
        Portion of ``resources.mem_gb`` made available to ORCA. It must be
        greater than zero and no greater than one. The default, ``0.8``,
        reserves 20 percent of the Slurm allocation for Python, filesystem,
        and other process overhead.

    Returns
    -------
    float
        ORCA memory budget in GB.

    Examples
    --------
    A 64 GB Slurm allocation leaves 51.2 GB available to ORCA:

    >>> orca_memory_gb(Resources(cpus=12, mem_gb=64, timeout_min=720))
    51.2
    """
    if not 0 < fraction <= 1:
        raise ValueError("orca_memory_fraction must be greater than zero and no greater than one")
    return float(resources.mem_gb) * fraction


CHAIN_PRESET_MODULES: dict[ChainPreset, str] = {
    ChainPreset.TS_PER_RPOS: "frust.pipelines.run_ts_per_rpos",
    ChainPreset.INT3_PER_RPOS: "frust.pipelines.run_int3_per_rpos",
    ChainPreset.SCREEN_TS_PER_RPOS: "frust.pipelines.run_screen_ts_per_rpos",
}


CHAIN_PRESET_STAGE_ORDER: dict[ChainPreset, list[str]] = {
    ChainPreset.TS_PER_RPOS: [
        "run_init",
        "run_hess",
        "run_OptTS",
        "run_freq",
        "run_solv",
        "run_cleanup",
    ],
    ChainPreset.INT3_PER_RPOS: [
        "run_init",
        "run_Opt",
        "run_freq",
        "run_solv",
        "run_cleanup",
    ],
    ChainPreset.SCREEN_TS_PER_RPOS: [
        "run_init",
        "run_hess",
        "run_OptTS",
        "run_freq",
        "run_solv",
        "run_cleanup",
    ],
}


CHAIN_PRESET_RESOURCES: dict[ChainPreset, dict[str, Resources]] = {
    ChainPreset.TS_PER_RPOS: {
        "run_init": Resources(24, 20, 7200),
        "run_hess": Resources(8, 64, 7200),
        "run_OptTS": Resources(24, 20, 7200),
        "run_freq": Resources(8, 64, 7200),
        "run_solv": Resources(24, 20, 3600),
        "run_cleanup": Resources(2, 2, 60),
    },
    ChainPreset.INT3_PER_RPOS: {
        "run_init": Resources(24, 20, 7200),
        "run_Opt": Resources(24, 20, 7200),
        "run_freq": Resources(8, 64, 7200),
        "run_solv": Resources(24, 20, 3600),
        "run_cleanup": Resources(2, 2, 60),
    },
    ChainPreset.SCREEN_TS_PER_RPOS: {
        "run_init": Resources(24, 20, 7200),
        "run_hess": Resources(8, 64, 7200),
        "run_OptTS": Resources(24, 20, 7200),
        "run_freq": Resources(8, 64, 7200),
        "run_solv": Resources(24, 20, 3600),
        "run_cleanup": Resources(2, 2, 60),
    },
}
