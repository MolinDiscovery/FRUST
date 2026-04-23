from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

"""Configuration models and built-in presets for FRUST cluster submission."""


class ChainPreset(StrEnum):
    """Named dependent-stage submission presets bundled with FRUST."""

    TS_PER_RPOS = "ts_per_rpos"
    INT3_PER_RPOS = "int3_per_rpos"


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
    """

    job_ids: list[str | int]
    tags: list[str]
    save_dirs: list[str]
    mode: str
    backend: str


DEFAULT_CUSTOM_STAGE_RESOURCES = Resources(cpus=4, mem_gb=20, timeout_min=720)


CHAIN_PRESET_MODULES: dict[ChainPreset, str] = {
    ChainPreset.TS_PER_RPOS: "frust.pipelines.run_ts_per_rpos",
    ChainPreset.INT3_PER_RPOS: "frust.pipelines.run_int3_per_rpos",
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
}
