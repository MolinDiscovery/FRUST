from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

"""Configuration models and built-in presets for FRUST cluster submission."""


class ChainPreset(StrEnum):
    """Named dependent-stage submission presets bundled with FRUST."""

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
    stderr_to_stdout : bool, optional
        Merge stderr into stdout. Screening submissions enable this for fewer
        per-job log files; standard submissions keep separate streams.
    max_array_size : int or None, optional
        Known cluster limit on elements in one array. When provided, FRUST
        rejects larger arrays before submission. ``None`` lets Slurm validate
        its own limit; FRUST never silently splits an array.
    """

    backend: str = "slurm"
    partition: str | None = None
    log_dir: str | Path = "logs"
    work_dir: str | Path | None = None
    extra_slurm_parameters: dict[str, str] | None = None
    stderr_to_stdout: bool = False
    max_array_size: int | None = None


@dataclass(frozen=True)
class JobSubmissionResult:
    """Summary information returned after submission.

    Parameters
    ----------
    job_ids : list[str or int]
        Unique worker-job identifiers in submission order, excluding collection.
    tags : list[str]
        Target tags in selection order. In batched or staged runs these do not
        correspond one-to-one with ``job_ids``; use ``records`` for that mapping.
    save_dirs : list[str]
        Output directories associated with the selected targets.
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
    records : list of SubmissionRecord, optional
        Explicit target-to-job mappings, including stage group and attempt.
    array_job_ids : list of str, optional
        Actual Slurm array parent IDs. Ordinary jobs and local jobs have none.
    submission_path : str or None, optional
        Versioned submission metadata file; separate from scientific manifests.
    """

    job_ids: list[str | int]
    tags: list[str]
    save_dirs: list[str]
    mode: str
    backend: str
    collection_job_id: str | int | None = None
    collection_output: str | None = None
    collection_report: str | None = None
    records: list[SubmissionRecord] = field(default_factory=list)
    array_job_ids: list[str] = field(default_factory=list)
    submission_path: str | None = None


@dataclass(frozen=True)
class SubmissionRecord:
    """Map one target and stage group to its submitted worker job.

    Parameters
    ----------
    target : str
        Stable target tag.
    group : str
        Execution stage-group name, such as ``single_job`` or ``dft_freq``.
    batch_index : int
        Position of the planned job within its stage group.
    batch_targets : tuple of str
        Target tags sharing that job, in execution order.
    save_dir : str
        Target output directory.
    output_path : str
        Expected checkpoint or final parquet.
    attempt_id : str
        Unique submission attempt identity.
    status : {"planned", "submitted", "submission_failed"}
        ``planned`` has not been submitted; ``submitted`` has an executor ID;
        ``submission_failed`` encountered a submission error. These describe
        submission only, not calculation success. A scheduler acceptance error
        can be ambiguous; inspect the scheduler before resubmitting.
    job_id : str or int or None, optional
        Actual executor job ID, available after submission.
    array_job_id : str or None, optional
        Actual Slurm array parent ID; absent for ordinary or local jobs.
    array_index : int or None, optional
        Actual Slurm array element index; absent outside Slurm arrays.
    error : str or None, optional
        Submission error when status is ``submission_failed``.
    """

    target: str
    group: str
    batch_index: int
    batch_targets: tuple[str, ...]
    save_dir: str
    output_path: str
    attempt_id: str
    status: str = "planned"
    job_id: str | int | None = None
    array_job_id: str | None = None
    array_index: int | None = None
    error: str | None = None


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
    ChainPreset.SCREEN_TS_PER_RPOS: "frust.pipelines.run_screen_ts_per_rpos",
}


CHAIN_PRESET_STAGE_ORDER: dict[ChainPreset, list[str]] = {
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
    ChainPreset.SCREEN_TS_PER_RPOS: {
        "run_init": Resources(24, 20, 7200),
        "run_hess": Resources(8, 64, 7200),
        "run_OptTS": Resources(24, 20, 7200),
        "run_freq": Resources(8, 64, 7200),
        "run_solv": Resources(24, 20, 3600),
        "run_cleanup": Resources(2, 2, 60),
    },
}
