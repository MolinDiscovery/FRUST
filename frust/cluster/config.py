from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ChainPreset(StrEnum):
    TS_PER_RPOS = "ts_per_rpos"
    INT3_PER_RPOS = "int3_per_rpos"


@dataclass(frozen=True)
class Resources:
    cpus: int
    mem_gb: int | float
    timeout_min: int


@dataclass(frozen=True)
class ClusterConfig:
    backend: str = "slurm"
    partition: str | None = None
    log_dir: str | Path = "logs"
    work_dir: str | Path | None = None
    extra_slurm_parameters: dict[str, str] | None = None


@dataclass(frozen=True)
class JobSubmissionResult:
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

