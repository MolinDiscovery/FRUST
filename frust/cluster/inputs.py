from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd

from frust.cluster.naming import sanitize_tag
from frust.utils.mols import create_ts_per_rpos


TS_PIPELINES = {
    "run_ts_per_rpos",
    "run_ts_per_rpos_UMA",
    "run_ts_per_rpos_UMA_short",
    "run_orca_smoke_test",
}

DATAFRAME_PIPELINES = {
    "run_mols",
    "run_ts_per_lig",
}

SUPPORTED_PIPELINES = TS_PIPELINES | DATAFRAME_PIPELINES


def load_csv_input(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column")
    if df["smiles"].isna().any():
        raise ValueError("Input CSV contains missing values in the 'smiles' column")
    return df


def load_pipeline(pipeline: str):
    if pipeline not in SUPPORTED_PIPELINES:
        supported = ", ".join(sorted(SUPPORTED_PIPELINES))
        raise ValueError(f"Unknown pipeline {pipeline!r}. Supported pipelines: {supported}")
    pipes_mod = importlib.import_module("frust.pipes")
    return getattr(pipes_mod, pipeline)


def prepare_pipeline_inputs(
    csv_path: str | Path,
    pipeline: str,
    ts_xyz: str | Path | None = None,
    select_mols: str | list[str] = "all",
):
    if pipeline not in SUPPORTED_PIPELINES:
        supported = ", ".join(sorted(SUPPORTED_PIPELINES))
        raise ValueError(f"Unknown pipeline {pipeline!r}. Supported pipelines: {supported}")

    df = load_csv_input(csv_path)

    if pipeline in TS_PIPELINES:
        if ts_xyz is None:
            raise ValueError(f"`ts_xyz` is required for pipeline {pipeline!r}")
        if pipeline == "run_orca_smoke_test":
            jobs = [{"smoke_test": ("placeholder", [0], "placeholder")}]
            tags = [sanitize_tag("smoke_test")]
            return {"mode": "ts", "payloads": jobs, "tags": tags, "dataframe": df}

        ts_jobs = create_ts_per_rpos(df, str(ts_xyz), return_format="list")
        tags = [sanitize_tag(list(job.keys())[0]) for job in ts_jobs]
        return {"mode": "ts", "payloads": ts_jobs, "tags": tags, "dataframe": df}

    if pipeline == "run_mols":
        return {"mode": "dataframe", "payloads": [df], "tags": [sanitize_tag("mols")], "dataframe": df}

    if pipeline == "run_ts_per_lig":
        if ts_xyz is None:
            raise ValueError(f"`ts_xyz` is required for pipeline {pipeline!r}")
        return {"mode": "dataframe", "payloads": [df], "tags": [sanitize_tag("ts_per_lig")], "dataframe": df}

    raise ValueError(f"Unsupported combination for pipeline {pipeline!r}")


def prepare_chain_inputs(csv_path: str | Path, preset: str, ts_xyz: str | Path):
    df = load_csv_input(csv_path)
    ts_jobs = create_ts_per_rpos(df, str(ts_xyz), return_format="list")
    tags = [sanitize_tag(list(job.keys())[0]) for job in ts_jobs]
    return {"mode": preset, "payloads": ts_jobs, "tags": tags, "dataframe": df}

