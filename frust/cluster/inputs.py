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
    """Load and validate a CSV input table for cluster submission.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe containing at least a ``smiles`` column.

    Raises
    ------
    ValueError
        If the file is missing, the ``smiles`` column is absent, or that
        column contains missing values.
    """
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
    """Load a supported pipeline function from :mod:`frust.pipes`.

    Parameters
    ----------
    pipeline : str
        Pipeline function name.

    Returns
    -------
    callable
        Callable pipeline function from :mod:`frust.pipes`.

    Raises
    ------
    ValueError
        If the pipeline is not one of the supported submission targets.
    """
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
    """Prepare submission payloads for independent FRUST jobs.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        CSV input containing at least a ``smiles`` column.
    pipeline : str
        Supported pipeline name from :mod:`frust.pipes`.
    ts_xyz : str or pathlib.Path or None, optional
        TS template file for TS-dependent pipelines.
    select_mols : str or list[str], optional
        Molecule selection passthrough for molecule workflows.

    Returns
    -------
    dict
        Dictionary containing ``mode``, ``payloads``, ``tags``, and the loaded
        input ``dataframe``.
    """
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
    """Prepare TS payloads for dependent chain submission.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        CSV input containing at least a ``smiles`` column.
    preset : str
        Chain preset label used for reporting the prepared mode.
    ts_xyz : str or pathlib.Path
        TS template file used to generate stage inputs.

    Returns
    -------
    dict
        Dictionary containing ``mode``, ``payloads``, ``tags``, and the loaded
        input ``dataframe``.
    """
    df = load_csv_input(csv_path)
    ts_jobs = create_ts_per_rpos(df, str(ts_xyz), return_format="list")
    tags = [sanitize_tag(list(job.keys())[0]) for job in ts_jobs]
    return {"mode": preset, "payloads": ts_jobs, "tags": tags, "dataframe": df}
