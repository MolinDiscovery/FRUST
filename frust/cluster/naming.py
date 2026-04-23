from __future__ import annotations

from pathlib import Path
import re


def sanitize_tag(value: str) -> str:
    """Sanitize a string for job names, file names, and output directories.

    Parameters
    ----------
    value : str
        Raw tag value.

    Returns
    -------
    str
        Scheduler- and filesystem-friendly tag string.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "job"


def pipeline_output_parquet(out_dir: str | Path, pipeline: str, tag: str) -> str:
    """Build a parquet output path for an independent pipeline submission.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Root output directory.
    pipeline : str
        Pipeline name.
    tag : str
        Job tag.

    Returns
    -------
    str
        Full parquet output path.
    """
    return str(Path(out_dir) / f"{sanitize_tag(pipeline)}_{sanitize_tag(tag)}.parquet")


def chain_save_dir(root_out_dir: str | Path, tag: str) -> str:
    """Build the per-tag save directory for a dependent chain.

    Parameters
    ----------
    root_out_dir : str or pathlib.Path
        Root chain output directory.
    tag : str
        Job tag.

    Returns
    -------
    str
        Full save directory path for this chain tag.
    """
    return str(Path(root_out_dir) / sanitize_tag(tag))


def next_chain_parquet(current: str, stage_name: str) -> str:
    """Return the next parquet filename in a dependent stage chain.

    Parameters
    ----------
    current : str
        Current parquet filename in the chain.
    stage_name : str
        Stage function name.

    Returns
    -------
    str
        Filename expected to be produced by the given stage.
    """
    stem = current.rsplit(".", 1)[0]
    if stage_name == "run_hess":
        return f"{stem}.hess.parquet"
    if stage_name == "run_OptTS":
        return f"{stem}.optts.parquet"
    if stage_name == "run_Opt":
        return f"{stem}.opt.parquet"
    if stage_name == "run_freq":
        return f"{stem}.freq.parquet"
    if stage_name == "run_solv":
        return f"{stem}.solv.parquet"
    return current
