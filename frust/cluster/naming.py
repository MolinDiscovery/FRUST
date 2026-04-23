from __future__ import annotations

from pathlib import Path
import re


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "job"


def pipeline_output_parquet(out_dir: str | Path, pipeline: str, tag: str) -> str:
    return str(Path(out_dir) / f"{sanitize_tag(pipeline)}_{sanitize_tag(tag)}.parquet")


def chain_save_dir(root_out_dir: str | Path, tag: str) -> str:
    return str(Path(root_out_dir) / sanitize_tag(tag))


def next_chain_parquet(current: str, stage_name: str) -> str:
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

