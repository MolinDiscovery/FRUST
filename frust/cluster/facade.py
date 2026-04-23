from __future__ import annotations

import inspect
from pathlib import Path

from frust.cluster.chains import submit_chain_jobs
from frust.cluster.config import ClusterConfig, JobSubmissionResult, Resources
from frust.cluster.executor import create_executor, update_executor
from frust.cluster.inputs import prepare_pipeline_inputs, load_pipeline
from frust.cluster.naming import pipeline_output_parquet, sanitize_tag


def submit_jobs(
    *,
    csv_path: str | Path,
    pipeline: str,
    out_dir: str | Path,
    cluster: ClusterConfig,
    resources: Resources,
    ts_xyz: str | Path | None = None,
    debug: bool = False,
    production: bool = True,
    n_confs: int | None = None,
    save_output_dir: bool = True,
    dft: bool = False,
    select_mols: str | list[str] = "all",
    work_dir: str | Path | None = None,
) -> JobSubmissionResult:
    prepared = prepare_pipeline_inputs(csv_path, pipeline, ts_xyz=ts_xyz, select_mols=select_mols)
    pipeline_fn = load_pipeline(pipeline)
    sig = inspect.signature(pipeline_fn)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    executor = create_executor(cluster)
    job_ids: list[str | int] = []
    tags: list[str] = []
    save_dirs: list[str] = []

    for payload, raw_tag in zip(prepared["payloads"], prepared["tags"]):
        tag = sanitize_tag(raw_tag)
        update_executor(executor, cluster, resources, job_name=f"{sanitize_tag(pipeline)}_{tag}")
        output_parquet = pipeline_output_parquet(out_path, pipeline, tag)

        kwargs = {
            "n_confs": None if production and n_confs is None else n_confs,
            "n_cores": resources.cpus,
            "mem_gb": resources.mem_gb,
            "debug": debug,
            "out_dir": str(out_path),
            "output_parquet": output_parquet,
            "save_output_dir": save_output_dir,
            "DFT": dft,
            "select_mols": select_mols,
            "work_dir": work_dir or cluster.work_dir,
        }

        if pipeline == "run_mols":
            kwargs["ligand_smiles_df"] = payload
        elif pipeline == "run_ts_per_lig":
            kwargs["ligand_smiles_df"] = payload
            kwargs["ts_guess_xyz"] = str(ts_xyz)
        elif pipeline in {"run_ts_per_rpos", "run_ts_per_rpos_UMA", "run_ts_per_rpos_UMA_short", "run_orca_smoke_test"}:
            kwargs["ts_struct"] = payload
        else:
            raise ValueError(f"Unsupported pipeline {pipeline!r}")

        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        job = executor.submit(pipeline_fn, **call_kwargs)
        job_ids.append(getattr(job, "job_id", f"{pipeline}_{tag}"))
        tags.append(tag)
        save_dirs.append(str(out_path))

    print("Submitted job IDs:", job_ids)
    return JobSubmissionResult(
        job_ids=job_ids,
        tags=tags,
        save_dirs=save_dirs,
        mode=pipeline,
        backend=cluster.backend,
    )


def submit_chain(
    *,
    csv_path: str | Path,
    preset: str | None = None,
    module_path: str | None = None,
    stage_order: list[str] | None = None,
    ts_xyz: str | Path,
    out_dir: str | Path,
    cluster: ClusterConfig,
    stage_resources: dict[str, Resources] | None = None,
    debug: bool = False,
    production: bool = True,
    n_confs: int | None = None,
    save_output_dir: bool = True,
    work_dir: str | Path | None = None,
) -> JobSubmissionResult:
    return submit_chain_jobs(
        csv_path=csv_path,
        preset=preset,
        module_path=module_path,
        stage_order=stage_order,
        ts_xyz=ts_xyz,
        out_dir=out_dir,
        cluster=cluster,
        stage_resources=stage_resources,
        debug=debug,
        production=production,
        n_confs=n_confs,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )

