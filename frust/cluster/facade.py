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
    """Submit independent FRUST workflow jobs from a CSV input file.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a CSV file containing at least a ``smiles`` column.
    pipeline : str
        High-level pipeline name from :mod:`frust.pipes`.
    out_dir : str or pathlib.Path
        Output directory under which parquet files and run outputs are written.
    cluster : frust.cluster.config.ClusterConfig
        Shared cluster or local-executor configuration.
    resources : frust.cluster.config.Resources
        CPU, memory, and timeout settings for every submitted job in this
        submission call.
    ts_xyz : str or pathlib.Path or None, optional
        TS template XYZ file required by TS-dependent pipelines.
    debug : bool, optional
        Forwarded to the selected FRUST pipeline.
    production : bool, optional
        If ``True`` and ``n_confs`` is ``None``, preserve the pipeline default
        conformer behavior.
    n_confs : int or None, optional
        Conformer count forwarded to the selected pipeline when supported.
    save_output_dir : bool, optional
        Forwarded to the selected FRUST pipeline.
    dft : bool, optional
        Forwarded to the selected FRUST pipeline as ``DFT`` when supported.
    select_mols : str or list[str], optional
        Molecule selection forwarded to molecule workflows when supported.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override. If omitted, ``cluster.work_dir`` is
        used.

    Returns
    -------
    frust.cluster.config.JobSubmissionResult
        Summary of the submitted jobs, including scheduler ids and tags.

    Raises
    ------
    ValueError
        If the pipeline name is unsupported, if required TS inputs are
        missing, or if the CSV input is invalid.
    """
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
        elif pipeline == "run_mols_per_rpos":
            kwargs["mol_struct"] = payload
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
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    save_output_dir: bool = True,
    work_dir: str | Path | None = None,
) -> JobSubmissionResult:
    """Submit a dependent stage chain from a CSV input file.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a CSV file containing at least a ``smiles`` column.
    preset : str or None, optional
        Built-in FRUST chain preset, such as ``"ts_per_rpos"`` or
        ``"int3_per_rpos"``. Use either ``preset`` or the custom
        ``module_path``/``stage_order`` combination.
    module_path : str or None, optional
        Custom Python module containing stage functions for advanced use.
    stage_order : list[str] or None, optional
        Explicit stage order for custom chains.
    ts_xyz : str or pathlib.Path
        TS template XYZ file used to prepare dependent chain inputs.
    out_dir : str or pathlib.Path
        Root directory under which per-tag stage outputs are written.
    cluster : frust.cluster.config.ClusterConfig
        Shared cluster or local-executor configuration.
    stage_resources : dict[str, Resources] or None, optional
        Optional per-stage resource overrides.
    debug : bool, optional
        Forwarded to the stage functions when supported.
    production : bool, optional
        If ``True`` and ``n_confs`` is ``None``, preserve the stage default
        conformer behavior.
    n_confs : int or None, optional
        Conformer count forwarded to initialization stages when supported.
    functional : str or None, optional
        ORCA functional override for preset stage modules. If omitted, the
        preset module defaults are used.
    basisset : str or None, optional
        ORCA basis set override for gas-phase preset stages. If omitted, the
        preset module defaults are used.
    basisset_solv : str or None, optional
        ORCA basis set override for solvent single-point preset stages. If
        omitted, the preset module defaults are used.
    save_output_dir : bool, optional
        Forwarded to initialization stages when supported.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override. If omitted, ``cluster.work_dir`` is
        used.

    Returns
    -------
    frust.cluster.config.JobSubmissionResult
        Summary of the submitted chain jobs.
    """
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
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )


def submit_screen_chain(
    *,
    csv_path: str | Path,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    out_dir: str | Path,
    cluster: ClusterConfig,
    stage_resources: dict[str, Resources] | None = None,
    debug: bool = False,
    production: bool = True,
    n_confs: int | None = None,
    top_n: int = 10,
    functional: str | None = None,
    basisset: str | None = None,
    basisset_solv: str | None = None,
    composite_method: str | None = None,
    save_output_dir: bool = True,
    work_dir: str | Path | None = None,
) -> JobSubmissionResult:
    """Submit a screen-based TS chain for substrate/catalyst systems.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Screen CSV containing ``role`` and ``smiles`` columns. Substrate rows
        may include ``rpos``; catalyst rows are paired with every substrate.
    ts_types : tuple or list of str, optional
        Transition-state types to submit. Defaults to TS1-TS4.
    out_dir : str or pathlib.Path
        Root directory under which per-target stage outputs are written.
    cluster : frust.cluster.config.ClusterConfig
        Shared cluster or local-executor configuration.
    stage_resources : dict[str, Resources] or None, optional
        Optional per-stage resource overrides.
    debug : bool, optional
        Forwarded to stage functions.
    production : bool, optional
        If ``True`` and ``n_confs`` is ``None``, preserve the screen TS guess
        module's automatic conformer-count behavior.
    n_confs : int or None, optional
        Conformer count generated inside the initialization stage. ``None``
        selects the legacy rotatable-bond heuristic.
    top_n : int, optional
        Number of low-energy xTB conformers retained before DFT filtering.
    functional : str or None, optional
        ORCA functional override for preset stage modules.
    basisset : str or None, optional
        ORCA gas-phase basis set override.
    basisset_solv : str or None, optional
        ORCA solvent single-point basis set override.
    composite_method : str or None, optional
        Complete ORCA composite-method keyword, such as ``"r2SCAN-3c"``. When
        provided, no separate basis set keywords are forwarded. Mutually
        exclusive with ``functional``, ``basisset``, and ``basisset_solv``.
    save_output_dir : bool, optional
        Forwarded to initialization stages.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override. If omitted, ``cluster.work_dir`` is
        used.

    Returns
    -------
    frust.cluster.config.JobSubmissionResult
        Summary of the submitted screen-chain jobs.
    """
    if composite_method is not None:
        conflicting = [
            name
            for name, value in (
                ("functional", functional),
                ("basisset", basisset),
                ("basisset_solv", basisset_solv),
            )
            if value is not None
        ]
        if conflicting:
            joined = ", ".join(f"`{name}`" for name in conflicting)
            raise ValueError(
                "`composite_method` cannot be combined with "
                f"{joined}; ORCA composite methods already include their basis/corrections."
            )

    return submit_chain_jobs(
        csv_path=csv_path,
        preset="screen_ts_per_rpos",
        module_path=None,
        stage_order=None,
        ts_xyz=None,
        ts_types=ts_types,
        out_dir=out_dir,
        cluster=cluster,
        stage_resources=stage_resources,
        debug=debug,
        production=production,
        n_confs=n_confs,
        top_n=top_n,
        functional=functional,
        basisset=basisset,
        basisset_solv=basisset_solv,
        composite_method=composite_method,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
    )
