from __future__ import annotations

import inspect
from pathlib import Path

from frust.cluster.chains import submit_chain_jobs
from frust.cluster.config import (
    DEFAULT_ORCA_MEMORY_FRACTION,
    ClusterConfig,
    JobSubmissionResult,
    Resources,
    orca_memory_gb,
)
from frust.cluster.executor import create_executor, update_executor
from frust.cluster.inputs import prepare_pipeline_inputs, load_pipeline
from frust.cluster.naming import pipeline_output_parquet, sanitize_tag
from frust.tsguess2.specs import resolve_profile_specs
from frust.workflows.methods import orca
from frust.workflows.spec_profiles import geometry_key_for_calculator


def submit_jobs(
    *,
    csv_path: str | Path,
    pipeline: str,
    out_dir: str | Path,
    cluster: ClusterConfig,
    resources: Resources,
    debug: bool = False,
    production: bool = True,
    n_confs: int | None = None,
    save_output_dir: bool = True,
    dft: bool = False,
    select_mols: str | list[str] = "all",
    work_dir: str | Path | None = None,
    orca_memory_fraction: float = DEFAULT_ORCA_MEMORY_FRACTION,
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
        Accepted states are ``"dimer"``, ``"HH"``, ``"ligand"``,
        ``"catalyst"``, ``"int1"``, ``"int2"``, ``"HBpin-ligand"``, and
        ``"HBpin-mol"``; shortcuts are ``"all"``, ``"uniques"``, and
        ``"generics"``.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override. If omitted, ``cluster.work_dir`` is
        used.
    orca_memory_fraction : float, optional
        Fraction of the full Slurm allocation forwarded to ORCA-capable
        pipeline stages. Defaults to ``0.8``; Slurm still receives the full
        ``resources.mem_gb`` allocation.

    Returns
    -------
    frust.cluster.config.JobSubmissionResult
        Summary of the submitted jobs, including scheduler ids and tags.

    Raises
    ------
    ValueError
        If the pipeline name is unsupported or the CSV input is invalid.
    """
    prepared = prepare_pipeline_inputs(csv_path, pipeline, select_mols=select_mols)
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
            "mem_gb": orca_memory_gb(resources, orca_memory_fraction),
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
    spec_profile: str = "auto",
    spec_match: str = "prefer-exact",
    save_output_dir: bool = True,
    work_dir: str | Path | None = None,
    orca_memory_fraction: float = DEFAULT_ORCA_MEMORY_FRACTION,
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
    spec_profile : str, optional
        ``tsguess2`` geometry profile. ``"auto"`` selects it from the chain's
        ORCA transition-state method.
    spec_match : {"prefer-exact", "exact"}, optional
        Geometry-profile matching policy.
    save_output_dir : bool, optional
        Forwarded to initialization stages.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override. If omitted, ``cluster.work_dir`` is
        used.
    orca_memory_fraction : float, optional
        Fraction of each stage's Slurm allocation forwarded to ORCA. Defaults
        to ``0.8`` while Slurm retains the full requested allocation.

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

    resolved_profile = spec_profile
    if spec_profile == "auto":
        geometry_method = composite_method or functional or "wB97X-D3"
        geometry_basis = None if composite_method else basisset or "6-31G**"
        resolved_profile = geometry_key_for_calculator(
            orca(method=geometry_method, basis=geometry_basis, job="optts")
        ).profile_id
    resolve_profile_specs(ts_types, resolved_profile, match=spec_match)

    return submit_chain_jobs(
        csv_path=csv_path,
        preset="screen_ts_per_rpos",
        module_path=None,
        stage_order=None,
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
        spec_profile=resolved_profile,
        spec_match=spec_match,
        save_output_dir=save_output_dir,
        work_dir=work_dir,
        orca_memory_fraction=orca_memory_fraction,
    )
