from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from frust.cluster.config import (
    CHAIN_PRESET_MODULES,
    CHAIN_PRESET_RESOURCES,
    CHAIN_PRESET_STAGE_ORDER,
    ChainPreset,
    ClusterConfig,
    DEFAULT_CUSTOM_STAGE_RESOURCES,
    JobSubmissionResult,
    Resources,
)
from frust.cluster.executor import create_executor, update_executor_with_dependency
from frust.cluster.inputs import prepare_chain_inputs
from frust.cluster.naming import chain_save_dir, next_chain_parquet, sanitize_tag


def _resolve_chain_definition(
    *,
    preset: str | None,
    module_path: str | None,
    stage_order: list[str] | None,
):
    """Resolve preset or custom chain configuration into one internal shape.

    Parameters
    ----------
    preset : str or None
        Built-in chain preset.
    module_path : str or None
        Custom module path for advanced chains.
    stage_order : list[str] or None
        Explicit stage order for custom chains.

    Returns
    -------
    dict
        Dictionary containing resolved preset metadata, module path, stage
        order, and default resources.
    """
    if preset is not None and (module_path is not None or stage_order is not None):
        raise ValueError("Use either `preset` or `module_path`/`stage_order`, not both")
    if preset is None and (module_path is None or stage_order is None):
        raise ValueError("Custom chain mode requires both `module_path` and `stage_order`")

    if preset is not None:
        preset_enum = ChainPreset(preset)
        return {
            "preset": preset_enum,
            "module_path": CHAIN_PRESET_MODULES[preset_enum],
            "stage_order": CHAIN_PRESET_STAGE_ORDER[preset_enum],
            "resource_defaults": CHAIN_PRESET_RESOURCES[preset_enum],
        }

    return {
        "preset": None,
        "module_path": module_path,
        "stage_order": stage_order,
        "resource_defaults": {},
    }


def _resolve_stage_functions(module_path: str, stage_order: list[str]):
    """Load and validate stage callables from a chain module.

    Parameters
    ----------
    module_path : str
        Python module path containing stage functions.
    stage_order : list[str]
        Explicit stage names to load from the module.

    Returns
    -------
    dict[str, callable]
        Mapping of stage name to validated callable.
    """
    mod = importlib.import_module(module_path)
    funcs = {}
    for stage_name in stage_order:
        if not hasattr(mod, stage_name):
            raise ValueError(f"Stage {stage_name!r} not found in module {module_path!r}")
        fn = getattr(mod, stage_name)
        if not callable(fn):
            raise ValueError(f"Stage {stage_name!r} in module {module_path!r} is not callable")
        funcs[stage_name] = fn
    return funcs


def submit_chain_jobs(
    *,
    csv_path: str | Path,
    preset: str | None,
    module_path: str | None,
    stage_order: list[str] | None,
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
    """Submit a dependent stage chain through submitit.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        CSV input containing at least a ``smiles`` column.
    preset : str or None
        Built-in FRUST chain preset.
    module_path : str or None
        Custom stage module path for advanced chains.
    stage_order : list[str] or None
        Explicit stage order for custom chains.
    ts_xyz : str or pathlib.Path
        TS template file used to generate initial stage inputs.
    out_dir : str or pathlib.Path
        Root output directory for the chain submission.
    cluster : frust.cluster.config.ClusterConfig
        Shared executor configuration.
    stage_resources : dict[str, Resources] or None, optional
        Optional per-stage resource overrides.
    debug : bool, optional
        Forwarded to stage functions when supported.
    production : bool, optional
        Preserve stage defaults when ``True`` and ``n_confs`` is ``None``.
    n_confs : int or None, optional
        Conformer count forwarded to initialization stages.
    functional : str or None, optional
        ORCA functional override forwarded to preset stage modules when they
        accept it.
    basisset : str or None, optional
        ORCA gas-phase basis set override forwarded to preset stage modules
        when they accept it.
    basisset_solv : str or None, optional
        ORCA solvent single-point basis set override forwarded to preset stage
        modules when they accept it.
    save_output_dir : bool, optional
        Forwarded to initialization stages when supported.
    work_dir : str or pathlib.Path or None, optional
        Optional work directory override.

    Returns
    -------
    frust.cluster.config.JobSubmissionResult
        Summary of all submitted stage jobs for all prepared tags.
    """
    resolved = _resolve_chain_definition(
        preset=preset,
        module_path=module_path,
        stage_order=stage_order,
    )
    stage_funcs = _resolve_stage_functions(resolved["module_path"], resolved["stage_order"])
    prepared = prepare_chain_inputs(csv_path, preset or "custom", ts_xyz)

    root_out_dir = Path(out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)
    executor = create_executor(cluster)
    job_ids: list[str | int] = []
    tags: list[str] = []
    save_dirs: list[str] = []

    for ts_struct, raw_tag in zip(prepared["payloads"], prepared["tags"]):
        tag = sanitize_tag(raw_tag)
        save_dir = chain_save_dir(root_out_dir, tag)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        last_job = None
        current_parquet = "init.parquet"
        tags.append(tag)
        save_dirs.append(save_dir)

        for stage_name in resolved["stage_order"]:
            fn = stage_funcs[stage_name]
            sig = inspect.signature(fn)
            resources = (stage_resources or {}).get(
                stage_name,
                resolved["resource_defaults"].get(stage_name, DEFAULT_CUSTOM_STAGE_RESOURCES),
            )
            update_executor_with_dependency(
                executor,
                cluster,
                resources,
                job_name=f"{tag}_{stage_name}",
                dependency_job_id=getattr(last_job, "job_id", None),
            )

            kwargs = {"save_dir": save_dir, "work_dir": work_dir or cluster.work_dir, "debug": debug}
            if functional is not None:
                kwargs["functional"] = functional
            if basisset is not None:
                kwargs["basisset"] = basisset
            if basisset_solv is not None:
                kwargs["basisset_solv"] = basisset_solv
            if stage_name == "run_init":
                kwargs.update(
                    {
                        "ts_struct": ts_struct,
                        "n_confs": None if production and n_confs is None else n_confs,
                        "n_cores": resources.cpus,
                        "mem_gb": resources.mem_gb,
                        "save_output_dir": save_output_dir,
                    }
                )
            elif stage_name == "run_cleanup":
                kwargs = {"save_dir": save_dir}
            else:
                kwargs.update(
                    {
                        "parquet_path": current_parquet,
                        "n_cores": resources.cpus,
                        "mem_gb": resources.mem_gb,
                    }
                )

            call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            job = executor.submit(fn, **call_kwargs)
            job_ids.append(getattr(job, "job_id", f"{tag}_{stage_name}"))
            last_job = job
            current_parquet = next_chain_parquet(current_parquet, stage_name)

    print("Submitted job IDs:", job_ids)
    return JobSubmissionResult(
        job_ids=job_ids,
        tags=tags,
        save_dirs=save_dirs,
        mode=preset or "custom",
        backend=cluster.backend,
    )
