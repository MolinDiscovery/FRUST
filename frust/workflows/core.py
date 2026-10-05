"""Shared execution engine for FRUST workflow objects.

This module deliberately contains no chemistry-specific target expansion. A
concrete workflow in :mod:`frust.workflows.factories` supplies lightweight
``WorkflowTarget`` objects and an ordered list of ``StageDef`` objects; this
module turns them into local ``Stepper`` calls, submitted cluster jobs, and
collected parquet outputs.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from frust.artifacts import (
    ArtifactPolicy,
    compact_result_dataframe,
    validate_artifact_policy,
)
from frust.cluster.config import (
    DEFAULT_ORCA_MEMORY_FRACTION,
    ClusterConfig,
    JobSubmissionResult,
    Resources,
    orca_memory_gb,
)
from frust.cluster.executor import (
    create_executor,
    update_executor_with_dependencies,
    update_executor_with_dependency,
)
from frust.cluster.submission import (
    _plan_submission, _submit_array_jobs,
    _validate_array_scheduler_options, _validate_array_size, _atomic_write_submission_json,
    _submission_guard, _mutation_lock, _mark_completed, _target_outcomes, _submission_history,
    _ensure_collection_ready,
    _stage_outcome_path, _array_dependency, _submit_local_stages,
)
from frust.cluster.naming import sanitize_tag
from frust.results import ResultProfile, attach_result_contract
from frust.schema import (
    canonical_state_columns,
    normal_termination_columns,
    normalize_dataframe,
    output_column,
    stamp_schema,
)
from frust.stepper import Stepper, make_stepper_logger
from frust.utils.dataframes import lowest_energy_rows, merge_dataframe_attrs
from frust.utils.timing import (
    append_workflow_timing,
    build_workflow_timing_record,
    elapsed_seconds,
    monotonic_seconds,
    utc_timestamp,
    write_timing_sidecar,
)
from frust.utils.stage_summaries import conformer_generation_summary, filter_summary
from frust.utils.uma import uma_job_server_scope
from frust.workflows.diagnostics import _collection_failure_summary
from frust.workflows.methods import CalculatorSpec, MethodPlan, preset as method_preset

ExecutionMode = Literal["single_job", "dft_staged", "fully_staged"]
TargetRetention = Literal["compact_success", "all"]
DEFAULT_WORKFLOW_RESOURCES = Resources(cpus=4, mem_gb=20, timeout_min=720)
DEFAULT_COLLECTION_RESOURCES = Resources(cpus=2, mem_gb=4, timeout_min=120)
TARGET_TIMING_FILE = "timing.json"
ANALYSIS_TIER_FILES = {
    "low_cost": "tier_low_cost.parquet",
    "uma_ranked": "tier_uma_ranked.parquet",
    "dft_ranked": "tier_dft_ranked.parquet",
}
_LEGACY_STAGE_RESOURCE_KEYS = {
    "dft_rank_sp": "dft_pre_sp",
    "dft_preopt": "dft_pre_opt",
    "dft_hessian": "hess",
    "dft_ts_opt": "optts",
    "dft_freq": "freq",
    "dft_solv_sp": "solv",
}


@dataclass(frozen=True)
class WorkflowTarget:
    """One scientific target produced by a workflow.

    Parameters
    ----------
    tag : str
        Stable filesystem- and scheduler-safe target tag.
    payload : object
        Serializable target payload used by the workflow preparation stage.
    metadata : dict, optional
        Lightweight target metadata for inspection.

    Notes
    -----
    A target is intentionally lightweight. ``wf.targets()`` should be safe to
    call for inspection and scheduling because expensive embedding and
    calculator work belongs in ``_prepare_initial_df(...)`` during
    ``wf.run(...)`` or inside a submitted job.
    """

    tag: str
    payload: Any
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageDef:
    """One typed stage in a FRUST workflow graph.

    Parameters
    ----------
    id : str
        Stable stage identifier. This is normally the key used to look up the
        stage's :class:`frust.workflows.methods.CalculatorSpec` from the active
        method plan.
    name : str
        Human-readable calculation label shown by ``wf.show_stages()``.
        Canonical dataframe columns always use ``id`` as their prefix.
    kind : {"prepare", "calc", "filter", "prune"}, optional
        Stage kind. ``"prepare"`` creates the initial dataframe, ``"calc"``
        dispatches to a calculator, ``"filter"`` keeps lowest-energy rows, and
        ``"prune"`` removes geometrically redundant conformers.
    method_stage : str, optional
        Alternate method-plan key. Use this only when a stage should reuse the
        calculator settings from another stage id.
    constraint : bool, optional
        Whether constrained calculator input should be generated from dataframe
        constraint columns where supported.
    lowest : int, optional
        Number of rows to keep after this stage, grouped by FRUST structure
        identity.
    rank_by : str or None, optional
        Canonical stage id whose electronic energy controls selection. For a
        calculator stage this normally matches ``id``; filters must set it
        explicitly so selection never depends on dataframe column order.
    n_cores : int, optional
        Stage-local calculator core count forwarded to Stepper. Submission
        resources still control the scheduler allocation.
    read_files : list of str, optional
        Files that ORCA should read from the previous saved calculation output.
    use_last_hess : bool, optional
        Whether ORCA should reuse the previous Hessian when supported.
    save_files : list of str, optional
        Extra files to preserve from the stage output directory.
    prune_options : dict, optional
        Options forwarded to :meth:`frust.stepper.Stepper.prune_conformers`
        when ``kind="prune"``.
    """

    id: str
    name: str
    kind: Literal["prepare", "calc", "filter", "prune"] = "calc"
    method_stage: str | None = None
    constraint: bool = False
    lowest: int | None = None
    rank_by: str | None = None
    n_cores: int | None = None
    read_files: list[str] | None = None
    use_last_hess: bool = False
    save_files: list[str] | None = None
    prune_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    """Runtime options passed to workflow stage execution.

    Parameters
    ----------
    n_cores : int, optional
        Core count used by embedding and by calculators unless a ``StageDef``
        overrides its calculator-level core count.
    mem_gb : int, optional
        Memory in GB forwarded to Stepper and calculator backends.
    debug : bool, optional
        Debug flag forwarded to workflow preparation and calculator stages.
    save_output_dir : bool, optional
        Whether calculator output directories should be retained by Stepper.
    work_dir : str, optional
        Scratch/work directory used by calculator backends.
    artifact_policy : {"standard", "screening"}, optional
        Scientific artifact retention policy.
    uma_oet_tools : str or None, optional
        OET runtime selected inside a job that contains UMA stages. This lets
        the submitted job use a pinned runtime without changing the login
        process or the user's global ``OET_TOOLS`` setting.
    """

    n_cores: int = 4
    mem_gb: int = 20
    debug: bool = False
    save_output_dir: bool = True
    work_dir: str | None = None
    artifact_policy: ArtifactPolicy = "standard"
    uma_oet_tools: str | None = None


@dataclass(frozen=True)
class WorkflowJobResult:
    """Small Submitit result for a completed target or stage group."""

    target_id: str
    output_path: str
    row_count: int
    status: str


@dataclass(frozen=True)
class CollectionJobResult:
    """Small Submitit result for a completed collection job."""

    output_path: str
    report_path: str
    row_count: int
    status: str


class BaseWorkflow:
    """Base class for local and cluster FRUST workflows.

    Parameters
    ----------
    method : MethodPlan or str or None, optional
        Calculator plan for all workflow stages. Use ``None`` for the default
        ``"wb97xd3-631g"`` preset, a string preset name resolved with
        :func:`frust.workflows.methods.preset`, or a custom
        :class:`frust.workflows.methods.MethodPlan`. Built-in preset strings
        are ``"r2scan-3c"`` (ORCA r2SCAN-3c composite DFT stages),
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow),
        ``"r2scan-3c-solv"`` and ``"wb97xd3-631g-solv"``
        (solvent-inclusive DFT stages without a terminal solvent SP), and
        ``"r2scan-def2svp"`` (ORCA R2SCAN/def2-SVP DFT stages).
    n_confs : int or None, optional
        Conformer count forwarded to the workflow's initial dataframe
        preparation. ``None`` lets the relevant FRUST builder choose its
        heuristic count.
    top_n : int, optional
        Number of conformers/rows kept by stages that rank and filter.
    dft : bool, optional
        Whether the concrete workflow includes DFT stages.

    Notes
    -----
    Subclasses provide chemistry by implementing ``_build_targets()``,
    ``_prepare_initial_df(...)``, ``_stage_defs()``, and optionally
    ``_step_type_for_target(...)``. This base class owns target selection,
    calculation-free preview, stage grouping, local execution, cluster
    submission, and result collection.
    """

    workflow_name = "workflow"
    result_profile: ResultProfile | None = None

    def __init__(
        self,
        *,
        method: MethodPlan | str | None = None,
        n_confs: int | None = None,
        top_n: int = 10,
        dft: bool = False,
    ) -> None:
        self.method = _coerce_method(method)
        self.n_confs = n_confs
        self.top_n = top_n
        self.dft = dft
        self._target_cache: list[WorkflowTarget] | None = None

    def targets(self) -> list[WorkflowTarget]:
        """Return the workflow's scientific targets.

        Returns
        -------
        list of WorkflowTarget
            Cached target objects. Each target has a scheduler-safe ``tag``, a
            serializable ``payload`` used by the first stage, and optional
            metadata for inspection.

        Notes
        -----
        This method must not run calculators or expensive embedding. It is used
        for inspection, target indexing, and cluster submission planning.
        """
        if self._target_cache is None:
            self._target_cache = self._build_targets()
        return list(self._target_cache)

    def preview(
        self,
        *,
        n_confs: int | None = 1,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None = None,
        n_cores: int = 1,
    ) -> pd.DataFrame:
        """Generate selected target structures without running calculations.

        Parameters
        ----------
        n_confs : int or None, optional
            Number of embedded conformers per selected target. If ``None``,
            use the structure builder's conformer-count heuristic.
        targets : iterable of target objects or int or None, optional
            Typed structure targets to preview. Integers select positions from
            ``wf.targets()``. If omitted, all targets are generated.
        n_cores : int, optional
            RDKit embedding threads.

        Returns
        -------
        pandas.DataFrame
            Canonical embedded structures containing ``system_name``,
            ``state_id``, ``state_kind``, ``rpos``, ``atoms``, and
            ``coords_embedded``. No xTB or DFT stages are run.

        Examples
        --------
        >>> planned = wf.targets()
        >>> preview = wf.preview(n_confs=1, targets=[0])

        Notes
        -----
        Preview is available for workflows backed by typed
        :class:`frust.structures.StructureTarget` objects. It delegates to the
        same target builder used by ``run()`` and therefore preserves
        preview/run structure-generation parity.
        """
        from frust.structures.api import _create_from_targets
        from frust.structures.models import StructureTarget

        selected = self._select_targets(targets)
        if not all(isinstance(target, StructureTarget) for target in selected):
            raise TypeError(
                f"{self.workflow_name}.preview() requires typed StructureTarget "
                "objects; use the modern per_rpos/tsguess2 workflow path"
            )
        return _create_from_targets(
            selected,
            n_confs=n_confs,
            n_cores=n_cores,
            source=f"frust.workflows.{self.workflow_name}.preview",
            **self._structure_build_kwargs(),
        )

    def _structure_build_kwargs(self) -> dict[str, Any]:
        """Return workflow-specific options for typed structure construction.

        Returns
        -------
        dict
            Keyword arguments forwarded to the shared structure builder.
        """
        return {}

    def show_stages(
        self,
        execution: ExecutionMode | None = None,
        detail: str = "summary",
    ) -> pd.DataFrame:
        """Return the active workflow stage graph as a compact dataframe.

        Parameters
        ----------
        execution : {"single_job", "dft_staged", "fully_staged"} or None, optional
            Execution grouping to inspect. ``None`` uses the same default as
            ``submit(...)``: DFT workflows use ``"dft_staged"`` and non-DFT
            workflows use ``"single_job"``. ``"single_job"`` runs all stages in
            one job per target, ``"dft_staged"`` keeps initialization together
            and splits DFT stages into dependent jobs, and ``"fully_staged"``
            splits every stage into its own dependent job.
        detail : {"summary", "full"}, optional
            Level of planned configuration to return. ``"summary"`` keeps the
            compact stage table. ``"full"`` additionally includes raw
            calculator input blocks, calculator keyword arguments, file reuse,
            Hessian reuse, saved files, and complete pruning options. Multiline
            input blocks use literal ``\\n`` separators so Markdown tables stay
            readable.

        Returns
        -------
        pandas.DataFrame
            One row per active workflow stage. Important columns are ``group``
            for the scheduler/resource group, ``stage`` for the workflow stage
            id, ``method_key`` for the calculator key read from
            ``method.stages``, ``engine`` for the calculator backend, and
            ``options`` for the compact calculator keywords. ``solvent`` shows
            solvent settings such as ``"SMD(chloroform)"``. The table describes
            the method-plan keys this workflow will actually use; it does not
            list unused entries from ``method.stages`` and does not build
            targets, embed structures, or run calculators.

            Full detail describes the planned configuration. Runtime-derived
            values such as rendered coordinates, resource-derived memory, and
            resolved executable paths are recorded by the executed calculation
            rather than this planning table.

        Examples
        --------
        Inspect the stage groups before deciding ``stage_resources``:

        >>> import frust as ft
        >>> wf = ft.workflows.raw_mols(csv_path="raw_dimers.csv", method="r2scan-3c", dft=True)
        >>> wf.show_stages()[["group", "stage", "engine"]]
        >>> wf.show_stages(detail="full")[["stage", "xtra_inp_str", "read_files"]]
        """
        if detail not in {"summary", "full"}:
            raise ValueError("detail must be 'summary' or 'full'")

        mode = execution or ("dft_staged" if self.dft else "single_job")
        rows: list[dict[str, Any]] = []
        for group in self._stage_groups(mode):
            group_name = (
                "single_job" if mode == "single_job" else self._group_name(group)
            )
            for stage in group:
                method_key = stage.method_stage or stage.id
                spec: CalculatorSpec | None = None
                if stage.kind == "calc":
                    spec = self.method.for_stage(method_key)
                options_text = (
                    _format_pruning_stage_options(stage.prune_options)
                    if stage.kind == "prune"
                    else _format_stage_options(
                        spec.options if spec is not None else None
                    )
                )

                row = {
                    "group": group_name,
                    "stage": stage.id,
                    "calculation": stage.name,
                    "kind": stage.kind,
                    "method_key": method_key if stage.kind == "calc" else None,
                    "engine": _stage_engine(stage, spec),
                    "options": options_text,
                    "solvent": _format_stage_solvent(spec),
                    "lowest": stage.lowest,
                    "rank_by": stage.rank_by,
                    "constraint": stage.constraint,
                    "n_cores": stage.n_cores,
                }
                if detail == "full":
                    row.update(
                        {
                            "detailed_inp_str": _format_planned_input(
                                spec.detailed_inp_str if spec is not None else None
                            ),
                            "xtra_inp_str": _format_planned_input(
                                spec.xtra_inp_str if spec is not None else None
                            ),
                            "calculator_kwargs": _format_planned_mapping(
                                spec.kwargs if spec is not None else None
                            ),
                            "read_files": _format_planned_sequence(stage.read_files),
                            "use_last_hess": stage.use_last_hess,
                            "save_files": _format_planned_sequence(stage.save_files),
                            "prune_options": _format_planned_mapping(
                                stage.prune_options
                            ),
                        }
                    )
                rows.append(row)

        columns = [
            "group",
            "stage",
            "calculation",
            "kind",
            "method_key",
            "engine",
            "options",
            "solvent",
            "lowest",
            "rank_by",
            "constraint",
            "n_cores",
        ]
        if detail == "full":
            columns.extend(
                [
                    "detailed_inp_str",
                    "xtra_inp_str",
                    "calculator_kwargs",
                    "read_files",
                    "use_last_hess",
                    "save_files",
                    "prune_options",
                ]
            )
        return pd.DataFrame(rows, columns=columns)

    def run(
        self,
        *,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None = None,
        out_dir: str | Path | None = None,
        execution: ExecutionMode | None = None,
        n_cores: int = 10,
        mem_gb: int = 20,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
        target_retention: TargetRetention = "compact_success",
        artifact_policy: ArtifactPolicy = "standard",
        uma_oet_tools: str | Path | None = None,
    ) -> pd.DataFrame:
        """Run selected workflow targets locally.

        Parameters
        ----------
        targets : iterable of WorkflowTarget or int or None, optional
            Targets to run. Integers select positions from ``wf.targets()``. If
            omitted, all workflow targets are run.
        out_dir : str or pathlib.Path or None, optional
            Output root. When provided, FRUST creates one subdirectory per
            target and writes staged parquet checkpoints inside each target
            directory. Successful targets are compacted according to
            ``target_retention``. When omitted, stages run in memory and no
            staged parquet files are written.
        execution : {"single_job", "dft_staged", "fully_staged"} or None, optional
            Local stage grouping. ``None`` defaults to ``"single_job"`` for
            local runs. Staged modes are most useful when ``out_dir`` is set
            because they mirror the cluster parquet layout.
        n_cores : int, optional
            Core count forwarded to embedding and calculators.
        mem_gb : int, optional
            Memory in GB forwarded to Stepper.
        debug : bool, optional
            Debug flag forwarded to stage preparation and calculators.
        save_output_dir : bool, optional
            Whether Stepper should retain calculator output directories.
        work_dir : str or pathlib.Path or None, optional
            Scratch/work directory forwarded to Stepper.
        target_retention : {"compact_success", "all"}, optional
            Output retention policy for successful targets when ``out_dir`` is
            provided. ``"compact_success"`` keeps only the final target parquet
            and ``timing.json`` after a target completes successfully.
            ``"all"`` keeps all intermediate parquet checkpoints.
        artifact_policy : {"standard", "screening"}, optional
            ``"screening"`` retains final structure, scalar energies, and
            frequency values while omitting consumed intermediate data. Full
            UMA TS results retain final mode vectors for review.
            ``"standard"`` remains the default.
        uma_oet_tools : str or pathlib.Path or None, optional
            OET runtime used for UMA stages within each local target or stage
            group. For example, pass a dedicated FairChem 2.23 runtime here.

        Returns
        -------
        pandas.DataFrame
            Concatenated results for the selected targets with merged workflow
            provenance in ``df.attrs``.

        Examples
        --------
        Run a one-target smoke test with the same stage boundaries used for a
        later cluster run:

        >>> df = wf.run(targets=[0], out_dir="debug/screen_ts", execution="dft_staged")
        """
        _validate_target_retention(target_retention)
        artifact_policy = validate_artifact_policy(artifact_policy)
        _validate_screening_retention(artifact_policy, target_retention)
        selected = self._select_targets(targets)
        options = ExecutionOptions(
            n_cores=n_cores,
            mem_gb=mem_gb,
            debug=debug,
            save_output_dir=save_output_dir,
            work_dir=None if work_dir is None else str(work_dir),
            artifact_policy=artifact_policy,
            uma_oet_tools=None if uma_oet_tools is None else str(uma_oet_tools),
        )
        frames: list[pd.DataFrame] = []
        root = Path(out_dir) if out_dir is not None else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

        for target in selected:
            save_dir = None if root is None else root / target.tag
            final_parquet: Path | None = None
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
            if save_dir is None:
                df = _run_target_job(self, target, save_dir, options)
            else:
                mode = execution or "single_job"
                groups = self._stage_groups(mode)
                if mode == "single_job":
                    df = _run_target_job(self, target, save_dir, options)
                    final_parquet = save_dir / "final.parquet"
                else:
                    current_parquet: str | None = None
                    df = pd.DataFrame()
                    for group_index, group in enumerate(groups):
                        output_parquet = _next_parquet(
                            current_parquet, self._group_name(group)
                        )
                        df = _run_stage_group_job(
                            self,
                            target,
                            [stage.id for stage in group],
                            current_parquet,
                            output_parquet,
                            save_dir,
                            options,
                            is_final_group=group_index == len(groups) - 1,
                        )
                        current_parquet = output_parquet
                    if current_parquet is not None:
                        final_parquet = save_dir / current_parquet
                if (
                    artifact_policy != "screening"
                    and target_retention == "compact_success"
                    and final_parquet is not None
                ):
                    _compact_successful_target(save_dir, final_parquet)
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged.attrs.update(
            merge_dataframe_attrs(
                frames,
                source_files=[target.tag for target in selected],
            )
        )
        return merged

    def submit(
        self,
        *,
        out_dir: str | Path,
        cluster: ClusterConfig,
        execution: ExecutionMode | None = None,
        stage_resources: dict[str, Resources] | None = None,
        array: bool = False,
        array_parallelism: int | Mapping[str, int] | None = None,
        targets_per_task: int = 1,
        retry: bool = False,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None = None,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
        collect: bool = True,
        collect_output: str | Path | None = None,
        collect_report: str | Path | None = None,
        collect_require_normal_termination: bool = True,
        collect_resources: Resources | None = None,
        target_retention: TargetRetention = "compact_success",
        orca_memory_fraction: float = DEFAULT_ORCA_MEMORY_FRACTION,
        artifact_policy: ArtifactPolicy = "standard",
        uma_oet_tools: str | Path | None = None,
        _defer_screening_cleanup: bool = False,
    ) -> JobSubmissionResult:
        """Submit selected workflow targets to a submitit cluster executor.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Root output directory. FRUST creates one subdirectory per selected
            workflow target and writes staged parquet checkpoints inside that
            target directory. Successful collected targets are compacted
            according to ``target_retention``.
        cluster : frust.cluster.config.ClusterConfig
            Shared executor configuration, such as Slurm partition, log
            directory, and optional scratch ``work_dir``.
        execution : {"single_job", "dft_staged", "fully_staged"} or None, optional
            Job grouping strategy. If omitted, DFT workflows use
            ``"dft_staged"`` and non-DFT workflows use ``"single_job"``.
            ``"single_job"`` submits one job per target. ``"dft_staged"`` keeps
            initialization stages together, then submits dependent DFT-stage
            jobs. ``"fully_staged"`` submits one dependent job per stage.
        stage_resources : dict[str, Resources] or None, optional
            Optional resource overrides by stage-group name. Missing groups use
            ``Resources(cpus=4, mem_gb=20, timeout_min=720)``. Call
            ``wf.show_stages(execution="dft_staged")`` to see the active group
            names before choosing overrides. In ``"dft_staged"`` mode, raw
            molecule and molecule workflows usually use ``"init"``,
            ``"dft_opt"``, ``"dft_freq"``, and ``"dft_solv_sp"``; screen TS
            workflows usually use ``"init"``, ``"dft_hessian"``,
            ``"dft_ts_opt"``, ``"dft_freq"``, and ``"dft_solv_sp"``.
        array : bool, optional
            Submit worker jobs together as Slurm arrays. Defaults to False,
            preserving individual submission. Staged modes submit one array per
            group; each element waits for its own preceding element's success.
            Slurm must support ``aftercorr`` and ``kill-on-invalid-dep=yes``;
            invalid dependencies are cancelled so collection can finish.
        array_parallelism : int or mapping of str to int or None, optional
            Maximum running elements per array, required when ``array=True``.
            ``single_job`` requires one positive integer. Staged modes accept one
            integer for all groups or a complete mapping keyed by stage group.
            This is not a run-wide limit across separate arrays. Local array
            submission waits for free execution slots before submitting more
            workers, so this call can block until earlier workers finish. Local
            staged submission dispatches ready work until the entire chain ends.
        targets_per_task : int, optional
            Number of sequential targets per element; defaults to 1. Larger
            batches share a lazy job-local UMA server in ``single_job`` mode.
            Resources apply to the element and its timeout covers the whole
            batch. Staged arrays require 1. Target exceptions are recorded and
            independent targets continue, unless the server becomes unavailable.
        retry : bool, optional
            Explicitly retry selected failed targets in a compatible run.
            Staged retries require ``array=True`` and rerun the complete selected
            target chain; intermediate checkpoint resume is unsupported.
            Earlier jobs and collection must be finished. Old target files
            are archived, successful targets are rejected, and resources/batch
            size may change within the mode's constraints. Execution groups must
            match. Defaults to False; overlapping writes are blocked.
        targets : iterable of WorkflowTarget or int or None, optional
            Targets to submit. Integers select positions from ``wf.targets()``.
            If omitted, all workflow targets are submitted.
        debug : bool, optional
            Forwarded to workflow stages and calculators.
        save_output_dir : bool, optional
            Forwarded to workflow stages that save calculator output
            directories.
        work_dir : str or pathlib.Path or None, optional
            Scratch/work directory override. If omitted, ``cluster.work_dir`` is
            used when configured.
        collect : bool, optional
            If ``True``, submit one final dependent collection job after the
            target jobs. The collector writes a merged parquet and JSON report.
        collect_output : str or pathlib.Path or None, optional
            Merged parquet written by the automatic collector. ``None`` writes
            ``merged.parquet`` inside ``out_dir``.
        collect_report : str or pathlib.Path or None, optional
            JSON report written by the automatic collector. ``None`` writes
            ``collection_report.json`` inside ``out_dir``.
        collect_require_normal_termination : bool, optional
            If ``True``, the automatic collector skips target outputs whose
            normal-termination columns are present and not all true. Skipped
            files are listed in the JSON report.
        collect_resources : Resources or None, optional
            Scheduler resources for the automatic collection job. ``None`` uses
            ``Resources(cpus=2, mem_gb=4, timeout_min=120)``.
        target_retention : {"compact_success", "all"}, optional
            Output retention policy for successful collected targets.
            ``"compact_success"`` keeps only the final target parquet and
            ``timing.json`` for collected targets. Failed, skipped, missing, and
            non-normal-termination targets remain untouched. ``"all"`` keeps
            all intermediate target parquets.
        orca_memory_fraction : float, optional
            Fraction of each target job's Slurm memory allocation forwarded to
            ORCA through Stepper. The default, ``0.8``, reserves 20 percent
            of the requested allocation for job overhead. Slurm still receives
            the full ``Resources.mem_gb`` value.
        artifact_policy : {"standard", "screening"}, optional
            ``"screening"`` projects normally terminated target results onto
            the compact scientific schema. It requires
            ``target_retention="compact_success"``.
        uma_oet_tools : str or pathlib.Path or None, optional
            OET runtime selected inside each submitted UMA job. For example,
            ``"/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu"`` selects the
            pinned cluster runtime without starting a server on the login node.

        Returns
        -------
        frust.cluster.config.JobSubmissionResult
            Submitted scheduler job IDs, target tags, target save directories,
            workflow execution mode, backend, and automatic collection metadata
            when ``collect=True``.
        """
        _validate_target_retention(target_retention)
        artifact_policy = validate_artifact_policy(artifact_policy)
        _validate_screening_retention(artifact_policy, target_retention)
        selected = self._select_targets(targets)
        mode = execution or ("dft_staged" if self.dft else "single_job")
        groups = self._stage_groups(mode)
        planned_groups = []
        planned_output = None
        for group in groups:
            group_name = "single_job" if mode == "single_job" else self._group_name(group)
            planned_output = (
                "final.parquet" if mode == "single_job"
                else _next_parquet(planned_output, group_name)
            )
            planned_groups.append((
                group_name,
                _resource_for_group(group_name, group, stage_resources, default=DEFAULT_WORKFLOW_RESOURCES),
                planned_output,
            ))
        plan = _plan_submission(
            [target.tag for target in selected], planned_groups, execution=mode,
            array=array, array_parallelism=array_parallelism, targets_per_task=targets_per_task,
        )
        if array:
            _validate_array_scheduler_options(cluster)
            _validate_array_size(cluster, len(plan[0].batches))
        if not isinstance(retry, bool):
            raise ValueError("retry must be a boolean")
        if retry and artifact_policy == "screening":
            raise NotImplementedError("Screen retries must use the screen reuse contract; direct retry is not supported")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        with _submission_guard(root, plan, self, selected, cluster, mode=mode, array=array, retry=retry) as ledger:
            executor = create_executor(cluster) if selected else None
            submitted_workflow = copy.copy(self)
            submitted_workflow._target_cache = None
            if hasattr(submitted_workflow, "dataframe"):
                submitted_workflow.dataframe = None
            if hasattr(submitted_workflow, "csv_path"):
                submitted_workflow.csv_path = None
            if hasattr(submitted_workflow, "smiles"):
                submitted_workflow.smiles = None

            job_ids: list[str | int] = []
            tags: list[str] = []
            save_dirs: list[str] = []
            final_job_ids: list[str | int] = []
            final_jobs: list[Any] = []
            expected_parquets: dict[str, str] = {}

            if array and selected and mode != "single_job":
                tags = [target.tag for target in selected]
                save_dirs = [str(root / tag) for tag in tags]
                for directory in save_dirs:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                staged_options = ExecutionOptions(
                    debug=debug, save_output_dir=save_output_dir,
                    work_dir=str(work_dir or cluster.work_dir) if (work_dir or cluster.work_dir) else None,
                    artifact_policy=artifact_policy,
                    uma_oet_tools=None if uma_oet_tools is None else str(uma_oet_tools),
                )
                final_jobs, final_job_ids = _submit_staged_arrays(
                    submitted_workflow, selected, root, cluster, plan, groups,
                    ledger, executor, staged_options, orca_memory_fraction,
                )
                job_ids = [job.job_id for job in final_jobs]
                expected_parquets = dict.fromkeys(tags, plan[-1].output_name)

            elif array and selected:
                resources = plan[0].resources
                update_executor_with_dependency(
                    executor, cluster, resources,
                    job_name=f"{self.workflow_name}_workflow", dependency_job_id=None,
                )
                options = ExecutionOptions(
                    n_cores=resources.cpus,
                    mem_gb=orca_memory_gb(resources, orca_memory_fraction),
                    debug=debug, save_output_dir=save_output_dir,
                    work_dir=str(work_dir or cluster.work_dir) if (work_dir or cluster.work_dir) else None,
                    artifact_policy=artifact_policy,
                    uma_oet_tools=None if uma_oet_tools is None else str(uma_oet_tools),
                )
                arguments = []
                for target in selected:
                    target_dir = root / target.tag
                    target_dir.mkdir(parents=True, exist_ok=True)
                    tags.append(target.tag)
                    save_dirs.append(str(target_dir))
                    expected_parquets[target.tag] = "final.parquet"
                if targets_per_task == 1:
                    worker = _run_target_submitted_job
                    arguments = [(submitted_workflow, target, root / target.tag, options, utc_timestamp(), ledger.attempt_id, index)
                                 for index, target in enumerate(selected)]
                else:
                    worker = _run_target_batch_submitted_job
                    by_tag = {target.tag: target for target in selected}
                    arguments = [
                        (submitted_workflow, [by_tag[tag] for tag in batch], root, options,
                         utc_timestamp(), ledger.attempt_id, index)
                        for index, batch in enumerate(plan[0].batches)
                    ]
                try:
                    final_jobs = _submit_array_jobs(
                        executor, cluster, worker, arguments,
                        parallelism=plan[0].parallelism,
                        on_submitted=lambda index, job: ledger.submitted("single_job", index, job),
                        on_array_submitted=lambda jobs: ledger.submitted_many("single_job", dict(enumerate(jobs))),
                    )
                except Exception as error:
                    ledger.failed("single_job", range(len(arguments)), error)
                    raise
                job_ids = [job.job_id for job in final_jobs]
                final_job_ids = list(dict.fromkeys(
                    record.array_job_id or record.job_id for record in ledger.records
                ))

            for target_index, target in enumerate([] if array else selected):
                target_dir = root / target.tag
                target_dir.mkdir(parents=True, exist_ok=True)
                tags.append(target.tag)
                save_dirs.append(str(target_dir))
                last_job = None
                current_parquet: str | None = None

                if mode == "single_job":
                    resources = _resource_for_group(
                        "single_job",
                        groups[0],
                        stage_resources,
                        default=DEFAULT_WORKFLOW_RESOURCES,
                    )
                    update_executor_with_dependency(
                        executor,
                        cluster,
                        resources,
                        job_name=f"{target.tag}_workflow",
                        dependency_job_id=None,
                    )
                    options = ExecutionOptions(
                        n_cores=resources.cpus,
                        mem_gb=orca_memory_gb(resources, orca_memory_fraction),
                        debug=debug,
                        save_output_dir=save_output_dir,
                        work_dir=(
                            str(work_dir or cluster.work_dir)
                            if (work_dir or cluster.work_dir)
                            else None
                        ),
                        artifact_policy=artifact_policy,
                        uma_oet_tools=None if uma_oet_tools is None else str(uma_oet_tools),
                    )
                    submitted_at = utc_timestamp()
                    job = ledger.submit(
                        "single_job", target_index, executor,
                        _run_target_submitted_job,
                        submitted_workflow,
                        target,
                        target_dir,
                        options,
                        submitted_at, ledger.attempt_id, target_index,
                    )
                    job_id = getattr(job, "job_id", f"{target.tag}_workflow")
                    job_ids.append(job_id)
                    final_job_ids.append(job_id)
                    final_jobs.append(job)
                    expected_parquets[target.tag] = "final.parquet"
                    continue

                for group_index, group in enumerate(groups):
                    group_name = self._group_name(group)
                    resources = _resource_for_group(
                        group_name,
                        group,
                        stage_resources,
                        default=DEFAULT_WORKFLOW_RESOURCES,
                    )
                    update_executor_with_dependency(
                        executor,
                        cluster,
                        resources,
                        job_name=f"{target.tag}_{group_name}",
                        dependency_job_id=getattr(last_job, "job_id", None),
                    )
                    output_parquet = _next_parquet(current_parquet, group_name)
                    options = ExecutionOptions(
                        n_cores=resources.cpus,
                        mem_gb=orca_memory_gb(resources, orca_memory_fraction),
                        debug=debug,
                        save_output_dir=save_output_dir,
                        work_dir=(
                            str(work_dir or cluster.work_dir)
                            if (work_dir or cluster.work_dir)
                            else None
                        ),
                        artifact_policy=artifact_policy,
                        uma_oet_tools=None if uma_oet_tools is None else str(uma_oet_tools),
                    )
                    submitted_at = utc_timestamp()
                    job = ledger.submit(
                        group_name, target_index, executor,
                        _run_stage_group_submitted_job,
                        submitted_workflow,
                        target,
                        [stage.id for stage in group],
                        current_parquet,
                        output_parquet,
                        target_dir,
                        options,
                        submitted_at,
                        group_index == len(groups) - 1,
                    )
                    job_id = getattr(job, "job_id", f"{target.tag}_{group_name}")
                    job_ids.append(job_id)
                    last_job = job
                    current_parquet = output_parquet

                if last_job is not None and current_parquet is not None:
                    final_job_id = getattr(
                        last_job, "job_id", f"{target.tag}_{self._group_name(groups[-1])}"
                    )
                    final_job_ids.append(final_job_id)
                    final_jobs.append(last_job)
                    expected_parquets[target.tag] = current_parquet

            collection_job_id: str | int | None = None
            collection_output_path: Path | None = None
            collection_report_path: Path | None = None
            if collect and selected and final_job_ids:
                if array:
                    if cluster.backend == "local":
                        for job in final_jobs:
                            job.wait()
                    executor = create_executor(cluster)
                collection_output_path = (
                    Path(collect_output)
                    if collect_output is not None
                    else (root / ".frust/retries" / ledger.attempt_id / "merged.parquet" if retry else root / "merged.parquet")
                )
                collection_report_path = (
                    Path(collect_report)
                    if collect_report is not None
                    else (root / ".frust/retries" / ledger.attempt_id / "collection_report.json" if retry else root / "collection_report.json")
                )
                update_executor_with_dependencies(
                    executor,
                    cluster,
                    collect_resources or DEFAULT_COLLECTION_RESOURCES,
                    job_name=f"{self.workflow_name}_collect",
                    dependency_job_ids=final_job_ids,
                    dependency_type="afterany",
                )
                wait_jobs = final_jobs if cluster.backend == "local" and not array else None
                collection_job = executor.submit(
                    _collect_expected_outputs_submitted,
                    submitted_workflow,
                    selected,
                    root,
                    expected_parquets,
                    collection_output_path,
                    collection_report_path,
                    collect_require_normal_termination,
                    wait_jobs,
                    target_retention,
                    artifact_policy,
                    _defer_screening_cleanup, ledger.attempt_id,
                )
                ledger.collected(collection_job)
                collection_job_id = getattr(
                    collection_job, "job_id", f"{self.workflow_name}_collect"
                )

            return JobSubmissionResult(
                job_ids=job_ids,
                tags=tags,
                save_dirs=save_dirs,
                mode=f"{self.workflow_name}:{mode}",
                backend=cluster.backend,
                records=list(ledger.records),
                array_job_ids=list(dict.fromkeys(r.array_job_id for r in ledger.records if r.array_job_id)),
                submission_path=str(ledger.path),
                collection_job_id=collection_job_id,
                collection_output=(
                    None if collection_output_path is None else str(collection_output_path)
                ),
                collection_report=(
                    None if collection_report_path is None else str(collection_report_path)
                ),
            )

    def collect(
        self,
        out_dir: str | Path,
        *,
        output: str | Path | None = None,
        report: str | Path | None = None,
        require_normal_termination: bool = False,
        target_retention: TargetRetention = "all",
    ) -> pd.DataFrame:
        """Collect finished per-target workflow outputs.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Root directory passed to ``wf.run(...)`` or ``wf.submit(...)``.
            FRUST looks below each known target subdirectory and reads the
            deepest staged parquet file, or ``final.parquet`` for single-job
            outputs.
        output : str or pathlib.Path or None, optional
            Merged dataframe path. Defaults to ``out_dir/merged.parquet``.
        report : str or pathlib.Path or None, optional
            Collection diagnostics, including ``retry_targets`` (failed or missing
            target tags) and per-target outcomes. Defaults to collection_report.json.
        require_normal_termination : bool, optional
            If ``True``, skip target outputs where normal-termination columns
            ending in ``"-NT"`` are present and not all true.
        target_retention : {"all", "compact_success"}, optional
            Output retention policy for collected targets. The default
            ``"all"`` preserves existing run directories during manual
            recovery. ``"compact_success"`` removes intermediate parquet
            checkpoints for successfully collected targets.

        Returns
        -------
        pandas.DataFrame
            Merged target outputs with dataframe attrs combined so helpers such
            as ``ft.show_steps(...)`` still summarize the workflow.

        Raises
        ------
        FileNotFoundError
            If no final staged parquet files are found below ``out_dir``.
        """
        _validate_target_retention(target_retention)
        root = Path(out_dir)
        targets = self.targets()
        with _mutation_lock(root):
            _ensure_collection_ready(root, targets)
            outcomes = _target_outcomes(root)
            expected = {}
            for target in targets:
                if target.tag in outcomes:
                    expected[target.tag] = Path(outcomes[target.tag]['output_path']).name
                else:
                    deepest = _deepest_parquet(root / target.tag)
                    expected[target.tag] = deepest.name if deepest is not None else 'final.parquet'
            return _collect_expected_outputs(
                self, targets, root, expected,
                output if output is not None else root / 'merged.parquet',
                report if report is not None else root / 'collection_report.json',
                require_normal_termination, target_retention=target_retention,
            )

    def _build_targets(self) -> list[WorkflowTarget]:
        """Build lightweight scientific targets for this workflow.

        Returns
        -------
        list of WorkflowTarget
            Targets used by ``targets()``, ``run(...)``, ``submit(...)``, and
            ``collect(...)``.

        Notes
        -----
        Subclasses should avoid expensive conformer generation, embedding, or
        calculator calls here. The returned payloads must be serializable for
        cluster submission.
        """
        raise NotImplementedError

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        """Create the first FRUST dataframe for one target.

        Parameters
        ----------
        target : WorkflowTarget
            Target selected by ``run(...)`` or ``submit(...)``.
        save_dir : pathlib.Path or None
            Target output directory, when output is being written.
        options : ExecutionOptions
            Runtime options for embedding and calculators.

        Returns
        -------
        pandas.DataFrame
            Initial dataframe with atoms, embedded coordinates, and workflow
            metadata columns needed by later stages.

        Notes
        -----
        This is where expensive workflow-specific structure generation belongs,
        because it runs inside the local execution path or inside the submitted
        cluster job.
        """
        raise NotImplementedError

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        """Return the Stepper ``step_type`` for a target.

        Parameters
        ----------
        target : WorkflowTarget
            Target about to be prepared or calculated.

        Returns
        -------
        str or None
            Stepper type such as ``"MOLS"``, ``"TS1"``, or ``"INT3"``. ``None``
            leaves Stepper to infer behavior from the dataframe where possible.
        """
        return None

    def _stage_defs(self) -> list[StageDef]:
        """Return the ordered stage graph for the workflow.

        Returns
        -------
        list of StageDef
            Stage definitions used identically by local execution and cluster
            submission.
        """
        raise NotImplementedError

    def _select_targets(
        self,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None,
    ) -> list[WorkflowTarget]:
        """Resolve user target selection to concrete target objects.

        Parameters
        ----------
        targets : iterable of WorkflowTarget or int or None
            ``None`` selects all targets. Integers index into ``wf.targets()``.
            Explicit ``WorkflowTarget`` objects are returned as supplied.

        Returns
        -------
        list of WorkflowTarget
            Selected targets in execution order.
        """
        all_targets = self.targets()
        if targets is None:
            return all_targets
        selected = list(targets)
        if not selected:
            return []
        if all(isinstance(item, int) for item in selected):
            return [all_targets[int(item)] for item in selected]
        return selected  # type: ignore[return-value]

    def _stage_groups(self, execution: ExecutionMode) -> list[list[StageDef]]:
        """Group stages according to an execution mode.

        Parameters
        ----------
        execution : {"single_job", "dft_staged", "fully_staged"}
            Execution grouping strategy.

        Returns
        -------
        list of list of StageDef
            Stage groups. Each group runs serially in one local section or one
            submitted job. ``"dft_staged"`` keeps initialization together and
            splits out known DFT stages for DFT workflows.
        """
        stages = self._stage_defs()
        if execution == "single_job":
            return [stages]
        if execution == "fully_staged":
            return [[stage] for stage in stages]
        if execution != "dft_staged":
            raise ValueError(
                "execution must be 'single_job', 'dft_staged', or 'fully_staged'"
            )

        if not self.dft:
            return [stages]

        dft_stage_ids = {
            "dft_hessian",
            "dft_ts_opt",
            "dft_freq",
            "dft_solv_sp",
            "dft_opt",
        }
        first_split = next(
            (idx for idx, stage in enumerate(stages) if stage.id in dft_stage_ids),
            len(stages),
        )
        groups: list[list[StageDef]] = []
        if first_split:
            groups.append(stages[:first_split])
        groups.extend([[stage] for stage in stages[first_split:]])
        return groups

    def _group_name(self, group: list[StageDef]) -> str:
        """Return the resource/parquet name for a stage group.

        Parameters
        ----------
        group : list of StageDef
            Stages that run together.

        Returns
        -------
        str
            ``"init"`` for a group containing the prepare stage, otherwise the
            last stage id in the group.
        """
        if any(stage.kind == "prepare" for stage in group):
            return "init"
        return group[-1].id

    def _run_stage_group(
        self,
        target: WorkflowTarget,
        stages: list[StageDef],
        *,
        input_df: pd.DataFrame | None,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        """Run a serial stage group for one target.

        Parameters
        ----------
        target : WorkflowTarget
            Target being processed.
        stages : list of StageDef
            Ordered stages to run in this group.
        input_df : pandas.DataFrame or None
            Input dataframe for non-prepare groups. The first group usually
            starts with ``None`` and a ``"prepare"`` stage.
        save_dir : pathlib.Path or None
            Target output directory, when outputs should be retained.
        options : ExecutionOptions
            Runtime options for preparation and calculators.

        Returns
        -------
        pandas.DataFrame
            Output dataframe after all stages in the group have run.
        """
        df = input_df
        group_name = self._group_name(stages)
        for stage in stages:
            stage_started_at = utc_timestamp()
            stage_start = monotonic_seconds()
            input_rows = None if df is None else int(len(df))
            if stage.kind == "prepare":
                df = self._prepare_initial_df(
                    target, save_dir=save_dir, options=options
                )
            else:
                if df is None:
                    raise ValueError(f"Stage {stage.id!r} requires an input dataframe")
                df = _run_stage_calculation(
                    self,
                    target,
                    stage,
                    df,
                    save_dir=save_dir,
                    options=options,
                )
            df = _attach_workflow_attrs(df, workflow=self, target=target)
            _log_workflow_stage_summary(
                workflow=self,
                target=target,
                stage=stage,
                df=df,
                input_rows=input_rows,
                options=options,
            )
            append_workflow_timing(
                df.attrs,
                build_workflow_timing_record(
                    workflow=self.workflow_name,
                    target=target.tag,
                    group=group_name,
                    stage=stage.id,
                    kind=stage.kind,
                    started_at=stage_started_at,
                    finished_at=utc_timestamp(),
                    elapsed_s=elapsed_seconds(stage_start),
                    input_rows=input_rows,
                    output_rows=int(len(df)),
                    resources={
                        "n_cores": options.n_cores,
                        "memory_gb": options.mem_gb,
                    },
                ),
            )
            _write_analysis_tier_snapshot(
                self,
                target,
                stage,
                df,
                save_dir=save_dir,
                artifact_policy=options.artifact_policy,
            )
            if (
                options.artifact_policy == "screening"
                and stage.id == "dft_ts_opt"
                and save_dir is not None
                and _all_normal_terminated(df)
            ):
                hessian_columns = [
                    column for column in df.columns if str(column).endswith(".hess")
                ]
                if hessian_columns:
                    df = df.drop(columns=hessian_columns)
                _remove_consumed_hessians(save_dir)
        if df is None:
            raise ValueError("No workflow stages were run")
        return df


def _uma_scope_for_stages(
    workflow: BaseWorkflow,
    stages: list[StageDef],
    options: ExecutionOptions,
):
    """Return a job-local UMA scope only when a stage needs its server.

    Parameters
    ----------
    workflow : BaseWorkflow
        Workflow whose method plan provides calculator specifications.
    stages : list of StageDef
        Stages executed serially in this job or local group.
    options : ExecutionOptions
        Runtime options, including an optional pinned OET runtime.

    Returns
    -------
    context manager
        A lazy UMA server scope or an inert context for non-UMA groups.
    """
    uses_uma = any(
        stage.kind == "calc"
        and (spec := workflow.method.for_stage(stage.method_stage or stage.id)).engine
        == "orca"
        and spec.kwargs.get("uma") is not None
        and spec.kwargs.get("uma_server", True)
        for stage in stages
    )
    if not uses_uma:
        return nullcontext()
    return uma_job_server_scope(oet_tools=options.uma_oet_tools, reuse=True)


def _run_target_batch_submitted_job(
    workflow, targets, root, options, submitted_at, attempt_id, batch_index,
) -> list[WorkflowJobResult]:
    """Run a sequential batch, preserving outputs and durable target outcomes."""
    root = Path(root)
    path = root / ".frust" / "batches" / attempt_id / f"{batch_index}.json"
    started = monotonic_seconds()
    records = [{"target": target.tag, "status": "unattempted", "error": None} for target in targets]
    payload = {
        "schema_version": 1, "attempt_id": attempt_id, "batch_index": batch_index,
        "job_id": _current_job_id(), "started_at": utc_timestamp(),
        "resources": _options_resources(options), "targets": records,
    }
    results = []
    errors = []
    scope = None
    _atomic_write_submission_json(path, payload)
    try:
        with _uma_scope_for_stages(workflow, workflow._stage_defs(), options) as scope:
            for target, record in zip(targets, records):
                print(f"[FRUST batch {batch_index}] target={target.tag}", flush=True)
                record.update(status="running", started_at=utc_timestamp())
                _atomic_write_submission_json(path, payload)
                try:
                    df = _run_target_job(workflow, target, root / target.tag, options, submitted_at, attempt_id)
                    record.update(
                        status="success" if _all_normal_terminated(df) else "non_normal",
                        output_path=str(root / target.tag / "final.parquet"), row_count=len(df),
                    )
                    results.append(WorkflowJobResult(target.tag, record["output_path"], len(df), record["status"]))
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    errors.append(target.tag)
                except BaseException as error:
                    record.update(status="interrupted", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    record["finished_at"] = utc_timestamp()
                    _atomic_write_submission_json(path, payload)
                if scope is not None:
                    scope.ensure_healthy()
            if errors:
                raise RuntimeError(f"Targets failed in batch {batch_index}: {', '.join(errors)}")
    except BaseException as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        payload.update(finished_at=utc_timestamp(), elapsed_s=elapsed_seconds(started))
        if scope is not None:
            payload["server_startup_s"] = scope._startup_elapsed_s
            if scope._handle is not None:
                payload["uma_server"] = {
                    "pid": scope._handle.pid, "hostname": scope._handle.hostname,
                    "bind": scope._handle.bind,
                }
        _atomic_write_submission_json(path, payload)
        _mark_completed(root, attempt_id, "single_job", batch_index)
    return results


def _run_target_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    save_dir: str | Path | None,
    options: ExecutionOptions,
    submitted_at: str | None = None,
    attempt_id: str | None = None,
) -> pd.DataFrame:
    """Run every workflow stage for one target.

    Parameters
    ----------
    workflow : BaseWorkflow
        Workflow object supplying stage definitions and stage behavior.
    target : WorkflowTarget
        Target to process.
    save_dir : str or pathlib.Path or None
        Target output directory, or ``None`` for in-memory execution.
    options : ExecutionOptions
        Runtime options forwarded to stage execution.

    Returns
    -------
    pandas.DataFrame
        Final target dataframe after all stages have run.
    """
    target_dir = None if save_dir is None else Path(save_dir)
    group_started_at = utc_timestamp()
    group_start = monotonic_seconds()
    stages = workflow._stage_defs()
    with _uma_scope_for_stages(workflow, stages, options):
        df = workflow._run_stage_group(
            target,
            stages,
            input_df=None,
            save_dir=target_dir,
            options=options,
        )
    group_record = build_workflow_timing_record(
        workflow=workflow.workflow_name,
        target=target.tag,
        group="single_job",
        stage=None,
        kind="group",
        job_id=_current_job_id(),
        started_at=group_started_at,
        finished_at=utc_timestamp(),
        elapsed_s=elapsed_seconds(group_start),
        input_rows=None,
        output_rows=int(len(df)),
        resources=_options_resources(options),
    )
    append_workflow_timing(df.attrs, group_record)
    if options.artifact_policy == "screening" and _all_normal_terminated(df):
        df = compact_result_dataframe(df)
    if attempt_id is not None:
        df.attrs["frust_submission"] = {"attempt_id": attempt_id, "target": target.tag}
    if target_dir is not None:
        _atomic_write_parquet(df, target_dir / "final.parquet")
        _write_target_timing(
            target_dir,
            df,
            workflow=workflow.workflow_name,
            target=target.tag,
            submitted_at=submitted_at,
            group_record=group_record,
            stage_ids=[stage.id for stage in stages],
            input_parquet=None,
            output_parquet="final.parquet",
        )
    return df


def _run_target_submitted_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    save_dir: str | Path,
    options: ExecutionOptions,
    submitted_at: str | None = None,
    attempt_id: str | None = None,
    batch_index: int = 0,
) -> WorkflowJobResult:
    """Run one target with compact status, attribution, and completion records."""
    if attempt_id is not None:
        return _run_target_batch_submitted_job(
            workflow, [target], Path(save_dir).parent, options, submitted_at, attempt_id, batch_index,
        )[0]
    df = _run_target_job(workflow, target, save_dir, options, submitted_at)
    return WorkflowJobResult(target.tag, str(Path(save_dir) / "final.parquet"), len(df), "success")


def _submit_staged_arrays(workflow, targets, root, cluster, plan, groups, ledger,
                          first_executor, base_options, memory_fraction):
    """Submit matching stage arrays, or dispatch a ready-only local graph."""
    executors = []
    arguments = []
    jobs = []
    dependency_ids = []
    upstream_id = None
    dependency_type = 'afterok'
    for group_index, (group_plan, stages) in enumerate(zip(plan, groups)):
        executor = first_executor if group_index == 0 else create_executor(cluster)
        executors.append(executor)
        update_executor_with_dependencies(
            executor, cluster, group_plan.resources,
            job_name=f'{workflow.workflow_name}_{group_plan.name}',
            dependency_job_ids=[] if upstream_id is None else [upstream_id],
            dependency_type=dependency_type, kill_on_invalid_dependency=True,
        )
        options = replace(base_options, n_cores=group_plan.resources.cpus,
                          mem_gb=orca_memory_gb(group_plan.resources, memory_fraction))
        previous = None if group_index == 0 else plan[group_index - 1]
        args = [
            (workflow, target, [stage.id for stage in stages],
             None if previous is None else previous.output_name,
             group_plan.output_name, root / target.tag, options, utc_timestamp(),
             group_index == len(plan) - 1, ledger.attempt_id, group_plan.name, index,
             None if previous is None else previous.name)
            for index, target in enumerate(targets)
        ]
        arguments.append(args)
        if cluster.backend == 'slurm':
            try:
                submitted = _submit_array_jobs(
                    executor, cluster, _run_stage_array_submitted_job, args,
                    parallelism=group_plan.parallelism,
                    on_submitted=lambda index, job: ledger.submitted(group_plan.name, index, job),
                    on_array_submitted=lambda batch: ledger.submitted_many(group_plan.name, dict(enumerate(batch))),
                )
            except Exception as error:
                ledger.failed(group_plan.name, range(len(args)), error)
                raise RuntimeError(
                    'Staged arrays require Slurm aftercorr and kill-on-invalid-dep=yes support. '
                    'Submission failed; inspect accepted jobs and the scheduler error before retrying.'
                ) from error
            jobs.extend(submitted)
            upstream_id, dependency_type = _array_dependency(submitted)
            dependency_ids.append(upstream_id)
    if cluster.backend == 'local':
        def blocked(group_index, index, error):
            group = plan[group_index].name
            _atomic_write_submission_json(
                _stage_outcome_path(root, ledger.attempt_id, group, index),
                {'target': targets[index].tag, 'group': group, 'status': 'blocked',
                 'error': error, 'finished_at': utc_timestamp()},
            )
            _mark_completed(root, ledger.attempt_id, group, index)
        jobs = _submit_local_stages(executors, plan, arguments, _run_stage_array_submitted_job, ledger, blocked)
        dependency_ids = [job.job_id for job in jobs]
    return jobs, dependency_ids


def _run_stage_array_submitted_job(workflow, target, stage_ids, input_parquet,
                                  output_parquet, save_dir, options, submitted_at,
                                  is_final_group, attempt_id, group, index, previous_group):
    """Run one attributed stage and make scientific failure block descendants."""
    root = Path(save_dir).parent
    path = _stage_outcome_path(root, attempt_id, group, index)
    payload = {'schema_version': 1, 'target': target.tag, 'group': group, 'attempt_id': attempt_id,
               'status': 'running', 'started_at': utc_timestamp(), 'started_ns': time.time_ns(),
               'job_id': _current_job_id(), 'resources': _options_resources(options)}
    _atomic_write_submission_json(path, payload)
    try:
        if previous_group is not None:
            upstream = _stage_outcome_path(root, attempt_id, previous_group, index)
            if not upstream.exists() or json.loads(upstream.read_text()).get('status') != 'success':
                payload.update(status='blocked', error=f'Upstream group {previous_group} has no validated success')
                raise RuntimeError(payload['error'])
        df = _run_stage_group_job(
            workflow, target, stage_ids, input_parquet, output_parquet, save_dir,
            options, submitted_at, is_final_group=is_final_group,
            attempt_id=attempt_id, expected_input_group=previous_group,
        )
        payload.update(status='success' if _all_normal_terminated(df) else 'non_normal',
                       output_path=str(Path(save_dir) / output_parquet), row_count=len(df))
        if payload['status'] == 'non_normal':
            payload['error'] = 'Scientific normal-termination checks failed'
            raise RuntimeError(payload['error'])
        return WorkflowJobResult(target.tag, payload['output_path'], len(df), 'success')
    except Exception as error:
        if payload['status'] == 'running':
            payload.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    except BaseException as error:
        payload.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        payload['finished_at'] = utc_timestamp()
        payload['finished_ns'] = time.time_ns()
        _atomic_write_submission_json(path, payload)
        _mark_completed(root, attempt_id, group, index)


def _run_stage_group_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage_ids: list[str],
    input_parquet: str | None,
    output_parquet: str,
    save_dir: str | Path,
    options: ExecutionOptions,
    submitted_at: str | None = None,
    *,
    is_final_group: bool = False,
    attempt_id: str | None = None,
    expected_input_group: str | None = None,
) -> pd.DataFrame:
    """Run one submitted or staged-local stage group.

    Parameters
    ----------
    workflow : BaseWorkflow
        Serialized workflow object.
    target : WorkflowTarget
        Serialized target object.
    stage_ids : list of str
        Stage ids to run in this group.
    input_parquet : str or None
        Previous staged parquet filename inside ``save_dir``. ``None`` means the
        group starts from workflow preparation.
    output_parquet : str
        Output parquet filename to write inside ``save_dir``.
    save_dir : str or pathlib.Path
        Target output directory.
    options : ExecutionOptions
        Runtime options forwarded to stage execution.

    Returns
    -------
    pandas.DataFrame
        Stage-group output dataframe, also written to ``output_parquet``.
    """
    target_dir = Path(save_dir)
    group_started_at = utc_timestamp()
    group_start = monotonic_seconds()
    input_df = (
        None if input_parquet is None else pd.read_parquet(target_dir / input_parquet)
    )
    if input_df is not None and attempt_id is not None:
        expected = {'attempt_id': attempt_id, 'target': target.tag, 'group': expected_input_group}
        if input_df.attrs.get('frust_submission') != expected:
            raise ValueError('Staged checkpoint attribution does not match this target, group, and attempt')
    stages_by_id = {stage.id: stage for stage in workflow._stage_defs()}
    stages = [stages_by_id[stage_id] for stage_id in stage_ids]
    group_name = workflow._group_name(stages)
    with _uma_scope_for_stages(workflow, stages, options):
        df = workflow._run_stage_group(
            target,
            stages,
            input_df=input_df,
            save_dir=target_dir,
            options=options,
        )
    group_record = build_workflow_timing_record(
        workflow=workflow.workflow_name,
        target=target.tag,
        group=group_name,
        stage=None,
        kind="group",
        job_id=_current_job_id(),
        started_at=group_started_at,
        finished_at=utc_timestamp(),
        elapsed_s=elapsed_seconds(group_start),
        input_rows=None if input_df is None else int(len(input_df)),
        output_rows=int(len(df)),
        resources=_options_resources(options),
    )
    append_workflow_timing(df.attrs, group_record)
    if (
        is_final_group
        and options.artifact_policy == "screening"
        and _all_normal_terminated(df)
    ):
        df = compact_result_dataframe(df)
    if attempt_id is not None:
        df.attrs['frust_submission'] = {'attempt_id': attempt_id, 'target': target.tag, 'group': group_name}
    _atomic_write_parquet(df, target_dir / output_parquet)
    _write_target_timing(
        target_dir,
        df,
        workflow=workflow.workflow_name,
        target=target.tag,
        submitted_at=submitted_at,
        group_record=group_record,
        stage_ids=stage_ids,
        input_parquet=input_parquet,
        output_parquet=output_parquet,
    )
    if (
        options.artifact_policy == "screening"
        and "dft_ts_opt" in stage_ids
        and input_parquet is not None
        and _all_normal_terminated(df)
    ):
        (target_dir / input_parquet).unlink(missing_ok=True)
    return df


def _run_stage_group_submitted_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage_ids: list[str],
    input_parquet: str | None,
    output_parquet: str,
    save_dir: str | Path,
    options: ExecutionOptions,
    submitted_at: str | None = None,
    is_final_group: bool = False,
) -> WorkflowJobResult:
    """Run one submitted stage group and return only compact status metadata."""
    df = _run_stage_group_job(
        workflow,
        target,
        stage_ids,
        input_parquet,
        output_parquet,
        save_dir,
        options,
        submitted_at,
        is_final_group=is_final_group,
    )
    return WorkflowJobResult(
        target_id=target.tag,
        output_path=str(Path(save_dir) / output_parquet),
        row_count=int(len(df)),
        status="success",
    )


def _collect_expected_outputs(
    workflow: BaseWorkflow,
    targets: list[WorkflowTarget],
    out_dir: str | Path,
    expected_parquets: dict[str, str],
    output: str | Path,
    report: str | Path,
    require_normal_termination: bool,
    wait_jobs: list[Any] | None = None,
    target_retention: TargetRetention = "compact_success",
    artifact_policy: ArtifactPolicy = "standard",
    return_dataframe: bool = True,
    defer_screening_cleanup: bool = False,
    attempt_id: str | None = None,
) -> pd.DataFrame:
    """Collect exact expected workflow outputs and write a JSON report.

    Parameters
    ----------
    workflow : BaseWorkflow
        Workflow object used for collection context.
    targets : list of WorkflowTarget
        Targets submitted for this workflow run.
    out_dir : str or pathlib.Path
        Root output directory passed to ``wf.submit(...)``.
    expected_parquets : dict of str to str
        Mapping from target tag to the exact final parquet filename expected in
        that target directory.
    output : str or pathlib.Path
        Merged parquet path to write.
    report : str or pathlib.Path
        JSON collection report path to write.
    require_normal_termination : bool
        Whether to skip outputs with failed normal-termination columns.
    wait_jobs : list, optional
        Submitit jobs to wait for before collecting. This is used for local
        submitit backends that do not receive Slurm dependency parameters.
    target_retention : {"compact_success", "all"}, optional
        Output retention policy for successfully collected targets.

    Returns
    -------
    pandas.DataFrame
        Merged dataframe written to ``output``.

    Raises
    ------
    FileNotFoundError
        If no usable rows can be collected. The report is written before this
        error is raised.
    """
    for job in wait_jobs or []:
        job.wait()
    _validate_target_retention(target_retention)
    artifact_policy = validate_artifact_policy(artifact_policy)

    root = Path(out_dir)
    output_path = Path(output)
    report_path = Path(report)
    attr_frames: list[pd.DataFrame] = []
    row_count = 0
    collected_files: list[str] = []
    skipped_files: list[str] = []
    missing_files: list[str] = []
    errored_files: list[str] = []
    errors: list[dict[str, str]] = []
    outcomes = _target_outcomes(root, attempt_id, workers_finished=attempt_id is not None)

    for target in targets:
        expected_name = expected_parquets.get(target.tag)
        final_file = (
            root / target.tag / expected_name
            if expected_name
            else root / target.tag / "final.parquet"
        )
        outcome = outcomes.setdefault(target.tag, {"target": target.tag})
        if outcome.get("status") in {"failed", "interrupted", "unattempted", "running", "blocked"}:
            missing_files.append(str(final_file))
            continue
        if not final_file.exists():
            outcome["status"] = "missing"
            missing_files.append(str(final_file))
            continue
        try:
            df = normalize_dataframe(pd.read_parquet(final_file))
        except Exception as exc:
            outcome.update(status="unreadable", error=str(exc))
            errored_files.append(str(final_file))
            errors.append({"file": str(final_file), "error": str(exc)})
            continue
        if outcome.get("tracked") and df.attrs.get("frust_submission", {}).get("attempt_id") != outcome["attempt_id"]:
            outcome.update(status="stale", error="Output does not belong to the selected submission attempt")
            missing_files.append(str(final_file))
            continue
        outcome["status"] = "success" if _all_normal_terminated(df) else "non_normal"
        if require_normal_termination and not _all_normal_terminated(df):
            skipped_files.append(str(final_file))
            continue
        if artifact_policy == "screening":
            df = compact_result_dataframe(df)
            _atomic_write_parquet(df, final_file)
        attr_frame = pd.DataFrame()
        attr_frame.attrs = copy.deepcopy(df.attrs)
        attr_frames.append(attr_frame)
        row_count += int(len(df))
        collected_files.append(str(final_file))
        outcome["collected"] = True

    timing_report = _collect_timing_sidecars(root, targets)
    report_payload: dict[str, Any] = {
        "workflow": workflow.workflow_name,
        "output": str(output_path),
        "require_normal_termination": bool(require_normal_termination),
        "n_targets": len(targets),
        "n_collected": len(collected_files),
        "n_skipped": len(skipped_files),
        "n_missing": len(missing_files),
        "n_errored": len(errored_files),
        "n_rows": row_count,
        "collected_files": collected_files,
        "skipped_files": skipped_files,
        "missing_files": missing_files,
        "errored_files": errored_files,
        "errors": errors,
        "timing": timing_report,
    }
    failure_summary = _collection_failure_summary(
        skipped_files=skipped_files,
        missing_files=missing_files,
        errors=errors,
        errored_files=errored_files,
    )
    for failure in failure_summary:
        outcome = outcomes.get(failure.get("target"), {})
        if outcome.get("status") in {"failed", "interrupted", "unattempted", "stale", "running", "blocked"}:
            failure.update(problem=outcome["status"], error=outcome.get("error"),
                           attempt_id=outcome.get("attempt_id"), job_id=outcome.get("job_id"))
            if outcome.get('failed_group'):
                failure['failed_group'] = outcome['failed_group']
                failure['upstream_status'] = outcome['upstream_status']
    report_payload["target_results"] = [outcomes[target.tag] for target in targets]
    report_payload["retry_targets"] = [target.tag for target in targets
                                       if outcomes[target.tag].get("status") not in {"success", "running"}]
    history = [(attempt['attempt_id'], {record['target'] for record in attempt['records']})
               for attempt in _submission_history(root)]
    report_payload["attempt_history"] = {
        target.tag: [identity for identity, tags in history if target.tag in tags]
        for target in targets
    }
    report_payload["n_failures"] = len(failure_summary)
    report_payload["failure_summary"] = failure_summary

    if attr_frames:
        merged_attrs = merge_dataframe_attrs(
            attr_frames,
            source_files=collected_files,
            skipped_files=[*skipped_files, *missing_files, *errored_files],
        )
        _stream_parquet_files(
            [Path(path) for path in collected_files],
            output_path,
            attrs=merged_attrs,
            expected_rows=row_count,
        )
        merged = pd.read_parquet(output_path) if return_dataframe else pd.DataFrame()
    else:
        merged = pd.DataFrame()
    if artifact_policy == "screening":
        if (
            not defer_screening_cleanup
            and not failure_summary
            and len(collected_files) == len(targets)
        ):
            compaction_report = _remove_collected_target_directories(
                root,
                collected_files,
            )
        else:
            compaction_report = _empty_compaction_report()
    elif target_retention == "compact_success":
        compaction_report = _compact_collected_targets(collected_files)
    else:
        compaction_report = _empty_compaction_report()
    report_payload["target_retention"] = target_retention
    report_payload["compaction"] = compaction_report

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_submission_json(report_path, report_payload)
    from uuid import uuid4
    _atomic_write_submission_json(root / '.frust/reports' / f'{uuid4().hex}.json', report_payload)

    if skipped_files or missing_files or errored_files:
        print(
            "FRUST collection warning: "
            f"collected={len(collected_files)}, "
            f"skipped={len(skipped_files)}, "
            f"missing={len(missing_files)}, "
            f"errored={len(errored_files)}. "
            f"See {report_path}."
        )

    if not attr_frames:
        raise FileNotFoundError(
            f"No usable workflow outputs collected under {root}. "
            f"See collection report: {report_path}"
        )
    return merged


def _collect_expected_outputs_submitted(
    workflow: BaseWorkflow,
    targets: list[WorkflowTarget],
    out_dir: str | Path,
    expected_parquets: dict[str, str],
    output: str | Path,
    report: str | Path,
    require_normal_termination: bool,
    wait_jobs: list[Any] | None = None,
    target_retention: TargetRetention = "compact_success",
    artifact_policy: ArtifactPolicy = "standard",
    defer_screening_cleanup: bool = False,
    attempt_id: str | None = None,
) -> CollectionJobResult:
    """Collect outputs while keeping the Submitit result pickle small."""
    try:
        with _mutation_lock(out_dir, wait=True):
            merged = _collect_expected_outputs(
                workflow,
                targets,
                out_dir,
                expected_parquets,
                output,
                report,
                require_normal_termination,
                wait_jobs,
                target_retention,
                artifact_policy,
                False,
                defer_screening_cleanup,
                attempt_id,
            )
            payload = json.loads(Path(report).read_text())
            return CollectionJobResult(
                output_path=str(output),
                report_path=str(report),
                row_count=int(payload.get("n_rows", 0)),
                status="success",
            )

    finally:
        _mark_completed(out_dir, attempt_id, "collect", 0)


def _collect_timing_sidecars(
    root: Path,
    targets: list[WorkflowTarget],
) -> dict[str, Any]:
    """Read workflow timing sidecars for a collection report."""
    records: list[dict[str, Any]] = []
    timing_files: list[str] = []
    missing_targets: list[str] = []
    errored_files: list[dict[str, str]] = []

    for target in targets:
        target_dir = root / target.tag
        consolidated = target_dir / TARGET_TIMING_FILE
        if consolidated.exists():
            try:
                payload = json.loads(consolidated.read_text())
            except Exception as exc:
                errored_files.append({"file": str(consolidated), "error": str(exc)})
                continue
            groups = payload.get("groups", [])
            if isinstance(groups, list):
                records.extend(
                    dict(record) for record in groups if isinstance(record, Mapping)
                )
            timing_files.append(str(consolidated))
            continue

        files = sorted(target_dir.glob("*.timing.json"))
        if not files:
            missing_targets.append(target.tag)
            continue
        for path in files:
            try:
                payload = json.loads(path.read_text())
            except Exception as exc:
                errored_files.append({"file": str(path), "error": str(exc)})
                continue
            record = payload.get("record")
            if isinstance(record, Mapping):
                records.append(dict(record))
                timing_files.append(str(path))

    total_elapsed_s = sum(float(record.get("elapsed_s") or 0.0) for record in records)
    total_core_hours = 0.0
    for record in records:
        resources = record.get("resources")
        n_cores = None
        if isinstance(resources, Mapping):
            n_cores = resources.get("n_cores")
        try:
            total_core_hours += (
                float(record.get("elapsed_s") or 0.0) * float(n_cores or 0.0) / 3600.0
            )
        except (TypeError, ValueError):
            continue

    slowest = sorted(
        records,
        key=lambda record: float(record.get("elapsed_s") or 0.0),
        reverse=True,
    )[:5]
    return {
        "n_timing_files": len(timing_files),
        "n_timing_groups": len(records),
        "timing_files": timing_files,
        "missing_timing_targets": missing_targets,
        "errored_timing_files": errored_files,
        "total_elapsed_s": round(total_elapsed_s, 6),
        "total_core_hours": round(total_core_hours, 6),
        "slowest_groups": slowest,
    }


def _write_target_timing(
    target_dir: Path,
    df: pd.DataFrame,
    *,
    workflow: str,
    target: str,
    submitted_at: str | None,
    group_record: dict[str, Any],
    stage_ids: list[str],
    input_parquet: str | None,
    output_parquet: str,
) -> None:
    """Write the consolidated timing sidecar for one workflow target."""
    timing_path = target_dir / TARGET_TIMING_FILE
    existing_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    if timing_path.exists():
        try:
            payload = json.loads(timing_path.read_text())
            for record in payload.get("groups", []) or []:
                if isinstance(record, Mapping):
                    existing_groups[_timing_record_key(record)] = dict(record)
        except Exception:
            existing_groups = {}

    current_group = {
        "submitted_at": submitted_at,
        "stage_ids": list(stage_ids),
        "input_parquet": input_parquet,
        "output_parquet": output_parquet,
    }
    existing_groups[_timing_record_key(group_record)] = {
        **existing_groups.get(_timing_record_key(group_record), {}),
        **dict(group_record),
        **current_group,
    }

    records = _workflow_timing_records(df)
    groups: list[dict[str, Any]] = []
    for record in records:
        if record.get("kind") != "group":
            continue
        key = _timing_record_key(record)
        groups.append({**dict(record), **existing_groups.get(key, {})})

    stages = [dict(record) for record in records if record.get("kind") != "group"]
    write_timing_sidecar(
        timing_path,
        {
            "schema_version": 2,
            "workflow": workflow,
            "target": target,
            "updated_at": utc_timestamp(),
            "groups": groups,
            "stages": stages,
        },
    )


def _workflow_timing_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return workflow timing records from dataframe attrs."""
    timing = df.attrs.get("frust_workflow_timing", {})
    if not isinstance(timing, Mapping):
        return []
    records = timing.get("records", [])
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _timing_record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a stable key for a workflow timing record."""
    return (
        record.get("group"),
        record.get("job_id"),
        record.get("started_at"),
        record.get("finished_at"),
    )


def _compact_collected_targets(collected_files: list[str]) -> dict[str, Any]:
    """Compact successful collected target directories."""
    report = _empty_compaction_report()
    for file_name in collected_files:
        final_file = Path(file_name)
        result = _compact_successful_target(final_file.parent, final_file)
        report["targets"].append(result)
        report["n_targets"] += 1
        report["n_removed_files"] += len(result["removed_files"])
        report["removed_bytes"] += int(result["removed_bytes"])
        report["errors"].extend(result["errors"])
    return report


def _remove_collected_target_directories(
    root: Path,
    collected_files: list[str],
) -> dict[str, Any]:
    """Remove successful standalone screening targets after durable collection."""
    report = _empty_compaction_report()
    resolved_root = root.resolve()
    for file_name in collected_files:
        target_dir = Path(file_name).parent
        target_result: dict[str, Any] = {
            "target_dir": str(target_dir),
            "kept_parquet": None,
            "removed_files": [],
            "removed_bytes": 0,
            "errors": [],
        }
        try:
            resolved_target = target_dir.resolve()
            resolved_target.relative_to(resolved_root)
            if target_dir.is_symlink():
                raise ValueError("target directory is a symlink")
            files = [path for path in target_dir.rglob("*") if path.is_file()]
            target_result["removed_files"] = [str(path) for path in files]
            target_result["removed_bytes"] = sum(path.stat().st_size for path in files)
            shutil.rmtree(target_dir)
        except Exception as exc:
            target_result["errors"].append({"file": str(target_dir), "error": str(exc)})
        report["targets"].append(target_result)
        report["n_targets"] += 1
        report["n_removed_files"] += len(target_result["removed_files"])
        report["removed_bytes"] += int(target_result["removed_bytes"])
        report["errors"].extend(target_result["errors"])
    return report


def _empty_compaction_report() -> dict[str, Any]:
    """Return an empty target compaction report."""
    return {
        "n_targets": 0,
        "n_removed_files": 0,
        "removed_bytes": 0,
        "targets": [],
        "errors": [],
    }


def _compact_successful_target(target_dir: Path, final_file: Path) -> dict[str, Any]:
    """Remove intermediate target checkpoints after a successful run."""
    target_dir = Path(target_dir)
    final_file = Path(final_file)
    result: dict[str, Any] = {
        "target_dir": str(target_dir),
        "kept_parquet": str(final_file),
        "removed_files": [],
        "removed_bytes": 0,
        "errors": [],
    }
    if not target_dir.is_dir():
        result["errors"].append(
            {"file": str(target_dir), "error": "target directory does not exist"}
        )
        return result
    if not final_file.exists():
        result["errors"].append(
            {"file": str(final_file), "error": "final parquet does not exist"}
        )
        return result

    for path in sorted(target_dir.glob("*.parquet")):
        if _same_file(path, final_file) or path.name in ANALYSIS_TIER_FILES.values():
            continue
        _remove_compaction_file(path, result)

    timing_path = target_dir / TARGET_TIMING_FILE
    if timing_path.exists():
        for path in sorted(target_dir.glob("*.timing.json")):
            _remove_compaction_file(path, result)
    return result


def _remove_compaction_file(path: Path, result: dict[str, Any]) -> None:
    """Remove one file and update a compaction result."""
    try:
        size = path.stat().st_size
        path.unlink()
    except Exception as exc:
        result["errors"].append({"file": str(path), "error": str(exc)})
        return
    result["removed_files"].append(str(path))
    result["removed_bytes"] += int(size)


def _same_file(path: Path, other: Path) -> bool:
    """Return whether two paths refer to the same existing file."""
    try:
        return path.samefile(other)
    except OSError:
        return path.resolve() == other.resolve()


def _validate_target_retention(value: str) -> None:
    """Validate a workflow target retention policy."""
    if value not in {"compact_success", "all"}:
        raise ValueError("target_retention must be 'compact_success' or 'all'")


def _validate_screening_retention(
    artifact_policy: ArtifactPolicy,
    target_retention: TargetRetention,
) -> None:
    """Require safe collection compaction for screening runs."""
    if artifact_policy == "screening" and target_retention != "compact_success":
        raise ValueError(
            "artifact_policy='screening' requires target_retention='compact_success'"
        )


def _atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a parquet through a sibling temporary file and publish atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        df.to_parquet(temporary, index=False)
        check = pd.read_parquet(temporary)
        if len(check) != len(df) or list(check.columns) != list(df.columns):
            raise ValueError(f"Parquet verification failed for {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _stream_parquet_files(
    files: list[Path],
    destination: Path,
    *,
    attrs: Mapping[str, Any],
    expected_rows: int,
) -> None:
    """Stream compatible target parquets into one atomically published file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    writer = None
    source_schemas = [pq.read_schema(path).remove_metadata() for path in files]
    unified_schema = pa.unify_schemas(
        source_schemas,
        promote_options="permissive",
    )
    columns = list(unified_schema.names)
    first_metadata = dict(pq.read_schema(files[0]).metadata or {})
    first_metadata[b"PANDAS_ATTRS"] = json.dumps(
        dict(attrs),
        default=str,
        sort_keys=True,
    ).encode("utf-8")
    schema = unified_schema.with_metadata(first_metadata)
    try:
        writer = pq.ParquetWriter(temporary, schema)
        for path in files:
            frame = pd.read_parquet(path)
            frame = frame.reindex(columns=columns)
            table = pa.Table.from_pandas(
                frame,
                schema=unified_schema,
                preserve_index=False,
                safe=False,
            )
            writer.write_table(table.replace_schema_metadata(schema.metadata))
        writer.close()
        writer = None
        parquet = pq.ParquetFile(temporary)
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError(
                f"Collected row count mismatch: expected {expected_rows}, "
                f"found {parquet.metadata.num_rows}"
            )
        if list(parquet.schema_arrow.names) != columns:
            raise ValueError("Collected parquet schema verification failed")
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _remove_consumed_hessians(target_dir: Path) -> None:
    """Remove Hessians only after the dependent TS optimization succeeds."""
    for path in Path(target_dir).rglob("*.hess"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _stage_engine(stage: StageDef, spec: CalculatorSpec | None) -> str | None:
    """Return the inspection-table engine label for a workflow stage."""
    if stage.kind == "prepare":
        return "prepare"
    if stage.kind == "filter":
        return "filter"
    if stage.kind == "prune":
        return "prism_pruner"
    if spec is None:
        return None
    return spec.engine


def _format_stage_options(options: Mapping[str, Any] | None) -> str | None:
    """Format calculator options for ``BaseWorkflow.show_stages``."""
    if not options:
        return None
    parts: list[str] = []
    for key, value in options.items():
        if value is None or value is True:
            parts.append(str(key))
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _format_stage_solvent(spec: CalculatorSpec | None) -> str | None:
    """Return a compact solvent label for workflow stage inspection."""
    if spec is None or spec.solvent is None:
        return None
    return f"{(spec.solvation_model or 'smd').upper()}({spec.solvent})"


def _format_planned_input(value: str | None) -> str | None:
    """Return a Markdown-safe one-line representation of an input block."""
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", r"\n")


def _format_planned_mapping(value: Mapping[str, Any] | None) -> str | None:
    """Return a stable compact representation of planned keyword arguments."""
    if not value:
        return None
    return json.dumps(dict(value), default=str, sort_keys=True)


def _format_planned_sequence(value: list[str] | None) -> str | None:
    """Return a compact representation of planned file lists."""
    if not value:
        return None
    return json.dumps(list(value))


def _format_pruning_stage_options(options: Mapping[str, Any] | None) -> str | None:
    """Format pruning options for ``BaseWorkflow.show_stages``."""
    if not options:
        return None
    modes = tuple(options.get("modes") or ())
    parts = ["modes=" + ",".join(map(str, modes))] if modes else []
    if "moi" in modes:
        parts.append(f"moi_max_deviation={options.get('moi_max_deviation')}")
    if "rmsd" in modes or "rot_corr_rmsd" in modes:
        parts.append(f"rmsd_max_rmsd={options.get('rmsd_max_rmsd')}")
        if options.get("rmsd_max_dev") is not None:
            parts.append(f"rmsd_max_dev={options.get('rmsd_max_dev')}")
    if "rot_corr_rmsd" in modes:
        parts.append(f"graph_source={options.get('graph_source')}")
    if options.get("coords_col") is not None:
        parts.append(f"coords_col={options.get('coords_col')}")
    if options.get("energy_col") is not None:
        parts.append(f"energy_col={options.get('energy_col')}")
    return " ".join(parts) if parts else _format_stage_options(options)


def _coerce_method(method: MethodPlan | str | None) -> MethodPlan:
    """Normalize user method input to a method plan.

    Parameters
    ----------
    method : MethodPlan or str or None
        Explicit method plan, registered preset name, or ``None`` for the
        workflow default. Built-in preset strings are ``"r2scan-3c"`` for ORCA
        r2SCAN-3c composite DFT stages, ``"wb97xd3-631g"`` for the default ORCA
        wB97X-D3/6-31G** workflow, ``"r2scan-3c-solv"`` and
        ``"wb97xd3-631g-solv"`` for solvent-inclusive DFT stages, and
        ``"r2scan-def2svp"`` for ORCA R2SCAN/def2-SVP DFT stages.

    Returns
    -------
    MethodPlan
        Resolved calculator plan.
    """
    if method is None:
        return method_preset("wb97xd3-631g")
    if isinstance(method, str):
        return method_preset(method)
    if isinstance(method, MethodPlan) or (
        method.__class__.__name__ == "MethodPlan"
        and hasattr(method, "stages")
        and hasattr(method, "for_stage")
    ):
        return method
    raise TypeError("method must be a MethodPlan, preset name, or None")


def _resource_for_group(
    group_name: str,
    group: list[StageDef],
    resources: dict[str, Resources] | None,
    *,
    default: Resources,
) -> Resources:
    """Resolve scheduler resources for a stage group.

    Parameters
    ----------
    group_name : str
        Resource key for the stage group, usually ``"init"`` or the last stage
        id in the group.
    group : list of StageDef
        Stages in the group. Individual stage ids are accepted as fallback keys.
    resources : dict of str to Resources or None
        User-provided overrides.
    default : Resources
        Resources used when no override matches.

    Returns
    -------
    Resources
        Resource settings for the submitted job.
    """
    if resources is None:
        return default
    if group_name in resources:
        return resources[group_name]
    legacy_group = _LEGACY_STAGE_RESOURCE_KEYS.get(group_name)
    if legacy_group in resources:
        return resources[legacy_group]
    for stage in group:
        if stage.id in resources:
            return resources[stage.id]
        legacy_stage = _LEGACY_STAGE_RESOURCE_KEYS.get(stage.id)
        if legacy_stage in resources:
            return resources[legacy_stage]
    return default


def _next_parquet(current: str | None, group_name: str) -> str:
    """Return the staged parquet filename after a group.

    Parameters
    ----------
    current : str or None
        Previous staged parquet filename. ``None`` starts the chain.
    group_name : str
        New stage-group name.

    Returns
    -------
    str
        ``"init.parquet"`` for the first group, then dotted filenames such as
        ``"init.dft_hessian.dft_ts_opt.parquet"``.
    """
    if current is None:
        return "init.parquet"
    stem = current.rsplit(".", 1)[0]
    return f"{stem}.{sanitize_tag(group_name)}.parquet"


def _deepest_parquet(target_dir: Path) -> Path | None:
    """Return the final-looking parquet file from one target directory.

    Parameters
    ----------
    target_dir : pathlib.Path
        Directory for one workflow target.

    Returns
    -------
    pathlib.Path or None
        ``final.parquet`` when present; otherwise the staged parquet with the
        deepest dotted name; otherwise ``None``.
    """
    if not target_dir.is_dir():
        return None
    final_file = target_dir / "final.parquet"
    if final_file.exists():
        return final_file
    tier_names = set(ANALYSIS_TIER_FILES.values())
    files = sorted(
        path for path in target_dir.glob("*.parquet") if path.name not in tier_names
    )
    if not files:
        return None
    return max(files, key=lambda path: (len(path.suffixes), path.name))


def _all_normal_terminated(df: pd.DataFrame) -> bool:
    """Return whether all normal-termination columns are true.

    Parameters
    ----------
    df : pandas.DataFrame
        Workflow output dataframe.

    Returns
    -------
    bool
        ``True`` when there are no ``"-NT"`` columns or when all such columns are
        truthy after missing values are treated as failures.
    """
    nt_cols = normal_termination_columns(df)
    if not nt_cols:
        return True
    return bool(df[nt_cols].fillna(False).astype(bool).all().all())


def _options_resources(options: ExecutionOptions) -> dict[str, Any]:
    """Return workflow execution resources for timing metadata."""
    return {
        "n_cores": options.n_cores,
        "memory_gb": options.mem_gb,
    }


def _current_job_id() -> str | None:
    """Return the active scheduler job id when one is visible to Python."""
    if os.getenv("SLURM_JOB_ID"):
        return os.getenv("SLURM_JOB_ID")
    try:
        import submitit

        return str(submitit.JobEnvironment().job_id)
    except Exception:
        return None


def _log_workflow_stage_summary(
    *,
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage: StageDef,
    df: pd.DataFrame,
    input_rows: int | None,
    options: ExecutionOptions,
) -> None:
    """Log compact workflow-stage summaries for non-calculator stages."""
    if stage.kind not in {"prepare", "filter"}:
        return

    message = _workflow_stage_summary_message(stage, df, input_rows=input_rows)
    if not message:
        return

    logger = make_stepper_logger(
        workflow._step_type_for_target(target),
        debug=options.debug,
        job_id=_current_job_id(),
    )
    logger.info(message)


def _workflow_stage_summary_message(
    stage: StageDef,
    df: pd.DataFrame,
    *,
    input_rows: int | None,
) -> str | None:
    """Return a compact log message for a workflow-level stage."""
    if stage.kind == "prepare":
        conformers = df.attrs.get("frust_conformers")
        if (
            isinstance(conformers, Mapping)
            and conformers.get("source") == "Stepper.build_initial_df"
        ):
            return None
        return conformer_generation_summary(df, label=stage.name)
    if stage.kind == "filter":
        return filter_summary(
            name=stage.name,
            input_rows=input_rows,
            output_rows=int(len(df)),
        )
    return None


def _apply_calculator(
    step: Stepper,
    df: pd.DataFrame,
    stage: StageDef,
    spec: CalculatorSpec,
    *,
    uma_server_cores: int,
) -> pd.DataFrame:
    """Dispatch one calculator stage through Stepper.

    Parameters
    ----------
    step : frust.stepper.Stepper
        Stepper configured for the current target and runtime options.
    df : pandas.DataFrame
        Input dataframe for this stage.
    stage : StageDef
        Workflow stage being run.
    spec : CalculatorSpec
        Calculator engine, options, and extra input selected from the method
        plan.
    uma_server_cores : int
        Stable server budget from the workflow job allocation.

    Returns
    -------
    pandas.DataFrame
        Dataframe returned by ``Stepper.xtb(...)``, ``Stepper.gxtb(...)``, or
        ``Stepper.orca(...)``.
    """
    if stage.lowest is not None and stage.rank_by != stage.id:
        raise ValueError(
            f"Calculator stage {stage.id!r} with lowest= must set rank_by={stage.id!r}; "
            "use an explicit filter stage to rank by a different calculation"
        )
    kwargs = {
        "name": stage.id,
        "options": spec.options,
        "constraint": stage.constraint,
        "lowest": stage.lowest,
        "n_cores": stage.n_cores,
        **spec.kwargs,
    }
    if spec.engine == "orca" and kwargs.get("uma") is not None and kwargs.get("uma_server", True):
        kwargs.setdefault("uma_server_cores", uma_server_cores)
    if spec.engine == "xtb":
        return step.xtb(
            df,
            detailed_inp_str=spec.detailed_inp_str,
            **kwargs,
        )
    if spec.engine == "gxtb":
        return step.gxtb(
            df,
            detailed_inp_str=spec.detailed_inp_str,
            **kwargs,
        )
    if spec.engine == "orca":
        return step.orca(
            df,
            xtra_inp_str=spec.xtra_inp_str,
            read_files=stage.read_files,
            use_last_hess=stage.use_last_hess,
            save_files=stage.save_files,
            **kwargs,
        )
    raise ValueError(f"Unsupported engine {spec.engine!r}")


def _run_stage_calculation(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage: StageDef,
    df: pd.DataFrame,
    *,
    save_dir: Path | None,
    options: ExecutionOptions,
) -> pd.DataFrame:
    """Run one calculation or filter stage.

    Parameters
    ----------
    workflow : BaseWorkflow
        Workflow supplying method plan and target step type.
    target : WorkflowTarget
        Target being processed.
    stage : StageDef
        Non-prepare stage to run.
    df : pandas.DataFrame
        Input dataframe.
    save_dir : pathlib.Path or None
        Target output directory for Stepper output folders.
    options : ExecutionOptions
        Runtime options forwarded to Stepper.

    Returns
    -------
    pandas.DataFrame
        Stage output dataframe.
    """
    if stage.kind == "filter":
        if not stage.rank_by:
            raise ValueError(f"Filter stage {stage.id!r} must define rank_by")
        input_rows = len(df)
        energy_col = output_column(stage.rank_by, "electronic_energy")
        if stage.id == "uma_rank_filter":
            energies = pd.to_numeric(df[energy_col], errors="coerce")
            df = df.loc[np.isfinite(energies)].copy()
        result = lowest_energy_rows(
            df,
            n=stage.lowest or 1,
            energy_col=energy_col,
        )
        steps = dict(result.attrs.get("frust_steps", {}))
        steps[stage.id] = {
            "engine": "filter",
            "filtering": {
                "lowest": stage.lowest or 1,
                "energy_col": energy_col,
                "input_rows": input_rows,
                "output_rows": len(result),
                "dropped_rows": input_rows - len(result),
            },
        }
        result.attrs["frust_steps"] = steps
        return result

    step = Stepper(
        step_type=workflow._step_type_for_target(target),
        n_cores=options.n_cores,
        memory_gb=options.mem_gb,
        debug=options.debug,
        output_base=save_dir,
        save_output_dir=options.save_output_dir,
        work_dir=options.work_dir,
    )
    if stage.kind == "prune":
        return step.prune_conformers(
            df,
            name=stage.name,
            **(stage.prune_options or {}),
        )

    spec = workflow.method.for_stage(stage.method_stage or stage.id)
    return _apply_calculator(step, df, stage, spec, uma_server_cores=options.n_cores)


def _write_analysis_tier_snapshot(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage: StageDef,
    df: pd.DataFrame,
    *,
    save_dir: Path | None,
    artifact_policy: ArtifactPolicy = "standard",
) -> None:
    """Persist an exact lower-tier winner without filtering the main pipeline.

    Parameters
    ----------
    workflow : BaseWorkflow
        Workflow whose requested depth determines the nested tiers.
    target : WorkflowTarget
        Scientific target represented by the dataframe.
    stage : StageDef
        Completed stage. Snapshots are written after ``xtb_opt`` and, for a
        full workflow, after ``dft_rank_sp``.
    df : pandas.DataFrame
        Post-stage ensemble. Selection occurs on a copy so subsequent full
        refinement still receives the complete retained ensemble.
    save_dir : pathlib.Path or None
        Target directory. Portable tier snapshots require an output directory.
    """
    if save_dir is None or workflow.result_profile is None:
        return
    requested_level = str(
        getattr(
            workflow,
            "calculation_level",
            "full" if workflow.dft else "low_cost",
        )
    )
    tier: str | None = None
    screening_opt_stage = (
        "uma_opt" if "uma_opt" in workflow.method.stages else "xtb_opt"
    )
    include_dft_rank_sp = bool(getattr(workflow, "include_dft_rank_sp", True))
    if stage.id == screening_opt_stage and requested_level in {
        "uma_ranked", "dft_ranked", "full"
    }:
        tier = "low_cost"
    elif stage.id == "uma_rank_sp" and requested_level == "full":
        tier = "uma_ranked"
    elif stage.id == "dft_rank_sp" and requested_level == "full":
        tier = "dft_ranked"
    if tier is None:
        return

    energy_column = output_column(stage.id, "electronic_energy")
    source = df.copy()
    if tier == "uma_ranked":
        energies = pd.to_numeric(source[energy_column], errors="coerce")
        source = source.loc[np.isfinite(energies)].copy()
    snapshot = lowest_energy_rows(
        source,
        n=1,
        energy_col=energy_column,
    )
    snapshot = _attach_workflow_attrs(snapshot, workflow=workflow, target=target)
    snapshot.attrs["frust_workflow"]["calculation_level"] = tier
    attach_result_contract(
        snapshot,
        workflow.result_profile,
        dft=False,
        calculation_level=tier,
        include_terminal_solv_sp=False,
        screening_opt_stage=screening_opt_stage,
        include_dft_rank_sp=include_dft_rank_sp,
        thermochemistry=None,
        ranking_stage=("uma_rank_sp" if tier == "uma_ranked" else None),
    )
    contract = snapshot.attrs["frust_results"]
    analysis_column = str(contract["columns"]["analysis"]["electronic_energy"])
    analysis_stage = analysis_column.removesuffix("-EE")
    optimized_column = str(contract["columns"]["optimized"]["coords"])
    optimized_stage = optimized_column.removesuffix("-oc")
    contract["energy_protocol"] = {
        "calculation_level": tier,
        "analysis_stage": analysis_stage,
        "geometry_stage": optimized_stage,
        "calculator": workflow.method.for_stage(analysis_stage).to_dict(),
    }
    snapshot.attrs["frust_analysis_tier"] = {
        "schema_version": 1,
        "analysis_level": tier,
        "requested_level": requested_level,
        "selection_stage": stage.id,
        "selection_energy_column": energy_column,
        "source_rows": int(len(df)),
        "selected_rows": int(len(snapshot)),
    }
    if artifact_policy == "screening":
        snapshot = compact_result_dataframe(snapshot)
    _atomic_write_parquet(snapshot, Path(save_dir) / ANALYSIS_TIER_FILES[tier])


def _attach_workflow_attrs(
    df: pd.DataFrame,
    *,
    workflow: BaseWorkflow,
    target: WorkflowTarget,
) -> pd.DataFrame:
    """Attach compact workflow provenance to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Stage output dataframe.
    workflow : BaseWorkflow
        Workflow that produced the dataframe.
    target : WorkflowTarget
        Target that produced the dataframe.

    Returns
    -------
    pandas.DataFrame
        Same dataframe with ``df.attrs["frust_workflow"]`` populated.
    """
    df = canonical_state_columns(df)
    df.attrs.setdefault("frust_workflow", {})
    df.attrs["frust_workflow"].update(
        {
            "workflow": workflow.workflow_name,
            "method": workflow.method.name,
            "method_fingerprint": workflow.method.fingerprint(),
            "calculation_level": getattr(
                workflow,
                "calculation_level",
                "full" if workflow.dft else "low_cost",
            ),
            "target": target.tag,
            "result_profile": workflow.result_profile,
        }
    )
    structure_options = workflow._structure_build_kwargs()
    if workflow.result_profile == "transition_state" and workflow.method.result_family == "uma":
        df.attrs["frust_workflow"]["ts_refine_n"] = int(workflow.ts_refine_n)
        df.attrs["frust_workflow"]["screen_top_n"] = int(workflow.top_n)
    if "uma_rank_sp" in workflow.method.stages:
        df.attrs["frust_workflow"]["ranking_stage"] = "uma_rank_sp"
        df.attrs["frust_workflow"]["uma_rank_top_n"] = int(
            workflow.uma_rank_top_n
        )
        df.attrs["frust_workflow"]["screen_top_n"] = int(workflow.top_n)
    if "spec_profile" in structure_options:
        df.attrs["frust_workflow"]["guess_profile"] = structure_options["spec_profile"]
        df.attrs["frust_workflow"]["guess_profile_match"] = structure_options.get(
            "spec_match", "prefer-exact"
        )
    if workflow.result_profile is not None:
        calculation_level = getattr(
            workflow,
            "calculation_level",
            "full" if workflow.dft else None,
        )
        attach_result_contract(
            df,
            workflow.result_profile,
            dft=workflow.dft and workflow.method.result_family == "dft",
            calculation_level=calculation_level,
            full_method=workflow.method.result_family,
            include_terminal_solv_sp=workflow.method.include_terminal_solv_sp,
            screening_opt_stage=(
                "uma_opt" if "uma_opt" in workflow.method.stages else "xtb_opt"
            ),
            include_dft_rank_sp=bool(getattr(workflow, "include_dft_rank_sp", True)),
            thermochemistry=workflow.method.thermochemistry,
            ranking_stage=(
                "uma_rank_sp" if "uma_rank_sp" in workflow.method.stages else None
            ),
        )
        contract = df.attrs["frust_results"]
        analysis_column = str(contract["columns"]["analysis"]["electronic_energy"])
        analysis_stage = analysis_column.removesuffix("-EE")
        optimized_column = str(contract["columns"]["optimized"]["coords"])
        optimized_stage = optimized_column.removesuffix("-oc")
        contract["energy_protocol"] = {
            "calculation_level": contract["calculation_level"],
            "analysis_stage": analysis_stage,
            "geometry_stage": optimized_stage,
            "calculator": workflow.method.for_stage(analysis_stage).to_dict(),
        }
        if analysis_stage == "uma_solv_sp":
            contract["energy_protocol"]["frequency_calculator"] = (
                workflow.method.for_stage("uma_freq").to_dict()
            )
            contract["energy_protocol"]["thermochemistry"] = contract["thermochemistry"]
    stamp_schema(df)
    return df
