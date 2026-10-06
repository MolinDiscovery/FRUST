"""Composed end-to-end catalyst-screen calculation workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from frust.artifacts import (
    ArtifactPolicy,
    compact_result_dataframe,
    validate_artifact_policy,
)
from frust.cluster.config import ClusterConfig, JobSubmissionResult, Resources
from frust.cluster.executor import create_executor, update_executor_with_dependencies
from frust.cluster.submission import (
    _plan_submission, _mutation_lock, _job_terminal, _completion_path, _mark_completed,
    _atomic_write_submission_json,
    _validate_array_scheduler_options, _validate_array_size,
)
from frust.screen import expand as expand_screen
from frust.screen import read as read_screen
from frust.screen.cleanup import cleanup_submitit_jobs, initialize_submitit_directory
from frust.screen.references import ReferenceLibrary, ReferenceRecord, ReusePolicy
from frust.screen.runs import ScreenRun, build_analysis
from frust.schema import normal_termination_columns
from frust.structures import StructureTarget
from frust.structures.specs import DIMER_STATES
from frust.utils.dataframes import merge_dataframe_attrs
from frust.workflows.core import ANALYSIS_TIER_FILES
from frust.workflows.factories import Int3Workflow, MolsWorkflow, ScreenTSWorkflow
from frust.workflows.factories import SeededWb97TSWorkflow
from frust.workflows.methods import (
    CalculationLevel,
    MethodPlan,
    RankingPlan,
    ScreeningPlan,
    apply_screening_plan,
    preset as method_preset,
    ranking_preset,
    screening_preset,
    with_ranking_solvation,
)

ScreenScope = Literal["barriers", "full_cycle"]
DimerReference = Literal[
    "lowest",
    "dimer",
    "dimer_bh_bridged",
    "dimer_eight_membered",
]
DEFAULT_G_CORRECTIONS = {"TS1": -1.89, "TS3": -1.89}
DEFAULT_FINALIZE_RESOURCES = Resources(cpus=2, mem_gb=4, timeout_min=120)


@dataclass(frozen=True)
class CatalystScreenTarget:
    """One target in a composed catalyst-screen plan."""

    branch: str
    target: StructureTarget
    action: Literal["calculate", "reuse"] = "calculate"
    reference_id: str | None = None


@dataclass(frozen=True)
class ScreenSubmissionResult:
    """Submission summary for a composed catalyst-screen run.

    Parameters
    ----------
    run_dir : str
        Portable run-bundle directory.
    child_submissions : dict
        Submission results for each homogeneous calculation branch.
    finalization_job_id : str or int or None
        Job that snapshots references and generates portable analysis.
    backend : str
        Cluster backend name.
    submitit_dir : str or None
        Managed screening-mode bookkeeping directory. Standard mode returns
        ``None`` and continues to use ``ClusterConfig.log_dir``.
    """

    run_dir: str
    child_submissions: dict[str, JobSubmissionResult]
    finalization_job_id: str | int | None
    backend: str
    submitit_dir: str | None = None


@dataclass(frozen=True)
class FinalizationJobResult:
    """Small Submitit result returned by the screening finalizer."""

    run_dir: str
    report_path: str
    status: str


class CatalystScreenWorkflow:
    """Coordinate TS, reference, and optional full-cycle workflows."""

    workflow_name = "catalyst_screen"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
        screening: ScreeningPlan | str = "gxtb-default",
        level: CalculationLevel = "full",
        method: MethodPlan | str | None = None,
        ranking: RankingPlan | str | None = None,
        ranking_solvation: str = "method",
        spec_profile: str = "auto",
        spec_match: str = "prefer-exact",
        include_dft_rank_sp: bool | None = None,
        scope: ScreenScope = "barriers",
        dimer_reference: DimerReference = "lowest",
        g_corrections_kcal_mol: dict[str, float] | None = None,
        reference_store: str | Path | None = None,
        reuse_policy: ReusePolicy = "approved",
        n_confs: int | None = None,
        top_n: int = 20,
        uma_rank_top_n: int = 1,
        ts_refine_n: int = 3,
        prune_initial: bool | dict[str, Any] = True,
    ) -> None:
        if (csv_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one of csv_path or dataframe")
        if scope not in {"barriers", "full_cycle"}:
            raise ValueError("scope must be 'barriers' or 'full_cycle'")
        if dimer_reference not in {"lowest", *DIMER_STATES}:
            raise ValueError(
                "dimer_reference must be 'lowest', 'dimer', "
                "'dimer_bh_bridged', or 'dimer_eight_membered'"
            )
        if reuse_policy not in {"approved", "auto_valid"}:
            raise ValueError("reuse_policy must be 'approved' or 'auto_valid'")
        normalized_level = str(level).strip().lower()
        if normalized_level not in {"low_cost", "uma_ranked", "dft_ranked", "full"}:
            raise ValueError("level must be 'low_cost', 'uma_ranked', 'dft_ranked', or 'full'")
        self.csv_path = None if csv_path is None else Path(csv_path)
        self.dataframe = None if dataframe is None else dataframe.copy()
        self.ts_types = tuple(str(value).upper() for value in ts_types)
        self.level: CalculationLevel = normalized_level  # type: ignore[assignment]
        self.screening = _coerce_screening(screening)
        if ranking is None:
            self.ranking = None
        elif isinstance(ranking, RankingPlan):
            self.ranking = ranking
        elif isinstance(ranking, str):
            self.ranking = ranking_preset(ranking)
        else:
            raise TypeError("ranking must be a RankingPlan, preset name, or None")
        if self.ranking is None and self.level == "uma_ranked":
            raise ValueError("level='uma_ranked' requires UMA ranking")
        if self.ranking is not None:
            if self.level not in {"uma_ranked", "full"}:
                raise ValueError("UMA ranking requires level='uma_ranked' or 'full'")
            if "xtb_opt" not in self.screening.stages:
                raise ValueError("UMA ranking requires g-xTB screening")
            if ranking_solvation != "method":
                raise ValueError("ranking_solvation applies only to DFT ranking")
        self.include_dft_rank_sp = (
            "uma_opt" not in self.screening.stages and self.ranking is None
            if include_dft_rank_sp is None
            else bool(include_dft_rank_sp)
        )
        if self.ranking is not None and self.include_dft_rank_sp:
            raise ValueError("UMA ranking requires include_dft_rank_sp=False")
        self.spec_profile = str(spec_profile).strip().lower()
        self.spec_match = str(spec_match).strip().lower()
        composed_method = apply_screening_plan(_coerce_method(method), self.screening)
        if self.ranking is not None:
            if composed_method.result_family != "dft":
                raise ValueError("UMA reranking requires a DFT final method")
            composed_method = composed_method.with_stage(
                self.ranking.stage_id, self.ranking.calculator
            )
        if composed_method.result_family == "uma" and self.level == "full":
            if scope != "barriers":
                raise ValueError("Full UMA currently supports scope='barriers' only")
            if "uma_opt" not in self.screening.stages:
                raise ValueError("Full UMA requires UMA screening in the same environment")
            screening_potential = self.screening.stages["uma_opt"].kwargs
            final_potential = composed_method.for_stage("uma_freq").kwargs
            if any(
                screening_potential.get(key) != final_potential.get(key)
                for key in ("uma", "uma_xtb_alpb")
            ):
                raise ValueError("Full UMA screening and final stages must use one model and environment")
            if self.include_dft_rank_sp:
                raise ValueError("Full UMA does not include DFT ranking")
            if ranking_solvation != "method":
                raise ValueError("ranking_solvation applies only to DFT ranking")
            self.method = composed_method
            self.ranking_solvation = {
                "requested": "method", "model": None, "solvent": None
            }
        else:
            self.method, self.ranking_solvation = with_ranking_solvation(
                composed_method,
                ranking_solvation,
            )
        self.ranking_solvation["applied"] = self.level == "dft_ranked" or (
            self.level == "full" and self.include_dft_rank_sp
        )
        if self.level == "full" and self.method.thermochemistry is None:
            raise ValueError(
                f"Method plan {self.method.name!r} needs a ThermochemistrySpec "
                "for full catalyst-screen analysis"
            )
        self.scope = scope
        self.dimer_reference: DimerReference = dimer_reference
        self.g_corrections_kcal_mol = {
            **DEFAULT_G_CORRECTIONS,
            **{
                str(key): float(value)
                for key, value in (g_corrections_kcal_mol or {}).items()
            },
        }
        configured_store = reference_store or os.environ.get("FRUST_REFERENCE_STORE")
        self.reference_store = (
            None if configured_store is None else Path(configured_store)
        )
        self.reuse_policy = reuse_policy
        self.n_confs = n_confs
        self.top_n = int(top_n)
        self.uma_rank_top_n = int(uma_rank_top_n)
        if self.uma_rank_top_n < 1:
            raise ValueError("uma_rank_top_n must be positive")
        self.ts_refine_n = int(ts_refine_n)
        if self.ts_refine_n < 1:
            raise ValueError("ts_refine_n must be positive")
        self.prune_initial = prune_initial
        self._components_cache: pd.DataFrame | None = None
        self._systems_cache: pd.DataFrame | None = None
        self._children_cache: dict[str, Any] | None = None

    def components(self) -> pd.DataFrame:
        """Return the normalized component table."""
        if self._components_cache is None:
            source = self.dataframe if self.dataframe is not None else self.csv_path
            self._components_cache = read_screen(source, strict=True)
        return self._components_cache.copy()

    def systems(self) -> pd.DataFrame:
        """Return expanded substrate/catalyst systems."""
        if self._systems_cache is None:
            self._systems_cache = expand_screen(self.components())
        return self._systems_cache.copy()

    def children(self) -> dict[str, Any]:
        """Return the homogeneous workflows coordinated by this run."""
        if self._children_cache is None:
            components = self.components()
            children: dict[str, Any] = {
                "transition_states": ScreenTSWorkflow(
                    dataframe=components,
                    ts_types=self.ts_types,
                    method=self.method,
                    spec_profile=self.spec_profile,
                    spec_match=self.spec_match,
                    include_dft_rank_sp=self.include_dft_rank_sp,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    uma_rank_top_n=self.uma_rank_top_n,
                    ts_refine_n=self.ts_refine_n,
                    calculation_level=self.level,
                    prune_initial=self.prune_initial,
                ),
                "references": MolsWorkflow(
                    dataframe=components,
                    select_mols=self._reference_states(),
                    method=self.method,
                    include_dft_rank_sp=self.include_dft_rank_sp,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    uma_rank_top_n=self.uma_rank_top_n,
                    calculation_level=self.level,
                    prune_initial=self.prune_initial,
                ),
            }
            if self.scope == "full_cycle":
                children["cycle_molecules"] = MolsWorkflow(
                    dataframe=components,
                    select_mols=["int1", "int2", "HBpin-ligand"],
                    method=self.method,
                    include_dft_rank_sp=self.include_dft_rank_sp,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    uma_rank_top_n=self.uma_rank_top_n,
                    calculation_level=self.level,
                    prune_initial=self.prune_initial,
                )
                children["int3"] = Int3Workflow(
                    dataframe=components,
                    method=self.method,
                    spec_profile=self.spec_profile,
                    spec_match=self.spec_match,
                    include_dft_rank_sp=self.include_dft_rank_sp,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    uma_rank_top_n=self.uma_rank_top_n,
                    calculation_level=self.level,
                    prune_initial=self.prune_initial,
                )
            self._children_cache = children
        return dict(self._children_cache)

    def targets(self) -> list[CatalystScreenTarget]:
        """Return lightweight targets and reference reuse decisions."""
        children = self.children()
        planned: list[CatalystScreenTarget] = []
        for branch, workflow in children.items():
            for target in workflow.targets():
                action: Literal["calculate", "reuse"] = "calculate"
                reference_id: str | None = None
                if branch == "references":
                    record = self._cached_reference(target)
                    if record is not None:
                        action = "reuse"
                        reference_id = record.reference_id
                planned.append(
                    CatalystScreenTarget(branch, target, action, reference_id)
                )
        return planned

    def plan(self) -> pd.DataFrame:
        """Return a compact calculation and cache-action table."""
        rows = []
        for item in self.targets():
            target = item.target
            rows.append(
                {
                    "branch": item.branch,
                    "state_id": target.state_id,
                    "system_name": target.system.system_name,
                    "rpos": target.rpos,
                    "scope": target.scope,
                    "action": item.action,
                    "reference_id": item.reference_id,
                    "target": target.tag,
                }
            )
        result = pd.DataFrame(rows)
        result.attrs["calculation_level"] = self.level
        result.attrs["screening"] = self.screening.to_dict()
        result.attrs["screening_fingerprint"] = self.screening.fingerprint()
        result.attrs["method"] = self.method.name
        result.attrs["method_fingerprint"] = self.method.fingerprint()
        if self.ranking is not None:
            result.attrs["ranking"] = self.ranking.to_dict()
            result.attrs["uma_rank_top_n"] = self.uma_rank_top_n
        result.attrs["ranking_solvation"] = self.ranking_solvation
        result.attrs["thermochemistry"] = (
            None
            if self.method.thermochemistry is None
            else self.method.thermochemistry.to_dict()
        )
        result.attrs["scope"] = self.scope
        result.attrs["dimer_reference"] = self.dimer_reference
        return result

    def show_stages(
        self, execution: str | None = None, detail: str = "summary"
    ) -> pd.DataFrame:
        """Return child stage graphs with an explicit calculation branch."""
        frames = []
        for branch, workflow in self.children().items():
            table = workflow.show_stages(execution=execution, detail=detail).copy()
            table.insert(0, "branch", branch)
            frames.append(table)
        return pd.concat(frames, ignore_index=True)

    def preview(
        self,
        *,
        targets: list[int] | None = None,
        n_confs: int | None = 1,
        n_cores: int = 1,
    ) -> pd.DataFrame:
        """Preview selected typed structures without running calculators."""
        planned = self.targets()
        selected = planned if targets is None else [planned[index] for index in targets]
        frames: list[pd.DataFrame] = []
        children = self.children()
        for branch in children:
            branch_items = [item for item in selected if item.branch == branch]
            if not branch_items:
                continue
            cached = [item for item in branch_items if item.action == "reuse"]
            calculated = [item for item in branch_items if item.action == "calculate"]
            for item in cached:
                record = self._cached_reference(item.target)
                if record is not None:
                    frame = record.materialize(item.target)
                    frame["preview_source"] = "reference_library"
                    frames.append(frame)
            if calculated:
                frame = children[branch].preview(
                    targets=[item.target for item in calculated],
                    n_confs=n_confs,
                    n_cores=n_cores,
                )
                frame["preview_source"] = "generated"
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def run(
        self,
        *,
        out_dir: str | Path,
        execution: str | None = None,
        n_cores: int = 10,
        mem_gb: int = 20,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
        target_retention: str = "compact_success",
        artifact_policy: ArtifactPolicy = "standard",
        uma_oet_tools: str | Path | None = None,
    ) -> ScreenRun:
        """Run child workflows locally and build the screen analysis.

        Parameters
        ----------
        uma_oet_tools : str, pathlib.Path, or None, optional
            OET runtime selected inside each child workflow group containing
            UMA stages. Other arguments control the child workflows and run
            artifacts as described by :meth:`submit`.
        """
        artifact_policy = validate_artifact_policy(artifact_policy)
        if artifact_policy == "screening" and target_retention != "compact_success":
            raise ValueError(
                "artifact_policy='screening' requires target_retention='compact_success'"
            )
        if not save_output_dir:
            raise ValueError(
                "catalyst_screen requires save_output_dir=True so reference "
                "calculator evidence remains portable"
            )
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._write_manifest(root, artifact_policy=artifact_policy)
        children = self.children()
        reference_items = [
            item for item in self.targets() if item.branch == "references"
        ]
        self._snapshot_reused_references(
            root,
            reference_items,
            artifact_policy=artifact_policy,
        )

        for branch, workflow in children.items():
            branch_dir = _branch_dir(root, branch)
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_targets = None
            output_name = "merged.parquet"
            if branch == "references":
                branch_targets = [
                    item.target
                    for item in reference_items
                    if item.action == "calculate"
                ]
                output_name = "computed.parquet"
                if not branch_targets:
                    _write_verified_parquet(pd.DataFrame(), branch_dir / output_name)
                    continue
            frame = workflow.run(
                targets=branch_targets,
                out_dir=branch_dir,
                execution=execution,
                n_cores=n_cores,
                mem_gb=mem_gb,
                debug=debug,
                save_output_dir=save_output_dir,
                work_dir=work_dir,
                target_retention=target_retention,
                artifact_policy=artifact_policy,
                uma_oet_tools=uma_oet_tools,
            )
            _write_verified_parquet(frame, branch_dir / output_name)
        _finalize_run(self, root, artifact_policy=artifact_policy)
        return ScreenRun(root)

    def submit(
        self,
        *,
        out_dir: str | Path,
        cluster: ClusterConfig,
        execution: str | None = None,
        stage_resources: dict[str, Resources] | None = None,
        array: bool = False,
        array_parallelism: int | Mapping[str, int] | None = None,
        targets_per_task: int = 1,
        retry: bool = False,
        targets: Mapping[str, Any] | None = None,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
        collect_require_normal_termination: bool = True,
        collect_resources: Resources | None = None,
        finalize_resources: Resources | None = None,
        target_retention: str = "compact_success",
        artifact_policy: ArtifactPolicy = "standard",
        uma_oet_tools: str | Path | None = None,
    ) -> ScreenSubmissionResult:
        """Submit the complete catalyst screen and its analysis finalizer.

        The method submits a calculation chain per target in each child
        branch. Barrier runs contain ``transition_states`` and ``references``;
        full-cycle runs additionally contain ``cycle_molecules`` and ``int3``.
        Every branch receives a collector, followed by one ``afterany``
        finalizer that snapshots references and builds the portable state,
        barrier, profile, quality, and report artifacts.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Portable run directory. It receives ``manifest.json``, calculation
            branches, collected parquets, local reference snapshots,
            ``analysis/``, and ``run_report.json``. An existing directory is
            accepted only when its recorded scientific signature matches this
            workflow.
        cluster : ClusterConfig
            Scheduler configuration, including backend, partition, log
            directory, and optional default scratch directory.
        execution : {"single_job", "dft_staged", "fully_staged"} or None, optional
            Scheduler grouping used by every child workflow:

            - ``"single_job"`` runs the complete target pipeline in one job.
            - ``"dft_staged"`` groups structure preparation, GFN-FF, the
              selected screening calculator, and any DFT ranking point,
              then separates later DFT refinement stages.
            - ``"fully_staged"`` submits one dependent job per stage.

            When omitted, ``low_cost`` and ``dft_ranked`` use
            ``"single_job"``; ``full`` uses ``"dft_staged"``.
        stage_resources : dict of str to Resources or None, optional
            Resource overrides keyed by the group names returned by
            ``wf.show_stages(execution=...)``. Use ``"single_job"`` for the
            default electronic-only submission. Full staged runs commonly use
            ``"init"``, ``"dft_opt"``, ``"dft_hessian"``, ``"dft_ts_opt"``,
            ``"dft_freq"``, and ``"dft_solv_sp"``. Missing groups use the
            child workflow defaults.
        array : bool, optional
            Submit separate arrays for each calculated branch and stage group.
            Collectors and the finalizer remain ordinary control jobs.
        array_parallelism : int or mapping of str to int or None, optional
            Required positive limit with arrays. A staged mapping must contain
            every group name across all branches. Limits apply per array, so
            two branches each limited to 4 can run 8 elements together.
        targets_per_task : int, optional
            Sequential targets per element in single-job mode; staged modes
            require 1. UMA targets share one server within a single-job batch.
        retry : bool, optional
            Retry failed targets in a compatible managed screen, after its
            earlier finalizer ends. By default use each report's retry_targets.
            Recollect complete branches before finalization, preserving earlier
            successes, reference reuse decisions, and attempt records.
            A successfully finalized screen requires a new output directory
            for further calculations, including after target cleanup.
        targets : mapping of str to iterable or None, optional
            Retry-only branch-specific child target objects or positions, for example
            {"transition_states": [0, 1]}. Omitted branches select no work when
            a mapping is supplied. An empty mapping is an initial no-op.
            Use child workflows for initial subsets. Reused references cannot
            be submitted.
        debug : bool, optional
            Forward debugging output to structure generation and calculator
            stages.
        save_output_dir : bool, optional
            Retain calculator output directories. This composed workflow
            requires ``True`` so reference evidence can be copied into the
            portable run and shared reference store.
        work_dir : str, pathlib.Path, or None, optional
            Scratch directory for calculator execution. When omitted, use
            ``cluster.work_dir`` if configured.
        collect_require_normal_termination : bool, optional
            If ``True``, branch collectors exclude target results with failed
            normal-termination columns. Excluded and failed targets remain
            listed in collection and finalization reports.
        collect_resources : Resources or None, optional
            Resources for each child branch's collection job. The underlying
            default is ``Resources(cpus=2, mem_gb=4, timeout_min=120)``.
        finalize_resources : Resources or None, optional
            Resources for the final reference-snapshot and analysis job. The
            default is ``Resources(cpus=2, mem_gb=4, timeout_min=120)``.
        target_retention : {"compact_success", "all"}, optional
            ``"compact_success"`` removes intermediate target parquets after
            successful collection while retaining the final parquet, timing,
            and required scientific evidence. Failed, skipped, or incomplete
            targets remain unmodified. ``"all"`` retains every intermediate
            parquet.
        artifact_policy : {"standard", "screening"}, optional
            ``"screening"`` keeps compact scientific results, uses a managed
            ``.submitit`` directory, and cleans successful artifacts only after
            final validation. ``"standard"`` preserves existing behavior.
        uma_oet_tools : str, pathlib.Path, or None, optional
            OET runtime selected inside child jobs containing UMA stages.
            For example, point to a pinned FairChem 2.23 CPU runtime.

        Returns
        -------
        ScreenSubmissionResult
            Run directory, child submission records, finalization job ID, and
            scheduler backend. Use ``finalization_job_id`` to identify the job
            after which ``ft.screen.open_run(out_dir)`` is ready for analysis.

        Raises
        ------
        ValueError
            If calculator evidence is disabled or an execution/retention
            option is invalid in a child workflow.
        FileExistsError
            If ``out_dir`` already contains a different scientific run.

        Examples
        --------
        Submit a g-xTB-only screen as one job per target:

        >>> from frust.cluster import ClusterConfig, Resources
        >>> cluster = ClusterConfig(
        ...     backend="slurm",
        ...     partition="kemi1",
        ...     log_dir="logs",
        ... )
        >>> submission = wf.submit(
        ...     out_dir="results_low_cost",
        ...     cluster=cluster,
        ...     execution="single_job",
        ...     stage_resources={
        ...         "single_job": Resources(
        ...             cpus=12,
        ...             mem_gb=12,
        ...             timeout_min=7200,
        ...         )
        ...     },
        ... )

        Inspect the exact resource keys before submitting a full DFT run:

        >>> wf.show_stages(execution="dft_staged")[
        ...     ["branch", "group", "stage", "engine", "solvent"]
        ... ]

        Submit arrays with up to two running elements in each branch/group:

        >>> submission = wf.submit(
        ...     out_dir="runs/screen", cluster=cluster,
        ...     array=True, array_parallelism=2,
        ... )

        After the finalization job ends, retry targets listed in the branch
        reports. Complete branches are recollected with earlier successes:

        >>> retried = wf.submit(
        ...     out_dir="runs/screen", cluster=cluster,
        ...     array=True, array_parallelism=2, retry=True,
        ... )

        Notes
        -----
        The finalizer uses an ``afterany`` dependency so a partial portable
        report is still produced when an upstream branch fails. Wait for the
        finalization job—not merely the target jobs—before opening the result
        bundle.

        Child workflows currently use their default
        ``orca_memory_fraction=0.8``. The complete ``Resources.mem_gb`` value
        is requested from the scheduler and 80 percent is forwarded to ORCA,
        leaving the remainder for job overhead. UMA screening also uses ORCA
        at ``level="low_cost"``.
        """
        artifact_policy = validate_artifact_policy(artifact_policy)
        if artifact_policy == "screening" and target_retention != "compact_success":
            raise ValueError(
                "artifact_policy='screening' requires target_retention='compact_success'"
            )
        if not save_output_dir:
            raise ValueError(
                "catalyst_screen requires save_output_dir=True so reference "
                "calculator evidence remains portable"
            )
        children, calculated, selected, limits = _screen_submission_plan(
            self, out_dir, execution, stage_resources, array, array_parallelism,
            targets_per_task, retry, targets,
        )
        if not any(selected.values()):
            if retry:
                raise ValueError("No failed screen targets selected for retry")
            return ScreenSubmissionResult(str(out_dir), {}, None, cluster.backend)
        if array:
            _validate_array_scheduler_options(cluster)
            for branch_targets in selected.values():
                _validate_array_size(cluster, (len(branch_targets) + targets_per_task - 1) // targets_per_task)
        with _screen_submission_guard(out_dir, cluster, managed=array or retry, retry=retry) as attempt:
            root = Path(out_dir)
            root.mkdir(parents=True, exist_ok=True)
            self._write_manifest(root, artifact_policy=artifact_policy)
            if attempt:
                _atomic_write_submission_json(attempt['_path'], attempt)
            submit_cluster = cluster
            finalize_cluster = cluster
            if artifact_policy == "screening":
                submitit_dir = initialize_submitit_directory(root)
                submit_cluster = replace(
                    cluster,
                    log_dir=submitit_dir / "jobs",
                    stderr_to_stdout=True,
                )
                finalize_cluster = replace(
                    cluster,
                    log_dir=submitit_dir / "control",
                    stderr_to_stdout=True,
                )
            reference_items = [
                item for item in self.targets() if item.branch == "references"
            ]
            if not retry:
                self._snapshot_reused_references(
                    root,
                    reference_items,
                    artifact_policy=artifact_policy,
                )
            submissions: dict[str, JobSubmissionResult] = {}
            dependency_ids: list[str | int] = []
            wait_paths: list[str] = []

            for branch, workflow in children.items():
                branch_dir = _branch_dir(root, branch)
                branch_targets = selected[branch]
                collect_output = branch_dir / ("computed.parquet" if branch == "references" else "merged.parquet")
                if not branch_targets:
                    previous_report = branch_dir / "collection_report.json"
                    if retry and previous_report.exists():
                        wait_paths.append(str(previous_report))
                    elif not retry and branch == "references" and not calculated[branch]:
                        branch_dir.mkdir(parents=True, exist_ok=True)
                        _write_verified_parquet(pd.DataFrame(), collect_output)
                    continue
                collect_report = branch_dir / (
                    f"collection_report_{attempt['attempt_id']}.json" if attempt else "collection_report.json"
                )
                submission = workflow.submit(
                    out_dir=branch_dir,
                    cluster=submit_cluster,
                    execution=execution,
                    stage_resources=stage_resources,
                    array=array, array_parallelism=limits[branch],
                    targets_per_task=targets_per_task, retry=retry,
                    _collect_targets=calculated[branch] if retry else None,
                    _screen_manifest_validated=attempt is None,
                    targets=branch_targets,
                    debug=debug,
                    save_output_dir=save_output_dir,
                    work_dir=work_dir,
                    collect=True,
                    collect_output=collect_output,
                    collect_report=collect_report,
                    collect_require_normal_termination=collect_require_normal_termination,
                    collect_resources=collect_resources,
                    target_retention=target_retention,
                    artifact_policy=artifact_policy,
                    uma_oet_tools=uma_oet_tools,
                    _defer_screening_cleanup=artifact_policy == "screening",
                )
                submissions[branch] = submission
                dependency_ids.append(
                    submission.collection_job_id or submission.job_ids[-1]
                )
                wait_paths.append(str(collect_report))

            if attempt and cluster.backend == "local":
                _wait_screen_collectors(submissions, submit_cluster.log_dir)
            executor = create_executor(finalize_cluster)
            update_executor_with_dependencies(
                executor,
                finalize_cluster,
                finalize_resources or DEFAULT_FINALIZE_RESOURCES,
                job_name="catalyst_screen_finalize",
                dependency_job_ids=dependency_ids,
                dependency_type="afterany",
            )
            finalizer_workflow = copy(self)
            finalizer_workflow._components_cache = None
            finalizer_workflow._systems_cache = None
            finalizer_workflow._children_cache = None
            final_job = executor.submit(
                _finalize_submitted_run,
                finalizer_workflow,
                root,
                wait_paths if cluster.backend == "local" and not attempt else None,
                wait_paths,
                artifact_policy,
                {
                    branch: {
                        "job_ids": list(submission.job_ids),
                        "collection_job_id": submission.collection_job_id,
                        "collection_output": submission.collection_output,
                        "collection_report": submission.collection_report,
                        "submission_path": submission.submission_path,
                        "array_job_ids": list(submission.array_job_ids),
                    }
                    for branch, submission in submissions.items()
                },
                None if attempt is None else attempt['attempt_id'],
            )
            if attempt:
                attempt['job_id'] = final_job.job_id
                attempt['log_dir'] = str(Path(finalize_cluster.log_dir).resolve())
                _atomic_write_submission_json(attempt['_path'], attempt)
            return ScreenSubmissionResult(
                run_dir=str(root),
                child_submissions=submissions,
                finalization_job_id=getattr(final_job, "job_id", None),
                backend=cluster.backend,
                submitit_dir=(
                    str(root / ".submitit") if artifact_policy == "screening" else None
                ),
            )


    def _reference_states(self) -> list[str]:
        dimer_states = (
            list(DIMER_STATES)
            if self.dimer_reference == "lowest"
            else [self.dimer_reference]
        )
        states = ["ligand", *dimer_states, "HBpin-mol", "HH"]
        if self.scope == "full_cycle":
            states.append("catalyst")
        return states

    def _analysis_levels(self) -> tuple[CalculationLevel, ...]:
        """Return calculation tiers available from the requested workflow."""
        if self.ranking is not None:
            return (
                ("low_cost", "uma_ranked", "full")
                if self.level == "full"
                else ("low_cost", "uma_ranked")
            )
        if self.level == "full":
            return (
                ("low_cost", "dft_ranked", "full")
                if self.include_dft_rank_sp
                else ("low_cost", "full")
            )
        if self.level == "dft_ranked":
            return ("low_cost", "dft_ranked")
        return ("low_cost",)

    def _reference_protocol(
        self,
        calculation_level: CalculationLevel | None = None,
    ) -> dict[str, Any]:
        """Return reference identity settings for one nested result tier."""
        level = self.level if calculation_level is None else calculation_level
        protocol = {
            "workflow": "frust.workflows.mols::v2",
            "calculation_level": level,
            "screening_fingerprint": self.screening.fingerprint(),
            "ranking_solvation": self.ranking_solvation,
            "n_confs": self.n_confs,
            "top_n": self.top_n,
            "prune_initial": self.prune_initial,
        }
        if self.ranking is not None and level in {"uma_ranked", "full"}:
            protocol["ranking_fingerprint"] = self.ranking.fingerprint()
            protocol["uma_rank_top_n"] = self.uma_rank_top_n
        if level == "full" and not self.include_dft_rank_sp:
            protocol["include_dft_rank_sp"] = False
        return protocol

    def _shared_library(self, *, initialize: bool = False) -> ReferenceLibrary | None:
        if self.reference_store is None:
            return None
        library = ReferenceLibrary(self.reference_store)
        if initialize:
            library.initialize()
        elif not library.root.exists():
            return None
        return library

    def _cached_reference(self, target: StructureTarget) -> ReferenceRecord | None:
        library = self._shared_library(initialize=False)
        if library is None:
            return None
        record = library.find(
            target,
            self.method,
            protocol=self._reference_protocol(self.level),
            reuse_policy=(self.reuse_policy if self.level == "full" else "auto_valid"),
            calculation_level=self.level,
        )
        if record is None:
            return None
        for tier in self._analysis_levels():
            if tier == self.level:
                continue
            nested = library.find(
                target,
                self.method,
                protocol=self._reference_protocol(tier),
                reuse_policy="auto_valid",
                calculation_level=tier,
            )
            if nested is None:
                return None
        return record

    def _snapshot_reused_references(
        self,
        root: Path,
        items: list[CatalystScreenTarget],
        *,
        artifact_policy: ArtifactPolicy = "standard",
    ) -> None:
        branch_dir = _branch_dir(root, "references")
        local_library = ReferenceLibrary(branch_dir).initialize()
        frames: list[pd.DataFrame] = []
        sources: list[Path] = []
        tier_frames: dict[str, list[pd.DataFrame]] = {
            tier: [] for tier in self._analysis_levels() if tier != self.level
        }
        tier_sources: dict[str, list[Path]] = {tier: [] for tier in tier_frames}
        shared_library = self._shared_library(initialize=False)
        for item in items:
            if item.action != "reuse":
                continue
            record = self._cached_reference(item.target)
            if record is None:
                continue
            local = local_library.import_record(record)
            frame = local.materialize(item.target)
            if artifact_policy == "screening":
                frame = compact_result_dataframe(frame)
            frame["reference_id"] = local.reference_id
            frame["reference_source"] = "shared_library"
            frames.append(frame)
            sources.append(local.path / "result.parquet")
            if shared_library is None:
                continue
            for tier in tier_frames:
                tier_record = shared_library.find(
                    item.target,
                    self.method,
                    protocol=self._reference_protocol(tier),
                    reuse_policy="auto_valid",
                    calculation_level=tier,
                )
                if tier_record is None:
                    continue
                local_tier = local_library.import_record(tier_record)
                tier_frame = local_tier.materialize(item.target)
                if artifact_policy == "screening":
                    tier_frame = compact_result_dataframe(tier_frame)
                tier_frame["reference_id"] = local_tier.reference_id
                tier_frame["reference_source"] = "shared_library"
                tier_frames[tier].append(tier_frame)
                tier_sources[tier].append(local_tier.path / "result.parquet")
        reused = _concat_reference_results(frames, source_files=sources)
        _write_verified_parquet(reused, branch_dir / "reused.parquet")
        for tier, nested_frames in tier_frames.items():
            tier_dir = branch_dir / "tiers" / tier
            tier_dir.mkdir(parents=True, exist_ok=True)
            nested = _concat_reference_results(
                nested_frames,
                source_files=tier_sources[tier],
            )
            _write_verified_parquet(nested, tier_dir / "reused.parquet")

    def _write_manifest(
        self,
        root: Path,
        *,
        artifact_policy: ArtifactPolicy = "standard",
    ) -> None:
        ts_targets = self.children()["transition_states"].targets()
        terminal_results = _calculation_result_paths(self.scope)
        level_results = {
            level: (
                terminal_results
                if level == self.level
                else _tier_calculation_result_paths(self.scope, level)
            )
            for level in self._analysis_levels()
        }
        reference_plan = [
            {
                "target_id": item.target.target_id,
                "state_id": item.target.state_id,
                "scope": item.target.scope,
                "system_name": item.target.system.system_name,
                "substrate_name": item.target.system.substrate_name,
                "catalyst_name": item.target.system.catalyst_name,
                "rpos": item.target.rpos,
                "action": item.action,
                "reference_id": item.reference_id,
            }
            for item in self.targets()
            if item.branch == "references"
        ]
        manifest = {
            "schema_version": 3,
            "run_type": "catalyst_screen",
            "created_at": _utc_now(),
            "scope": self.scope,
            "dimer_reference": self.dimer_reference,
            "dimer_candidates": (
                list(DIMER_STATES)
                if self.dimer_reference == "lowest"
                else [self.dimer_reference]
            ),
            "calculation_level": self.level,
            "artifact_policy": artifact_policy,
            "analysis_levels": list(self._analysis_levels()),
            "screening": self.screening.to_dict(),
            "screening_fingerprint": self.screening.fingerprint(),
            "method": self.method.to_dict(),
            "method_fingerprint": self.method.fingerprint(),
            "ranking_solvation": self.ranking_solvation,
            "include_dft_rank_sp": self.include_dft_rank_sp,
            "ts_spec_profile": self.spec_profile,
            "resolved_ts_spec_profile": self.children()[
                "transition_states"
            ].resolved_spec_profile,
            "ts_spec_match": self.spec_match,
            "ts_types": list(self.ts_types),
            "g_corrections_kcal_mol": self.g_corrections_kcal_mol,
            "mechanism_id": (
                "frust_ts_barriers::v2"
                if self.scope == "barriers"
                else "frust_balanced_cycle::v3"
            ),
            "components": _records(self.components()),
            "systems": _records(self.systems()),
            "reference_store_configured": self.reference_store is not None,
            "reuse_policy": self.reuse_policy,
            "reference_protocol": self._reference_protocol(),
            "reference_plan": reference_plan,
            "analysis_targets": [
                {
                    "state_id": target.state_id,
                    "system_name": target.system.system_name,
                    "substrate_name": target.system.substrate_name,
                    "catalyst_name": target.system.catalyst_name,
                    "rpos": int(target.rpos),
                }
                for target in ts_targets
            ],
            "calculation_results": terminal_results,
            "tier_calculation_results": level_results,
        }
        if self.ranking is not None:
            manifest["ranking"] = self.ranking.to_dict()
            manifest["ranking_fingerprint"] = self.ranking.fingerprint()
            manifest["uma_rank_top_n"] = self.uma_rank_top_n
        if self.method.result_family == "uma":
            manifest["ts_refine_n"] = self.ts_refine_n
        signature_keys = [
            "scope",
            "dimer_reference",
            "calculation_level",
            "screening_fingerprint",
            "method_fingerprint",
            "ranking_solvation",
            "include_dft_rank_sp",
            "ts_spec_profile",
            "ts_spec_match",
            "ts_types",
            "g_corrections_kcal_mol",
            "components",
            "systems",
            "reuse_policy",
            "reference_protocol",
            "analysis_targets",
            "artifact_policy",
        ]
        if self.method.result_family == "uma":
            signature_keys.append("ts_refine_n")
        if self.ranking is not None:
            signature_keys.extend(["ranking_fingerprint", "uma_rank_top_n"])
        manifest["run_signature"] = _json_hash(
            {key: manifest[key] for key in signature_keys}
        )
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("run_signature") != manifest["run_signature"]:
                raise FileExistsError(
                    f"Run directory {root} contains a different catalyst-screen "
                    "manifest; choose a new out_dir"
                )
            return
        unexpected = [path for path in root.iterdir() if path.name not in {"manifest.json", ".frust"}]
        if unexpected:
            raise FileExistsError(
                f"Run directory {root} is not empty and has no compatible manifest; "
                "choose a new out_dir"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


class Wb97ComparisonWorkflow(CatalystScreenWorkflow):
    """Opt-in ωB97 characterization seeded by a completed UMA run.

    Parameters
    ----------
    parent : ScreenRun
        Portable full-UMA run that supplied the TS candidates.
    candidate_rows : pandas.DataFrame
        Exact rows selected from ``parent.candidate_barriers()``.
    reference_store : str, pathlib.Path, or None, optional
        Shared reference library for exact-match ωB97 minima.
    reuse_policy : {"approved", "auto_valid"}, optional
        Rule for reusing full thermochemical references.
    """

    def __init__(
        self, parent: ScreenRun, candidate_rows: pd.DataFrame, *,
        reference_store: str | Path | None = None,
        reuse_policy: ReusePolicy = "approved",
    ) -> None:
        source = parent.manifest
        self.parent = parent
        self.candidate_rows = candidate_rows.reset_index(drop=True).copy()
        self.seed_dir: Path | None = None
        super().__init__(
            dataframe=pd.DataFrame(source["components"]),
            ts_types=tuple(dict.fromkeys(self.candidate_rows["ts_type"])),
            screening="gxtb-default",
            level="full",
            method="wb97xd3-631g",
            spec_profile=source["resolved_ts_spec_profile"],
            include_dft_rank_sp=False,
            dimer_reference=source["dimer_reference"],
            g_corrections_kcal_mol=source["g_corrections_kcal_mol"],
            reference_store=reference_store,
            reuse_policy=reuse_policy,
        )

    def children(self) -> dict[str, Any]:
        """Use parent geometries for TS targets and ordinary ωB97 references."""
        children = super().children()
        if not isinstance(children["transition_states"], SeededWb97TSWorkflow):
            keys = tuple(
                (str(row.ts_result_id), str(row.system_name), int(row.rpos),
                 str(row.ts_type))
                for row in self.candidate_rows.itertuples()
            )
            children["transition_states"] = SeededWb97TSWorkflow(
                dataframe=self.components(),
                ts_types=self.ts_types,
                method=self.method,
                calculation_level="full",
                include_dft_rank_sp=False,
                candidate_ids=keys,
                seed_dir=self.seed_dir,
            )
            self._children_cache = children
        return dict(children)

    def _analysis_levels(self) -> tuple[CalculationLevel, ...]:
        """Keep this follow-on bundle focused on final ωB97 results."""
        return ("full",)

    def _write_manifest(
        self, root: Path, *, artifact_policy: ArtifactPolicy = "standard"
    ) -> None:
        """Snapshot UMA seeds and provenance before jobs leave this process."""
        path = root / "manifest.json"
        requested_ids = self.candidate_rows["ts_result_id"].astype(str).tolist()
        if path.exists():
            existing = json.loads(path.read_text())
            previous = existing.get("comparison", {})
            if not (
                previous.get("parent_run_signature") == self.parent.manifest["run_signature"]
                and previous.get("candidate_result_ids") == requested_ids
                and existing.get("method_fingerprint") == self.method.fingerprint()
                and existing.get("screening_fingerprint") == self.screening.fingerprint()
                and existing.get("reuse_policy") == self.reuse_policy
                and existing.get("artifact_policy") == artifact_policy
                and existing.get("dimer_reference") == self.dimer_reference
            ):
                raise FileExistsError(
                    f"Run directory {root} contains a different catalyst-screen "
                    "manifest; choose a new out_dir"
                )
            self.seed_dir = root.resolve() / "comparison" / "seeds"
            self.children()["transition_states"].seed_dir = self.seed_dir
            return
        seeds: list[tuple[str, pd.DataFrame]] = []
        for row in self.candidate_rows.itertuples():
            result_id = str(row.ts_result_id)
            seed = self.parent._raw_result(result_id).copy()
            if "uma_ts_opt-oc" not in seed or seed["uma_ts_opt-oc"].isna().any():
                raise ValueError(
                    f"UMA candidate {result_id!r} has no optimized TS geometry"
                )
            geometry = seed["uma_ts_opt-oc"].iloc[0]
            keep = [column for column in seed if "-" not in column]
            seed = seed[keep].copy()
            seed["coords_embedded"] = pd.Series([geometry])
            seed["parent_uma_result_id"] = result_id
            seeds.append((result_id, seed))
        super()._write_manifest(root, artifact_policy=artifact_policy)
        self.seed_dir = root.resolve() / "comparison" / "seeds"
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.children()["transition_states"].seed_dir = self.seed_dir
        for result_id, seed in seeds:
            _write_verified_parquet(seed, self.seed_dir / f"{result_id}.parquet")
        self.candidate_rows.to_parquet(
            root / "comparison" / "uma_candidates.parquet", index=False
        )
        parent_states = self.parent.states()
        parent_states = parent_states[
            parent_states["result_id"].isin(self.candidate_rows["ts_result_id"])
        ]
        parent_states.to_parquet(root / "comparison" / "uma_states.parquet", index=False)
        manifest = json.loads(path.read_text())
        manifest["analysis_targets"] = [
            {**target, "parent_uma_result_id": str(item.builder_options["parent_uma_result_id"])}
            for target, item in zip(
                manifest["analysis_targets"],
                self.children()["transition_states"].targets(),
            )
        ]
        manifest["comparison"] = {
            "parent_run_signature": self.parent.manifest["run_signature"],
            "parent_method": self.parent.manifest["method"],
            "parent_method_fingerprint": self.parent.manifest["method_fingerprint"],
            "parent_dimer_reference": self.parent.manifest["dimer_reference"],
            "candidate_result_ids": requested_ids,
            "uma_candidates": "comparison/uma_candidates.parquet",
            "uma_states": "comparison/uma_states.parquet",
        }
        manifest["run_signature"] = _json_hash({
            "base": manifest["run_signature"],
            "comparison": manifest["comparison"],
        })
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def wb97_comparison(
    uma_run: ScreenRun | str | Path, *,
    candidates: Literal["selected"] | list[str] | tuple[str, ...] = "selected",
    reference_store: str | Path | None = None,
    reuse_policy: ReusePolicy = "approved",
) -> Wb97ComparisonWorkflow:
    """Prepare an optional ωB97 comparison for completed UMA TS candidates.

    Parameters
    ----------
    uma_run : ScreenRun, str, or pathlib.Path
        Completed portable full-UMA run or its directory.
    candidates : {"selected"} or sequence of str, optional
        ``"selected"`` advances each target's selected UMA candidate.
        A sequence names exact UMA ``ts_result_id`` values and may contain
        several candidates for the same TS target.
    reference_store : str, pathlib.Path, or None, optional
        Exact-match ωB97 reference library used by the follow-on run.
    reuse_policy : {"approved", "auto_valid"}, optional
        ``"approved"`` reuses only manually approved full references;
        ``"auto_valid"`` also reuses automatically validated minima.

    Returns
    -------
    Wb97ComparisonWorkflow
        Call ``run`` or ``submit`` with a new output directory, then open the
        result and call ``method_comparison()`` for side-by-side barriers.

    Examples
    --------
    >>> import frust as ft
    >>> parent = ft.screen.open_run("runs/uma")
    >>> wf = ft.workflows.wb97_comparison(parent, candidates="selected")
    >>> comparison = wf.run(out_dir="runs/uma_wb97")
    >>> comparison.method_comparison()
    """
    parent = uma_run if isinstance(uma_run, ScreenRun) else ScreenRun(uma_run)
    manifest = parent.manifest
    if manifest.get("calculation_level") != "full" or manifest.get("method", {}).get("result_family") != "uma":
        raise ValueError("ωB97 comparison requires a completed full UMA run")
    available = parent.candidate_barriers()
    if candidates == "selected":
        chosen = available[available["selected"]].copy()
    else:
        requested = tuple(str(value) for value in candidates)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("candidates must contain distinct UMA result IDs")
        missing = set(requested) - set(available["ts_result_id"])
        if missing:
            raise ValueError(f"Unknown UMA candidate result IDs: {sorted(missing)}")
        chosen = available.set_index("ts_result_id").loc[list(requested)].reset_index()
    if chosen.empty:
        raise ValueError("No selected UMA candidates are available")
    return Wb97ComparisonWorkflow(
        parent, chosen, reference_store=reference_store, reuse_policy=reuse_policy
    )


def catalyst_screen(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    screening: ScreeningPlan | str = "gxtb-default",
    level: CalculationLevel = "full",
    method: MethodPlan | str | None = None,
    ranking: RankingPlan | str | None = None,
    ranking_solvation: str = "method",
    spec_profile: str = "auto",
    spec_match: str = "prefer-exact",
    include_dft_rank_sp: bool | None = None,
    scope: ScreenScope = "barriers",
    dimer_reference: DimerReference = "lowest",
    g_corrections_kcal_mol: dict[str, float] | None = None,
    reference_store: str | Path | None = None,
    reuse_policy: ReusePolicy = "approved",
    n_confs: int | None = None,
    top_n: int = 20,
    uma_rank_top_n: int = 1,
    ts_refine_n: int = 3,
    prune_initial: bool | dict[str, Any] = True,
) -> CatalystScreenWorkflow:
    """Create an end-to-end catalyst-screen workflow.

    The default ``scope="barriers"`` calculates TS1--TS4 plus ligand, all
    three dimer topologies, HBpin, and H2 references. The lowest qualified
    dimer is selected per catalyst. ``scope="full_cycle"`` additionally
    calculates catalyst, int1, int2, HBpin-ligand, and INT3 states for a
    balanced profile.

    Parameters
    ----------
    csv_path : str or pathlib.Path or None, optional
        Component CSV containing substrate and catalyst rows. Provide exactly
        one of ``csv_path`` and ``dataframe``.
    dataframe : pandas.DataFrame or None, optional
        In-memory component table with ``role`` and ``smiles`` columns.
    ts_types : sequence of str, optional
        Transition-state families to calculate. Supported built-ins are
        ``"TS1"``, ``"TS2"``, ``"TS3"``, and ``"TS4"``.
    screening : ScreeningPlan or str, optional
        Inexpensive geometry-screening plan. The default ``"gxtb-default"``
        runs GFN-FF followed by direct g-xTB ranking and optimization. Choose
        ``"uma-gas"`` or ``"uma-alpb-chloroform"`` for UMA screening.
    level : {"low_cost", "uma_ranked", "dft_ranked", "full"}, optional
        ``"low_cost"`` reports electronic barriers from the selected g-xTB or
        UMA screen, ``"uma_ranked"`` reports UMA single-point electronic
        barriers on g-xTB optimized geometries, ``"dft_ranked"`` reports DFT
        single-point electronic barriers on screened geometries, and
        ``"full"`` performs refinement and frequencies with the selected
        final method, DFT or UMA, so both electronic and Gibbs barriers are
        available when the frequency stage returns thermochemistry. Deeper runs
        retain independently selected lower-tier winners. A ``"full"`` run
        with UMA reranking retains low-cost, UMA-ranked, and full barriers.
        A ``"full"`` run with
        ``include_dft_rank_sp=True`` retains low-cost, DFT-ranked, and full
        barriers; otherwise it retains low-cost and full barriers.
    method : MethodPlan, str, or None, optional
        Final calculation plan. ``"uma-gas"`` and
        ``"uma-alpb-chloroform"`` characterize TS and reference minima with
        UMA; DFT presets retain independent DFT validation.
    ranking : RankingPlan, str, or None, optional
        Optional ``"uma-gas"`` or ``"uma-alpb-chloroform"`` single-point
        reranking after g-xTB optimization. The ``"full"`` result still uses
        the selected DFT method; no UMA optimization is run.
    ranking_solvation : str, optional
        Solvation for DFT single points on screened geometries. ``"method"``
        inherits the method's analysis solvent, ``"gas"`` disables implicit
        solvent, and another value selects that SMD solvent. Full UMA uses
        ``"method"`` because it has no DFT ranking stage. UMA reranking also
        requires ``"method"``; its gas or ALPB environment comes from
        ``ranking``.
    spec_profile : str, optional
        TS/INT3 guess and constraint profile. ``"auto"`` follows the DFT
        validation method, normally ωB97 gas, or the reviewed UMA gas profile
        for full UMA. To select that profile explicitly, pass
        ``"omol-uma-s-1p2p1/gas"``. This choice does not change the UMA
        calculation environment.
    spec_match : {"prefer-exact", "exact"}, optional
        Profile resolution policy for TS/INT3 guesses.
    include_dft_rank_sp : bool or None, optional
        Include the ωB97 ranking single point in a ``"full"`` run. The
        default is ``False`` for UMA screening or UMA reranking and ``True``
        for plain g-xTB screening;
        ``"dft_ranked"`` always runs the ranking point.
    scope : {"barriers", "full_cycle"}, optional
        ``"barriers"`` calculates the dependencies of the four supplied
        barrier equations. ``"full_cycle"`` adds every state needed for the
        balanced catalytic-cycle profile. Full UMA currently supports
        ``"barriers"`` only.
    dimer_reference : {"lowest", "dimer", "dimer_bh_bridged", "dimer_eight_membered"}, optional
        Catalyst dimer used in barrier and profile equations. ``"lowest"``
        calculates all three topologies and strictly selects the lowest
        qualified result per catalyst. An explicit state calculates and uses
        only that topology. ``"dimer"`` reproduces the former FRUST behavior.
    g_corrections_kcal_mol : dict or None, optional
        Gibbs-only profile corrections in kcal/mol. Defaults to ``-1.89`` for
        TS1 and TS3. These corrections are never applied to electronic
        barriers.
    reference_store : str, pathlib.Path, or None, optional
        Shared inspectable reference library. When omitted, use
        ``FRUST_REFERENCE_STORE`` if set. The completed run always receives a
        local snapshot of references it reused.
    reuse_policy : {"approved", "auto_valid"}, optional
        Full thermochemical references use ``"approved"`` for manual-review
        reuse or ``"auto_valid"`` to accept automatic minimum checks.
        Exact-match ``"low_cost"``, ``"uma_ranked"``, and ``"dft_ranked"``
        screening artifacts are automatically reusable because they are
        stored separately and are not presented as approved DFT minima.
    n_confs : int or None, optional
        Initial conformer count forwarded consistently to every child
        workflow.
    top_n : int, optional
        Number retained by the low-cost screen. With UMA reranking, this is
        the broad g-xTB cutoff before the UMA single points.
    uma_rank_top_n : int, optional
        Number of UMA-ranked candidates advanced to final ωB97 refinement.
        Defaults to one and is independent of ``top_n``.
    ts_refine_n : int, optional
        Maximum distinct UMA optimized TS candidates sent to Hessian,
        released ``OptTS``, and final numerical frequencies in a full UMA run.
    prune_initial : bool or dict, optional
        Initial conformer-pruning configuration forwarded to child workflows.

    Returns
    -------
    CatalystScreenWorkflow
        Calculation-free composed workflow ready for ``plan()``, ``run()``,
        or ``submit()``.

    Examples
    --------
    >>> import frust as ft
    >>> wf = ft.workflows.catalyst_screen(
    ...     csv_path="screen.csv",
    ...     method="r2scan-3c",
    ...     scope="barriers",
    ... )
    >>> wf.plan()[["branch", "state_id", "action"]]
    """
    return CatalystScreenWorkflow(
        csv_path=csv_path,
        dataframe=dataframe,
        ts_types=ts_types,
        screening=screening,
        level=level,
        method=method,
        ranking=ranking,
        ranking_solvation=ranking_solvation,
        spec_profile=spec_profile,
        spec_match=spec_match,
        include_dft_rank_sp=include_dft_rank_sp,
        scope=scope,
        dimer_reference=dimer_reference,
        g_corrections_kcal_mol=g_corrections_kcal_mol,
        reference_store=reference_store,
        reuse_policy=reuse_policy,
        n_confs=n_confs,
        top_n=top_n,
        uma_rank_top_n=uma_rank_top_n,
        ts_refine_n=ts_refine_n,
        prune_initial=prune_initial,
    )


def _finalize_submitted_run(
    workflow: CatalystScreenWorkflow,
    root: Path,
    wait_paths: list[str] | None,
    expected_report_paths: list[str] | None = None,
    artifact_policy: ArtifactPolicy = "standard",
    submission_status: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
) -> FinalizationJobResult:
    if wait_paths:
        deadline = time.monotonic() + 3600
        while (
            not all(Path(path).exists() for path in wait_paths)
            and time.monotonic() < deadline
        ):
            time.sleep(1)
    try:
        with _mutation_lock(root, wait=True) if attempt_id is not None else nullcontext():
            if attempt_id is not None:
                for raw_path in expected_report_paths or []:
                    path = Path(raw_path)
                    if path.exists():
                        _atomic_write_submission_json(path.parent/'collection_report.json', json.loads(path.read_text()))
            report = _finalize_run(
                workflow,
                root,
                expected_report_paths=expected_report_paths,
                artifact_policy=artifact_policy,
                submission_status=submission_status,
            )
            return FinalizationJobResult(
                run_dir=str(root),
                report_path=str(root / "run_report.json"),
                status=str(report["overall_status"]),
            )
    finally:
        _mark_completed(root, attempt_id, 'screen_finalize', 0)


def _finalize_run(
    workflow: CatalystScreenWorkflow,
    root: Path,
    *,
    expected_report_paths: list[str] | None = None,
    artifact_policy: ArtifactPolicy = "standard",
    submission_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_policy = validate_artifact_policy(artifact_policy)
    branch_reports, incomplete_reports = _validate_collection_reports(
        root,
        expected_report_paths or [],
    )
    if incomplete_reports:
        status = (
            "failed"
            if any(item["reason"] == "missing" for item in incomplete_reports)
            else "partial"
        )
        report = {
            "schema_version": 2,
            "finalized_at": _utc_now(),
            "overall_status": status,
            "artifact_policy": artifact_policy,
            "branches": {
                branch: _collection_report_summary(payload)
                for branch, payload in branch_reports.items()
            },
            "incomplete_collections": incomplete_reports,
            "submission": dict(submission_status or {}),
        }
        (root / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        raise RuntimeError(
            "Catalyst-screen finalization requires complete collection reports; "
            f"see {root / 'run_report.json'}"
        )

    reference_dir = _branch_dir(root, "references")
    local_library = ReferenceLibrary(reference_dir).initialize()
    shared_library = workflow._shared_library(initialize=True)
    tier_report = _finalize_nested_tiers(
        workflow,
        root,
        local_library=local_library,
        shared_library=shared_library,
        artifact_policy=artifact_policy,
    )
    computed_path = reference_dir / "computed.parquet"
    computed = (
        pd.read_parquet(computed_path) if computed_path.exists() else pd.DataFrame()
    )
    reused_path = reference_dir / "reused.parquet"
    reused = pd.read_parquet(reused_path) if reused_path.exists() else pd.DataFrame()
    reused_ids = set(
        reused.get("reference_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    manifest = json.loads((root / "manifest.json").read_text())
    reference_plan = {
        str(entry["target_id"]): entry for entry in manifest.get("reference_plan", [])
    }
    computed_frames: list[pd.DataFrame] = []
    computed_sources: list[Path] = []
    publication_entries: list[dict[str, Any]] = []
    reference_targets = workflow.children()["references"].targets()
    for target in reference_targets:
        target_dir = reference_dir / target.tag
        result_path = _deepest_parquet(target_dir)
        if result_path is None:
            planned = reference_plan.get(target.target_id, {})
            planned_reference_id = planned.get("reference_id")
            reused_as_planned = (
                planned.get("action") == "reuse"
                and str(planned_reference_id) in reused_ids
            )
            publication_entries.append(
                {
                    "target_id": target.target_id,
                    "state_id": target.state_id,
                    "target": target.tag,
                    "result_path": None,
                    "status": "reused" if reused_as_planned else "missing_result",
                    "validation_status": None,
                    "reference_id": (
                        str(planned_reference_id) if reused_as_planned else None
                    ),
                    "issue": (
                        ""
                        if reused_as_planned
                        else "No completed or snapshotted reused result was found"
                    ),
                }
            )
            continue
        frame = pd.read_parquet(result_path).copy()
        reference_id: str | None = None
        validation_status: str | None = None
        publication_status = "published"
        publication_issue = ""
        try:
            local_record = local_library.publish(
                frame,
                target,
                workflow.method,
                protocol=workflow._reference_protocol(),
                calculation_level=workflow.level,
                source_run=root,
                source_target_dir=target_dir,
                artifact_policy=artifact_policy,
            )
        except ValueError as exc:
            publication_status = "not_published"
            validation_status = "invalid"
            publication_issue = str(exc)
        else:
            reference_id = local_record.reference_id
            metadata = local_record.metadata
            validation_status = str(
                metadata.get("validation_status")
                or metadata.get("auto_validation", {}).get("status")
                or "auto_valid"
            )
            if shared_library is not None:
                shared_library.publish(
                    frame,
                    target,
                    workflow.method,
                    protocol=workflow._reference_protocol(),
                    calculation_level=workflow.level,
                    source_run=root,
                    source_target_dir=target_dir,
                    artifact_policy=artifact_policy,
                )
        frame["reference_id"] = reference_id
        frame["reference_source"] = "calculated"
        computed_frames.append(frame)
        computed_sources.append(result_path)
        publication_entries.append(
            {
                "target_id": target.target_id,
                "state_id": target.state_id,
                "target": target.tag,
                "result_path": str(result_path.relative_to(root)),
                "status": publication_status,
                "validation_status": validation_status,
                "reference_id": reference_id,
                "issue": publication_issue,
            }
        )
    if computed_frames:
        computed = _concat_reference_results(
            computed_frames,
            source_files=computed_sources,
        )
        _write_verified_parquet(computed, computed_path)
    publication_report = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "n_targets": len(reference_targets),
        "n_results": int(len(computed)),
        "n_published": sum(
            entry["status"] == "published" for entry in publication_entries
        ),
        "n_not_published": sum(
            entry["status"] == "not_published" for entry in publication_entries
        ),
        "n_reused": sum(entry["status"] == "reused" for entry in publication_entries),
        "n_missing_results": sum(
            entry["status"] == "missing_result" for entry in publication_entries
        ),
        "entries": publication_entries,
    }
    publication_report_path = reference_dir / "publication_report.json"
    publication_report_path.write_text(
        json.dumps(publication_report, indent=2, sort_keys=True, default=str) + "\n"
    )
    available = [frame for frame in (computed, reused) if not frame.empty]
    available_paths = [
        path
        for frame, path in ((computed, computed_path), (reused, reused_path))
        if not frame.empty
    ]
    combined = _concat_reference_results(
        available,
        source_files=available_paths,
    )
    _write_verified_parquet(combined, reference_dir / "merged.parquet")

    analysis_report = build_analysis(root)
    discovered_branch_reports: dict[str, Any] = {}
    for branch in workflow.children():
        report_path = _branch_dir(root, branch) / "collection_report.json"
        if report_path.exists():
            discovered_branch_reports[branch] = json.loads(report_path.read_text())
    branch_reports.update(discovered_branch_reports)
    reference_complete = (
        int(publication_report["n_not_published"]) == 0
        and int(publication_report["n_missing_results"]) == 0
        and (
            int(publication_report["n_published"]) + int(publication_report["n_reused"])
            == int(publication_report["n_targets"])
        )
    )
    tiers_complete = all(
        bool(branch_report.get("complete", False))
        for level_report in tier_report.get("levels", {}).values()
        for branch_report in level_report.values()
    )
    overall_success = reference_complete and tiers_complete
    report = {
        "schema_version": 2,
        "finalized_at": _utc_now(),
        "overall_status": "success" if overall_success else "partial",
        "artifact_policy": artifact_policy,
        "analysis": analysis_report,
        "branches": {
            branch: _collection_report_summary(payload)
            for branch, payload in branch_reports.items()
        },
        "analysis_tiers": tier_report,
        "n_references_calculated": int(len(computed)),
        "n_references_published": int(publication_report["n_published"]),
        "n_references_not_published": int(publication_report["n_not_published"]),
        "n_references_reused": int(len(reused)),
        "reference_publication_report": str(publication_report_path.relative_to(root)),
        "submission": dict(submission_status or {}),
    }
    (root / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    if artifact_policy == "screening" and report["overall_status"] == "success":
        report["artifact_cleanup"] = _cleanup_screening_targets(workflow, root)
        if report["artifact_cleanup"]["errors"]:
            report["overall_status"] = "partial"
        elif (root / ".submitit" / "ownership.json").exists():
            try:
                report["submitit_cleanup"] = cleanup_submitit_jobs(root)
            except Exception as exc:
                report["overall_status"] = "partial"
                report["submitit_cleanup"] = {"error": str(exc)}
        (root / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
    return report


def _validate_collection_reports(
    root: Path,
    expected_report_paths: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate required branch reports before scientific finalization."""
    reports: dict[str, Any] = {}
    incomplete: list[dict[str, Any]] = []
    branch_paths = {
        str(_branch_dir(root, branch).resolve()): branch
        for branch in ("transition_states", "references", "cycle_molecules", "int3")
    }
    for raw_path in expected_report_paths:
        path = Path(raw_path)
        branch = branch_paths.get(str(path.parent.resolve()), path.parent.name)
        if not path.exists():
            incomplete.append(
                {"branch": branch, "report": str(path), "reason": "missing"}
            )
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            incomplete.append(
                {
                    "branch": branch,
                    "report": str(path),
                    "reason": "invalid_json",
                    "error": str(exc),
                }
            )
            continue
        reports[branch] = payload
        expected = int(payload.get("n_targets", 0))
        collected = int(payload.get("n_collected", 0))
        failures = int(payload.get("n_failures", 0))
        output = Path(str(payload.get("output", "")))
        if failures or collected != expected:
            incomplete.append(
                {
                    "branch": branch,
                    "report": str(path),
                    "reason": "incomplete",
                    "n_targets": expected,
                    "n_collected": collected,
                    "n_failures": failures,
                }
            )
            continue
        try:
            merged = pd.read_parquet(output)
        except Exception as exc:
            incomplete.append(
                {
                    "branch": branch,
                    "report": str(path),
                    "reason": "invalid_output",
                    "output": str(output),
                    "error": str(exc),
                }
            )
            continue
        if expected and merged.empty:
            incomplete.append(
                {
                    "branch": branch,
                    "report": str(path),
                    "reason": "empty_output",
                    "output": str(output),
                }
            )
    return reports, incomplete


def _collection_report_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact collection status for the top-level run report."""
    keys = (
        "workflow",
        "output",
        "n_targets",
        "n_collected",
        "n_rows",
        "n_failures",
        "n_skipped",
        "n_missing",
        "n_errored",
        "target_retention",
    )
    summary = {key: payload.get(key) for key in keys if key in payload}
    timing = payload.get("timing")
    if isinstance(timing, Mapping):
        summary["timing"] = {
            key: timing.get(key)
            for key in (
                "n_timing_files",
                "n_timing_groups",
                "total_elapsed_s",
                "total_core_hours",
            )
            if key in timing
        }
    return summary


def _cleanup_screening_targets(
    workflow: CatalystScreenWorkflow,
    root: Path,
) -> dict[str, Any]:
    """Delete only normally terminated per-target directories after finalization."""
    result: dict[str, Any] = {
        "n_removed_targets": 0,
        "removed_bytes": 0,
        "removed_targets": [],
        "retained_targets": [],
        "errors": [],
    }
    for branch, child in workflow.children().items():
        branch_dir = _branch_dir(root, branch)
        for target in child.targets():
            target_dir = branch_dir / target.tag
            try:
                target_dir.resolve().relative_to(branch_dir.resolve())
                if target_dir.is_symlink():
                    raise ValueError("target directory is a symlink")
            except Exception as exc:
                result["errors"].append({"target": str(target_dir), "error": str(exc)})
                continue
            final_path = _deepest_parquet(target_dir)
            if final_path is None:
                if target_dir.exists():
                    result["retained_targets"].append(str(target_dir))
                continue
            try:
                frame = pd.read_parquet(final_path)
                nt_columns = normal_termination_columns(frame)
                normal = not nt_columns or bool(
                    frame[nt_columns].fillna(False).all().all()
                )
                if not normal:
                    result["retained_targets"].append(str(target_dir))
                    continue
                size = sum(
                    path.stat().st_size
                    for path in target_dir.rglob("*")
                    if path.is_file()
                )
                shutil.rmtree(target_dir)
            except Exception as exc:
                result["errors"].append({"target": str(target_dir), "error": str(exc)})
                continue
            result["n_removed_targets"] += 1
            result["removed_bytes"] += int(size)
            result["removed_targets"].append(str(target_dir))
    return result


def _finalize_nested_tiers(
    workflow: CatalystScreenWorkflow,
    root: Path,
    *,
    local_library: ReferenceLibrary,
    shared_library: ReferenceLibrary | None,
    artifact_policy: ArtifactPolicy = "standard",
) -> dict[str, Any]:
    """Collect and publish exact nested-tier snapshots from a composed run."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "levels": {},
    }
    for level in workflow._analysis_levels():
        if level == workflow.level:
            continue
        level_report: dict[str, Any] = {}
        for branch, child in workflow.children().items():
            branch_dir = _branch_dir(root, branch)
            tier_dir = branch_dir / "tiers" / level
            tier_dir.mkdir(parents=True, exist_ok=True)
            frames: list[pd.DataFrame] = []
            sources: list[Path] = []
            for target in child.targets():
                snapshot_path = branch_dir / target.tag / ANALYSIS_TIER_FILES[level]
                if not snapshot_path.exists():
                    continue
                frame = pd.read_parquet(snapshot_path).copy()
                if branch == "references":
                    record = local_library.publish(
                        frame,
                        target,
                        workflow.method,
                        protocol=workflow._reference_protocol(level),
                        calculation_level=level,
                        source_run=root,
                        source_target_dir=branch_dir / target.tag,
                        artifact_policy=artifact_policy,
                    )
                    if shared_library is not None:
                        shared_library.publish(
                            frame,
                            target,
                            workflow.method,
                            protocol=workflow._reference_protocol(level),
                            calculation_level=level,
                            source_run=root,
                            source_target_dir=branch_dir / target.tag,
                            artifact_policy=artifact_policy,
                        )
                    frame["reference_id"] = record.reference_id
                    frame["reference_source"] = "calculated"
                frames.append(frame)
                sources.append(snapshot_path)

            if branch == "references":
                computed = _concat_reference_results(frames, source_files=sources)
                computed_path = tier_dir / "computed.parquet"
                _write_verified_parquet(computed, computed_path)
                reused_path = tier_dir / "reused.parquet"
                reused = (
                    pd.read_parquet(reused_path)
                    if reused_path.exists()
                    else pd.DataFrame()
                )
                available = [value for value in (computed, reused) if not value.empty]
                available_paths = [
                    path
                    for value, path in (
                        (computed, computed_path),
                        (reused, reused_path),
                    )
                    if not value.empty
                ]
                merged = _concat_reference_results(
                    available,
                    source_files=available_paths,
                )
            else:
                merged = _concat_tier_results(frames, source_files=sources)
            _write_verified_parquet(merged, tier_dir / "merged.parquet")
            level_report[branch] = {
                "n_targets": len(child.targets()),
                "n_snapshots": len(frames),
                "n_rows": int(len(merged)),
                "complete": int(len(merged)) == len(child.targets()),
                "path": str((tier_dir / "merged.parquet").relative_to(root)),
            }
        report["levels"][level] = level_report
    return report


def _concat_tier_results(
    frames: list[pd.DataFrame],
    *,
    source_files: list[str | Path],
) -> pd.DataFrame:
    """Concatenate compatible non-reference tier snapshots."""
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged.attrs.clear()
    merged.attrs.update(merge_dataframe_attrs(frames, source_files=source_files))
    return merged


def _concat_reference_results(
    frames: list[pd.DataFrame],
    *,
    source_files: list[str | Path],
) -> pd.DataFrame:
    """Concatenate reference results without losing canonical metadata.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        Canonical minimum-result dataframes to concatenate.
    source_files : list of str or pathlib.Path
        Source labels corresponding one-to-one with ``frames``.

    Returns
    -------
    pandas.DataFrame
        Concatenated reference results with merged dataframe attrs.

    Raises
    ------
    ValueError
        If a source lacks a canonical result contract or the contracts are
        incompatible.
    """
    if not frames:
        return pd.DataFrame()
    if len(frames) != len(source_files):
        raise ValueError("reference frames and source_files must have equal length")

    missing_contract = [
        str(source)
        for frame, source in zip(frames, source_files)
        if not isinstance(frame.attrs.get("frust_results"), dict)
    ]
    if missing_contract:
        raise ValueError(
            "Reference result has no canonical frust_results contract: "
            + ", ".join(missing_contract)
        )

    compatible_frames: list[pd.DataFrame] = []
    for frame in frames:
        compatible = frame.copy()
        contract = deepcopy(compatible.attrs["frust_results"])
        contract.pop("artifact_policy", None)
        if contract.get("calculation_level") != "full":
            # Older direct low-cost and DFT-ranked runs could record the
            # MethodPlan thermochemistry recipe while tier snapshots did not.
            # Neither result depth has frequencies, so the field is inert and
            # must not make otherwise identical cached references incompatible.
            contract.pop("thermochemistry", None)
        compatible.attrs["frust_results"] = contract
        compatible_frames.append(compatible)

    attrs = merge_dataframe_attrs(
        compatible_frames,
        source_files=source_files,
    )
    if not isinstance(attrs.get("frust_results"), dict):
        raise ValueError(
            "Reference results have incompatible canonical frust_results contracts"
        )
    merged = pd.concat(compatible_frames, ignore_index=True)
    bindings = _merged_reference_bindings(compatible_frames)
    if bindings:
        attrs["frust_reference_bindings"] = {
            "schema_version": 1,
            "bindings": bindings,
        }
    merged.attrs.clear()
    merged.attrs.update(attrs)
    return merged


def _merged_reference_bindings(frames: list[pd.DataFrame]) -> list[dict[str, Any]]:
    """Collect unique target bindings across concatenated reference frames."""
    bindings: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for frame in frames:
        metadata = frame.attrs.get("frust_reference_bindings", {})
        if not isinstance(metadata, Mapping):
            continue
        for binding in metadata.get("bindings", []) or []:
            if not isinstance(binding, Mapping):
                continue
            item = deepcopy(dict(binding))
            fingerprint = _json_hash(item)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            bindings.append(item)
    return bindings


def _branch_dir(root: Path, branch: str) -> Path:
    mapping = {
        "transition_states": root / "calculations" / "transition_states",
        "references": root / "calculations" / "references",
        "cycle_molecules": root / "calculations" / "full_cycle" / "molecular_states",
        "int3": root / "calculations" / "full_cycle" / "int3",
    }
    return mapping[branch]


def _write_verified_parquet(df: pd.DataFrame, destination: Path) -> None:
    """Atomically publish a dataframe after schema and row-count validation."""
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


def _calculation_result_paths(scope: ScreenScope) -> dict[str, str | None]:
    """Return terminal calculation-result paths for a portable run."""
    return {
        "transition_states": "calculations/transition_states/merged.parquet",
        "references": "calculations/references/merged.parquet",
        "cycle_molecules": (
            "calculations/full_cycle/molecular_states/merged.parquet"
            if scope == "full_cycle"
            else None
        ),
        "int3": (
            "calculations/full_cycle/int3/merged.parquet"
            if scope == "full_cycle"
            else None
        ),
    }


def _tier_calculation_result_paths(
    scope: ScreenScope,
    level: CalculationLevel,
) -> dict[str, str | None]:
    """Return collected nested-tier result paths for a portable run."""
    return {
        "transition_states": (
            f"calculations/transition_states/tiers/{level}/merged.parquet"
        ),
        "references": f"calculations/references/tiers/{level}/merged.parquet",
        "cycle_molecules": (
            f"calculations/full_cycle/molecular_states/tiers/{level}/merged.parquet"
            if scope == "full_cycle"
            else None
        ),
        "int3": (
            f"calculations/full_cycle/int3/tiers/{level}/merged.parquet"
            if scope == "full_cycle"
            else None
        ),
    }


def _deepest_parquet(target_dir: Path) -> Path | None:
    tier_names = set(ANALYSIS_TIER_FILES.values())
    files = (
        [path for path in target_dir.glob("*.parquet") if path.name not in tier_names]
        if target_dir.is_dir()
        else []
    )
    return max(
        files,
        key=lambda path: (path.stem.count("."), path.stat().st_mtime),
        default=None,
    )


def _coerce_method(method: MethodPlan | str | None) -> MethodPlan:
    if method is None:
        return method_preset("wb97xd3-631g")
    if isinstance(method, MethodPlan):
        return method
    return method_preset(str(method))


def _coerce_screening(screening: ScreeningPlan | str) -> ScreeningPlan:
    """Normalize a screening-plan object or preset name."""
    if isinstance(screening, ScreeningPlan):
        return screening
    return screening_preset(str(screening))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _screen_submission_plan(workflow, out_dir, execution, resources, array, parallelism, batch_size, retry, selection):
    """Validate every branch before any job or scientific artifact is created."""
    if type(retry) is not bool:
        raise ValueError('retry must be a boolean')
    children = workflow.children()
    if selection is not None and (not isinstance(selection, Mapping) or set(selection) - set(children)):
        raise ValueError(f'targets must map known branches to child targets: {list(children)}')
    if selection and not retry:
        raise ValueError('Screen targets= selects retry work; use a child workflow for initial subsets')
    if retry:
        path = Path(out_dir)/'manifest.json'
        if not path.exists():
            raise ValueError('Screen retry requires an existing compatible run manifest')
        original = json.loads(path.read_text())
        report_path = path.parent / 'run_report.json'
        if report_path.exists() and json.loads(report_path.read_text()).get('overall_status') == 'success':
            raise ValueError('Screen already finalized successfully; use a new out_dir')
        calculated_ids = {entry['target_id'] for entry in original['reference_plan'] if entry['action']=='calculate'}
    else:
        calculated_ids = {item.target.target_id for item in workflow.targets()
                          if item.branch=='references' and item.action=='calculate'}
    calculated = {branch: [target for target in child.targets()
                           if branch!='references' or target.target_id in calculated_ids]
                  for branch, child in children.items()}
    selected = {}
    plans = {}
    for branch, child in children.items():
        mode = execution or ('dft_staged' if child.dft else 'single_job')
        groups = child._stage_groups(mode)
        plans[branch] = (mode, [('single_job' if mode=='single_job' else child._group_name(group),
                                Resources(1,1,1), 'unused.parquet') for group in groups])
        if retry and mode!='single_job' and not array:
            raise NotImplementedError('Staged screen retries require array=True')
        if selection is not None:
            selected[branch] = child._select_targets(selection.get(branch, []))
        elif retry and calculated[branch]:
            report = _branch_dir(Path(out_dir),branch)/'collection_report.json'
            if not report.exists():
                raise ValueError(f'No completed collection report for retry branch {branch}')
            failed = set(json.loads(report.read_text()).get('retry_targets', []))
            selected[branch] = [target for target in calculated[branch] if target.tag in failed]
        else:
            selected[branch] = [] if retry else calculated[branch]
        allowed = {target.tag for target in calculated[branch]}
        if any(target.tag not in allowed for target in selected[branch]):
            raise ValueError(f'Branch {branch} selection contains unknown or reused targets')
    union = {name for _, groups in plans.values() for name, _, _ in groups}
    if isinstance(parallelism, Mapping) and set(parallelism) != union:
        raise ValueError(f'array_parallelism must specify exactly these screen stage groups: {sorted(union)}')
    limits = {branch: {name:parallelism[name] for name, _, _ in groups} if isinstance(parallelism,Mapping) else parallelism
              for branch, (_, groups) in plans.items()}
    for branch, (mode, groups) in plans.items():
        _plan_submission([target.tag for target in selected[branch]],groups,execution=mode,
                         array=array,array_parallelism=limits[branch],targets_per_task=batch_size)
    return children, calculated, selected, limits


@contextmanager
def _screen_submission_guard(out_dir, cluster, *, managed, retry):
    """Reserve a managed screen until its finalizer has finished mutating it."""
    root = Path(out_dir)
    directory = root/'.frust/screen_submissions'
    managed = managed or directory.exists()
    if not managed:
        yield None
        return
    with _mutation_lock(root):
        history = sorted((json.loads(path.read_text()) for path in directory.glob('*.json')),
                         key=lambda item:item['created_ns'])
        for previous in history:
            done = _completion_path(root,previous['attempt_id'],'screen_finalize',0).exists()
            if not done and not _job_terminal(previous,previous.get('job_id')):
                raise RuntimeError('Previous screen finalizer is active or unverified; overlapping screen writes are blocked')
        if history and not retry:
            raise ValueError('Managed screen already has submission history; use retry=True or a new out_dir')
        if retry and not history:
            raise ValueError('Screen retry requires managed submission history; use a new out_dir for legacy runs')
        from uuid import uuid4
        identity = uuid4().hex
        attempt = {'attempt_id':identity,'created_ns':time.time_ns(),'backend':cluster.backend,
                   'job_id':None,'log_dir':str(Path(cluster.log_dir).resolve()),
                   '_path':str(directory/f'{identity}.json')}
        old_report = root/'run_report.json'
        if old_report.exists():
            _atomic_write_submission_json(root/'.frust/screen_history'/identity/'run_report.json',json.loads(old_report.read_text()))
        yield attempt


def _wait_screen_collectors(submissions, log_dir):
    from frust.cluster.executor import _load_submitit
    submitit = _load_submitit()
    for submission in submissions.values():
        job = submitit.LocalJob(folder=log_dir,job_id=str(submission.collection_job_id))
        deadline = time.monotonic()+3600
        while not job.paths.result_pickle.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Local screen collector did not finish within one hour')
            time.sleep(.1)
