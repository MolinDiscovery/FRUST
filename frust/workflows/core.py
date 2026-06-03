"""Shared execution engine for FRUST workflow objects.

This module deliberately contains no chemistry-specific target expansion. A
concrete workflow in :mod:`frust.workflows.factories` supplies lightweight
``WorkflowTarget`` objects and an ordered list of ``StageDef`` objects; this
module turns them into local ``Stepper`` calls, submitted cluster jobs, and
collected parquet outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from frust.cluster.config import ClusterConfig, JobSubmissionResult, Resources
from frust.cluster.executor import create_executor, update_executor_with_dependency
from frust.cluster.naming import sanitize_tag
from frust.schema import normalize_dataframe
from frust.stepper import Stepper
from frust.utils.dataframes import lowest_energy_rows, merge_dataframe_attrs
from frust.workflows.methods import CalculatorSpec, MethodPlan, preset as method_preset


ExecutionMode = Literal["single_job", "dft_staged", "fully_staged"]
DEFAULT_WORKFLOW_RESOURCES = Resources(cpus=4, mem_gb=20, timeout_min=720)


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
        Calculation name passed to :class:`frust.stepper.Stepper`. Stepper uses
        this as the dataframe column prefix, for example ``"OptTS"`` produces
        columns such as ``"OptTS-EE"`` and ``"OptTS-NT"``.
    kind : {"prepare", "calc", "filter"}, optional
        Stage kind. ``"prepare"`` creates the initial dataframe, ``"calc"``
        dispatches to a calculator, and ``"filter"`` keeps lowest-energy rows.
    method_stage : str, optional
        Alternate method-plan key. Use this only when a stage should reuse the
        calculator settings from another stage id.
    constraint : bool, optional
        Whether constrained calculator input should be generated from dataframe
        constraint columns where supported.
    lowest : int, optional
        Number of rows to keep after this stage, grouped by FRUST structure
        identity.
    n_cores : int, optional
        Stage-local calculator core count forwarded to Stepper. Submission
        resources still control the scheduler allocation.
    read_files : list of str, optional
        Files that ORCA should read from the previous saved calculation output.
    use_last_hess : bool, optional
        Whether ORCA should reuse the previous Hessian when supported.
    save_files : list of str, optional
        Extra files to preserve from the stage output directory.
    """

    id: str
    name: str
    kind: Literal["prepare", "calc", "filter"] = "calc"
    method_stage: str | None = None
    constraint: bool = False
    lowest: int | None = None
    n_cores: int | None = None
    read_files: list[str] | None = None
    use_last_hess: bool = False
    save_files: list[str] | None = None


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
    """

    n_cores: int = 4
    mem_gb: int = 20
    debug: bool = False
    save_output_dir: bool = True
    work_dir: str | None = None


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
        ``"wb97xd3-631g"`` (default ORCA wB97X-D3/6-31G** workflow), and
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
    stage grouping, local execution, cluster submission, and result collection.
    """

    workflow_name = "workflow"

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

    def show_stages(self, *, execution: ExecutionMode | None = None) -> pd.DataFrame:
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

        Returns
        -------
        pandas.DataFrame
            One row per active workflow stage. Important columns are ``group``
            for the scheduler/resource group, ``stage`` for the workflow stage
            id, ``method_key`` for the calculator key read from
            ``method.stages``, ``engine`` for the calculator backend, and
            ``options`` for the compact calculator keywords. The table
            describes the method-plan keys this workflow will actually use; it
            does not list unused entries from ``method.stages`` and does not
            build targets, embed structures, or run calculators.

        Examples
        --------
        Inspect the stage groups before deciding ``stage_resources``:

        >>> import frust as ft
        >>> wf = ft.workflows.raw_mols(csv_path="raw_dimers.csv", method="r2scan-3c", dft=True)
        >>> wf.show_stages()[["group", "stage", "engine"]]
        """
        mode = execution or ("dft_staged" if self.dft else "single_job")
        rows: list[dict[str, Any]] = []
        for group in self._stage_groups(mode):
            group_name = "single_job" if mode == "single_job" else self._group_name(group)
            for stage in group:
                method_key = stage.method_stage or stage.id
                spec: CalculatorSpec | None = None
                if stage.kind == "calc":
                    spec = self.method.for_stage(method_key)

                rows.append(
                    {
                        "group": group_name,
                        "stage": stage.id,
                        "calculation": stage.name,
                        "kind": stage.kind,
                        "method_key": method_key if stage.kind == "calc" else None,
                        "engine": _stage_engine(stage, spec),
                        "options": _format_stage_options(spec.options if spec is not None else None),
                        "lowest": stage.lowest,
                        "constraint": stage.constraint,
                        "n_cores": stage.n_cores,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "group",
                "stage",
                "calculation",
                "kind",
                "method_key",
                "engine",
                "options",
                "lowest",
                "constraint",
                "n_cores",
            ],
        )

    def run(
        self,
        *,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None = None,
        out_dir: str | Path | None = None,
        execution: ExecutionMode | None = None,
        n_cores: int = 4,
        mem_gb: int = 20,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Run selected workflow targets locally.

        Parameters
        ----------
        targets : iterable of WorkflowTarget or int or None, optional
            Targets to run. Integers select positions from ``wf.targets()``. If
            omitted, all workflow targets are run.
        out_dir : str or pathlib.Path or None, optional
            Output root. When provided, FRUST creates one subdirectory per
            target and writes staged parquet files inside each target directory.
            When omitted, stages run in memory and no staged parquet files are
            written.
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
        selected = self._select_targets(targets)
        options = ExecutionOptions(
            n_cores=n_cores,
            mem_gb=mem_gb,
            debug=debug,
            save_output_dir=save_output_dir,
            work_dir=None if work_dir is None else str(work_dir),
        )
        frames: list[pd.DataFrame] = []
        root = Path(out_dir) if out_dir is not None else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

        for target in selected:
            save_dir = None if root is None else root / target.tag
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
            if save_dir is None:
                df = _run_target_job(self, target, save_dir, options)
            else:
                mode = execution or "single_job"
                groups = self._stage_groups(mode)
                if mode == "single_job":
                    df = _run_target_job(self, target, save_dir, options)
                    df.to_parquet(save_dir / "final.parquet")
                else:
                    current_parquet: str | None = None
                    df = pd.DataFrame()
                    for group in groups:
                        output_parquet = _next_parquet(current_parquet, self._group_name(group))
                        df = _run_stage_group_job(
                            self,
                            target,
                            [stage.id for stage in group],
                            current_parquet,
                            output_parquet,
                            save_dir,
                            options,
                        )
                        current_parquet = output_parquet
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
        targets: Iterable[WorkflowTarget] | Iterable[int] | None = None,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
    ) -> JobSubmissionResult:
        """Submit selected workflow targets to a submitit cluster executor.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Root output directory. FRUST creates one subdirectory per selected
            workflow target and writes staged parquet files inside that target
            directory.
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
            ``"dft_opt"``, ``"freq"``, and ``"solv"``; screen TS workflows
            usually use ``"init"``, ``"hess"``, ``"optts"``, ``"freq"``, and
            ``"solv"``.
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

        Returns
        -------
        frust.cluster.config.JobSubmissionResult
            Submitted scheduler job IDs, target tags, target save directories,
            workflow execution mode, and backend.
        """
        selected = self._select_targets(targets)
        mode = execution or ("dft_staged" if self.dft else "single_job")
        groups = self._stage_groups(mode)
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        executor = create_executor(cluster)

        job_ids: list[str | int] = []
        tags: list[str] = []
        save_dirs: list[str] = []

        for target in selected:
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
                    mem_gb=resources.mem_gb,
                    debug=debug,
                    save_output_dir=save_output_dir,
                    work_dir=str(work_dir or cluster.work_dir) if (work_dir or cluster.work_dir) else None,
                )
                job = executor.submit(_run_target_job, self, target, target_dir, options)
                job_ids.append(getattr(job, "job_id", f"{target.tag}_workflow"))
                continue

            for group in groups:
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
                    mem_gb=resources.mem_gb,
                    debug=debug,
                    save_output_dir=save_output_dir,
                    work_dir=str(work_dir or cluster.work_dir) if (work_dir or cluster.work_dir) else None,
                )
                job = executor.submit(
                    _run_stage_group_job,
                    self,
                    target,
                    [stage.id for stage in group],
                    current_parquet,
                    output_parquet,
                    target_dir,
                    options,
                )
                job_ids.append(getattr(job, "job_id", f"{target.tag}_{group_name}"))
                last_job = job
                current_parquet = output_parquet

        return JobSubmissionResult(
            job_ids=job_ids,
            tags=tags,
            save_dirs=save_dirs,
            mode=f"{self.workflow_name}:{mode}",
            backend=cluster.backend,
        )

    def collect(
        self,
        out_dir: str | Path,
        *,
        output: str | Path | None = None,
        require_normal_termination: bool = False,
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
            Optional parquet path for the merged dataframe.
        require_normal_termination : bool, optional
            If ``True``, skip target outputs where normal-termination columns
            ending in ``"-NT"`` are present and not all true.

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
        root = Path(out_dir)
        frames: list[pd.DataFrame] = []
        files: list[Path] = []
        skipped: list[Path] = []

        for target in self.targets():
            target_dir = root / target.tag
            final_file = _deepest_parquet(target_dir)
            if final_file is None:
                continue
            df = normalize_dataframe(pd.read_parquet(final_file))
            if require_normal_termination and not _all_normal_terminated(df):
                skipped.append(final_file)
                continue
            frames.append(df)
            files.append(final_file)

        if not frames:
            raise FileNotFoundError(f"No final workflow parquet files found under {root}")

        merged = pd.concat(frames, ignore_index=True)
        merged.attrs.update(
            merge_dataframe_attrs(
                frames,
                source_files=[str(path) for path in files],
                skipped_files=[str(path) for path in skipped],
            )
        )
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out_path)
        return merged

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
            raise ValueError("execution must be 'single_job', 'dft_staged', or 'fully_staged'")

        if not self.dft:
            return [stages]

        dft_stage_ids = {"hess", "optts", "freq", "solv", "dft_opt"}
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
        for stage in stages:
            if stage.kind == "prepare":
                df = self._prepare_initial_df(target, save_dir=save_dir, options=options)
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
        if df is None:
            raise ValueError("No workflow stages were run")
        return df


def _run_target_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    save_dir: str | Path | None,
    options: ExecutionOptions,
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
    return workflow._run_stage_group(
        target,
        workflow._stage_defs(),
        input_df=None,
        save_dir=target_dir,
        options=options,
    )


def _run_stage_group_job(
    workflow: BaseWorkflow,
    target: WorkflowTarget,
    stage_ids: list[str],
    input_parquet: str | None,
    output_parquet: str,
    save_dir: str | Path,
    options: ExecutionOptions,
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
    input_df = None if input_parquet is None else pd.read_parquet(target_dir / input_parquet)
    stages_by_id = {stage.id: stage for stage in workflow._stage_defs()}
    stages = [stages_by_id[stage_id] for stage_id in stage_ids]
    df = workflow._run_stage_group(
        target,
        stages,
        input_df=input_df,
        save_dir=target_dir,
        options=options,
    )
    df.to_parquet(target_dir / output_parquet)
    return df


def _stage_engine(stage: StageDef, spec: CalculatorSpec | None) -> str | None:
    """Return the inspection-table engine label for a workflow stage."""
    if stage.kind == "prepare":
        return "prepare"
    if stage.kind == "filter":
        return "filter"
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


def _coerce_method(method: MethodPlan | str | None) -> MethodPlan:
    """Normalize user method input to a method plan.

    Parameters
    ----------
    method : MethodPlan or str or None
        Explicit method plan, registered preset name, or ``None`` for the
        workflow default. Built-in preset strings are ``"r2scan-3c"`` for ORCA
        r2SCAN-3c composite DFT stages, ``"wb97xd3-631g"`` for the default ORCA
        wB97X-D3/6-31G** workflow, and ``"r2scan-def2svp"`` for ORCA
        R2SCAN/def2-SVP DFT stages.

    Returns
    -------
    MethodPlan
        Resolved calculator plan.
    """
    if method is None:
        return method_preset("wb97xd3-631g")
    if isinstance(method, str):
        return method_preset(method)
    if isinstance(method, MethodPlan):
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
    for stage in group:
        if stage.id in resources:
            return resources[stage.id]
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
        ``"init.hess.optts.parquet"``.
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
    files = sorted(target_dir.glob("*.parquet"))
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
    nt_cols = [col for col in df.columns if str(col).endswith("-NT")]
    if not nt_cols:
        return True
    return bool(df[nt_cols].fillna(False).astype(bool).all().all())


def _apply_calculator(
    step: Stepper,
    df: pd.DataFrame,
    stage: StageDef,
    spec: CalculatorSpec,
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

    Returns
    -------
    pandas.DataFrame
        Dataframe returned by ``Stepper.xtb(...)``, ``Stepper.gxtb(...)``, or
        ``Stepper.orca(...)``.
    """
    kwargs = {
        "name": stage.name,
        "options": spec.options,
        "constraint": stage.constraint,
        "lowest": stage.lowest,
        "n_cores": stage.n_cores,
        **spec.kwargs,
    }
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
        return lowest_energy_rows(df)

    step = Stepper(
        step_type=workflow._step_type_for_target(target),
        n_cores=options.n_cores,
        memory_gb=options.mem_gb,
        debug=options.debug,
        output_base=save_dir,
        save_output_dir=options.save_output_dir,
        work_dir=options.work_dir,
    )
    spec = workflow.method.for_stage(stage.method_stage or stage.id)
    return _apply_calculator(step, df, stage, spec)


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
    df.attrs.setdefault("frust_workflow", {})
    df.attrs["frust_workflow"].update(
        {
            "workflow": workflow.workflow_name,
            "method": workflow.method.name,
            "target": target.tag,
        }
    )
    return df
