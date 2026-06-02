"""Workflow graph execution for FRUST workflows."""

from __future__ import annotations

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
    """

    tag: str
    payload: Any
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageDef:
    """One typed stage in a FRUST workflow graph."""

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
    """Runtime options shared by local and cluster workflow execution."""

    n_cores: int = 4
    mem_gb: int = 20
    debug: bool = False
    save_output_dir: bool = True
    work_dir: str | None = None


class BaseWorkflow:
    """Base class for additive FRUST workflows."""

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
        """Return workflow scientific targets without running calculators."""
        if self._target_cache is None:
            self._target_cache = self._build_targets()
        return list(self._target_cache)

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
        """Run selected workflow targets locally in the current Python process."""
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
        """Submit selected workflow targets to a submitit cluster executor."""
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
                    default=Resources(cpus=4, mem_gb=20, timeout_min=720),
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
                    default=Resources(cpus=4, mem_gb=20, timeout_min=720),
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
        """Collect final per-target parquet files from a workflow output tree."""
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
        raise NotImplementedError

    def _prepare_initial_df(
        self,
        target: WorkflowTarget,
        *,
        save_dir: Path | None,
        options: ExecutionOptions,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _step_type_for_target(self, target: WorkflowTarget) -> str | None:
        return None

    def _stage_defs(self) -> list[StageDef]:
        raise NotImplementedError

    def _select_targets(
        self,
        targets: Iterable[WorkflowTarget] | Iterable[int] | None,
    ) -> list[WorkflowTarget]:
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
        """Run a sequence of stages for one target."""
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
    """Run all stages for one target."""
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
    """Run one serializable stage group for submitit."""
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


def _coerce_method(method: MethodPlan | str | None) -> MethodPlan:
    """Normalize method input to a MethodPlan."""
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
    """Resolve resources for a stage group."""
    if resources is None:
        return default
    if group_name in resources:
        return resources[group_name]
    for stage in group:
        if stage.id in resources:
            return resources[stage.id]
    return default


def _next_parquet(current: str | None, group_name: str) -> str:
    """Return the output parquet name after a stage group."""
    if current is None:
        return "init.parquet"
    stem = current.rsplit(".", 1)[0]
    return f"{stem}.{sanitize_tag(group_name)}.parquet"


def _deepest_parquet(target_dir: Path) -> Path | None:
    """Return the deepest parquet file from one target directory."""
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
    """Return whether all normal-termination columns are true."""
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
    """Apply one calculator spec through Stepper."""
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
    """Run one non-prepare stage."""
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
    """Attach compact workflow attrs to a dataframe."""
    df.attrs.setdefault("frust_workflow", {})
    df.attrs["frust_workflow"].update(
        {
            "workflow": workflow.workflow_name,
            "method": workflow.method.name,
            "target": target.tag,
        }
    )
    return df
