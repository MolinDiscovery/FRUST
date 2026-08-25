"""Composed end-to-end catalyst-screen calculation workflow."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from frust.cluster.config import ClusterConfig, JobSubmissionResult, Resources
from frust.cluster.executor import create_executor, update_executor_with_dependencies
from frust.screen import expand as expand_screen
from frust.screen import read as read_screen
from frust.screen.references import ReferenceLibrary, ReferenceRecord, ReusePolicy
from frust.screen.runs import ScreenRun, build_analysis
from frust.structures import StructureTarget
from frust.utils.dataframes import merge_dataframe_attrs
from frust.workflows.factories import Int3Workflow, MolsWorkflow, ScreenTSWorkflow
from frust.workflows.methods import MethodPlan, preset as method_preset


ScreenScope = Literal["barriers", "full_cycle"]
DEFAULT_CORRECTIONS = {"TS1": -1.89, "TS3": -1.89}
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
    """

    run_dir: str
    child_submissions: dict[str, JobSubmissionResult]
    finalization_job_id: str | int | None
    backend: str


class CatalystScreenWorkflow:
    """Coordinate TS, reference, and optional full-cycle workflows."""

    workflow_name = "catalyst_screen"

    def __init__(
        self,
        *,
        csv_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
        method: MethodPlan | str | None = None,
        scope: ScreenScope = "barriers",
        corrections_kcal_mol: dict[str, float] | None = None,
        reference_store: str | Path | None = None,
        reuse_policy: ReusePolicy = "approved",
        n_confs: int | None = None,
        top_n: int = 20,
        prune_initial: bool | dict[str, Any] = True,
    ) -> None:
        if (csv_path is None) == (dataframe is None):
            raise ValueError("Provide exactly one of csv_path or dataframe")
        if scope not in {"barriers", "full_cycle"}:
            raise ValueError("scope must be 'barriers' or 'full_cycle'")
        if reuse_policy not in {"approved", "auto_valid"}:
            raise ValueError("reuse_policy must be 'approved' or 'auto_valid'")
        self.csv_path = None if csv_path is None else Path(csv_path)
        self.dataframe = None if dataframe is None else dataframe.copy()
        self.ts_types = tuple(str(value).upper() for value in ts_types)
        self.method = _coerce_method(method)
        if self.method.thermochemistry is None:
            raise ValueError(
                f"Method plan {self.method.name!r} needs a ThermochemistrySpec "
                "for catalyst-screen analysis"
            )
        self.scope = scope
        self.corrections_kcal_mol = {
            **DEFAULT_CORRECTIONS,
            **{str(key): float(value) for key, value in (corrections_kcal_mol or {}).items()},
        }
        configured_store = reference_store or os.environ.get("FRUST_REFERENCE_STORE")
        self.reference_store = None if configured_store is None else Path(configured_store)
        self.reuse_policy = reuse_policy
        self.n_confs = n_confs
        self.top_n = int(top_n)
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
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    dft=True,
                    prune_initial=self.prune_initial,
                ),
                "references": MolsWorkflow(
                    dataframe=components,
                    select_mols=self._reference_states(),
                    method=self.method,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    dft=True,
                    prune_initial=self.prune_initial,
                ),
            }
            if self.scope == "full_cycle":
                children["cycle_molecules"] = MolsWorkflow(
                    dataframe=components,
                    select_mols=["int1", "int2", "HBpin-ligand"],
                    method=self.method,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    dft=True,
                    prune_initial=self.prune_initial,
                )
                children["int3"] = Int3Workflow(
                    dataframe=components,
                    method=self.method,
                    n_confs=self.n_confs,
                    top_n=self.top_n,
                    dft=True,
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
                planned.append(CatalystScreenTarget(branch, target, action, reference_id))
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
        result.attrs["method"] = self.method.name
        result.attrs["method_fingerprint"] = self.method.fingerprint()
        result.attrs["thermochemistry"] = self.method.thermochemistry.to_dict()
        result.attrs["scope"] = self.scope
        return result

    def show_stages(self, execution: str | None = None, detail: str = "summary") -> pd.DataFrame:
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
                    frame = record.dataframe().copy()
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
    ) -> ScreenRun:
        """Run all required child workflows locally and build analysis."""
        if not save_output_dir:
            raise ValueError(
                "catalyst_screen requires save_output_dir=True so reference "
                "calculator evidence remains portable"
            )
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._write_manifest(root)
        children = self.children()
        reference_items = [item for item in self.targets() if item.branch == "references"]
        self._snapshot_reused_references(root, reference_items)

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
                    pd.DataFrame().to_parquet(branch_dir / output_name, index=False)
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
            )
            frame.to_parquet(branch_dir / output_name, index=False)
        _finalize_run(self, root)
        return ScreenRun(root)

    def submit(
        self,
        *,
        out_dir: str | Path,
        cluster: ClusterConfig,
        execution: str | None = None,
        stage_resources: dict[str, Resources] | None = None,
        debug: bool = False,
        save_output_dir: bool = True,
        work_dir: str | Path | None = None,
        collect_require_normal_termination: bool = True,
        collect_resources: Resources | None = None,
        finalize_resources: Resources | None = None,
        target_retention: str = "compact_success",
    ) -> ScreenSubmissionResult:
        """Submit all required branches and one portable-analysis finalizer."""
        if not save_output_dir:
            raise ValueError(
                "catalyst_screen requires save_output_dir=True so reference "
                "calculator evidence remains portable"
            )
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._write_manifest(root)
        children = self.children()
        reference_items = [item for item in self.targets() if item.branch == "references"]
        self._snapshot_reused_references(root, reference_items)
        submissions: dict[str, JobSubmissionResult] = {}
        dependency_ids: list[str | int] = []
        wait_paths: list[str] = []

        for branch, workflow in children.items():
            branch_dir = _branch_dir(root, branch)
            branch_targets = None
            collect_output = branch_dir / "merged.parquet"
            if branch == "references":
                branch_targets = [
                    item.target
                    for item in reference_items
                    if item.action == "calculate"
                ]
                collect_output = branch_dir / "computed.parquet"
                if not branch_targets:
                    branch_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame().to_parquet(collect_output, index=False)
                    continue
            collect_report = branch_dir / "collection_report.json"
            submission = workflow.submit(
                out_dir=branch_dir,
                cluster=cluster,
                execution=execution,
                stage_resources=stage_resources,
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
            )
            submissions[branch] = submission
            dependency_ids.append(submission.collection_job_id or submission.job_ids[-1])
            wait_paths.append(str(collect_report))

        executor = create_executor(cluster)
        update_executor_with_dependencies(
            executor,
            cluster,
            finalize_resources or DEFAULT_FINALIZE_RESOURCES,
            job_name="catalyst_screen_finalize",
            dependency_job_ids=dependency_ids,
            dependency_type="afterany",
        )
        final_job = executor.submit(
            _finalize_submitted_run,
            self,
            root,
            wait_paths if cluster.backend == "local" else None,
        )
        return ScreenSubmissionResult(
            run_dir=str(root),
            child_submissions=submissions,
            finalization_job_id=getattr(final_job, "job_id", None),
            backend=cluster.backend,
        )

    def _reference_states(self) -> list[str]:
        states = ["ligand", "dimer", "HBpin-mol", "HH"]
        if self.scope == "full_cycle":
            states.append("catalyst")
        return states

    def _reference_protocol(self) -> dict[str, Any]:
        return {
            "workflow": "frust.workflows.mols::v1",
            "n_confs": self.n_confs,
            "top_n": self.top_n,
            "prune_initial": self.prune_initial,
        }

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
        return library.find(
            target,
            self.method,
            protocol=self._reference_protocol(),
            reuse_policy=self.reuse_policy,
        )

    def _snapshot_reused_references(
        self,
        root: Path,
        items: list[CatalystScreenTarget],
    ) -> None:
        branch_dir = _branch_dir(root, "references")
        local_library = ReferenceLibrary(branch_dir).initialize()
        frames: list[pd.DataFrame] = []
        sources: list[Path] = []
        for item in items:
            if item.action != "reuse":
                continue
            record = self._cached_reference(item.target)
            if record is None:
                continue
            local = local_library.import_record(record)
            frame = local.dataframe().copy()
            frame["reference_id"] = local.reference_id
            frame["reference_source"] = "shared_library"
            frames.append(frame)
            sources.append(local.path / "result.parquet")
        reused = _concat_reference_results(frames, source_files=sources)
        reused.to_parquet(branch_dir / "reused.parquet", index=False)

    def _write_manifest(self, root: Path) -> None:
        ts_targets = self.children()["transition_states"].targets()
        reference_plan = [
            {
                "target_id": item.target.target_id,
                "state_id": item.target.state_id,
                "action": item.action,
                "reference_id": item.reference_id,
            }
            for item in self.targets()
            if item.branch == "references"
        ]
        manifest = {
            "schema_version": 1,
            "run_type": "catalyst_screen",
            "created_at": _utc_now(),
            "scope": self.scope,
            "method": self.method.to_dict(),
            "method_fingerprint": self.method.fingerprint(),
            "ts_types": list(self.ts_types),
            "corrections_kcal_mol": self.corrections_kcal_mol,
            "mechanism_id": (
                "frust_ts_barriers::v1"
                if self.scope == "barriers"
                else "frust_balanced_cycle::v1"
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
            "calculation_results": {
                "transition_states": "calculations/transition_states/merged.parquet",
                "references": "calculations/references/merged.parquet",
                "cycle_molecules": (
                    "calculations/full_cycle/molecular_states/merged.parquet"
                    if self.scope == "full_cycle"
                    else None
                ),
                "int3": (
                    "calculations/full_cycle/int3/merged.parquet"
                    if self.scope == "full_cycle"
                    else None
                ),
            },
        }
        signature_keys = [
            "scope",
            "method_fingerprint",
            "ts_types",
            "corrections_kcal_mol",
            "components",
            "systems",
            "reuse_policy",
            "reference_protocol",
            "analysis_targets",
        ]
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
        unexpected = [path for path in root.iterdir() if path.name != "manifest.json"]
        if unexpected:
            raise FileExistsError(
                f"Run directory {root} is not empty and has no compatible manifest; "
                "choose a new out_dir"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def catalyst_screen(
    *,
    csv_path: str | Path | None = None,
    dataframe: pd.DataFrame | None = None,
    ts_types: tuple[str, ...] | list[str] = ("TS1", "TS2", "TS3", "TS4"),
    method: MethodPlan | str | None = None,
    scope: ScreenScope = "barriers",
    corrections_kcal_mol: dict[str, float] | None = None,
    reference_store: str | Path | None = None,
    reuse_policy: ReusePolicy = "approved",
    n_confs: int | None = None,
    top_n: int = 20,
    prune_initial: bool | dict[str, Any] = True,
) -> CatalystScreenWorkflow:
    """Create an end-to-end catalyst-screen workflow.

    The default ``scope="barriers"`` calculates TS1--TS4 plus ligand, dimer,
    HBpin, and H2 references. ``scope="full_cycle"`` additionally calculates
    catalyst, int1, int2, HBpin-ligand, and INT3 states for a balanced profile.

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
    method : MethodPlan, str, or None, optional
        One explicit calculation plan for the entire run. A string selects a
        built-in method preset.
    scope : {"barriers", "full_cycle"}, optional
        ``"barriers"`` calculates the dependencies of the four supplied
        barrier equations. ``"full_cycle"`` adds every state needed for the
        balanced catalytic-cycle profile.
    corrections_kcal_mol : dict or None, optional
        Literal profile corrections in kcal/mol. Defaults to ``-1.89`` for
        TS1 and TS3 and zero for other states.
    reference_store : str, pathlib.Path, or None, optional
        Shared inspectable reference library. When omitted, use
        ``FRUST_REFERENCE_STORE`` if set. The completed run always receives a
        local snapshot of references it reused.
    reuse_policy : {"approved", "auto_valid"}, optional
        ``"approved"`` reuses only manually approved entries.
        ``"auto_valid"`` also permits entries that passed automatic minimum
        checks but have not been reviewed.
    n_confs : int or None, optional
        Initial conformer count forwarded consistently to every child
        workflow.
    top_n : int, optional
        Number of low-energy candidates retained before final refinement.
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
        method=method,
        scope=scope,
        corrections_kcal_mol=corrections_kcal_mol,
        reference_store=reference_store,
        reuse_policy=reuse_policy,
        n_confs=n_confs,
        top_n=top_n,
        prune_initial=prune_initial,
    )


def _finalize_submitted_run(
    workflow: CatalystScreenWorkflow,
    root: Path,
    wait_paths: list[str] | None,
) -> dict[str, Any]:
    if wait_paths:
        deadline = time.monotonic() + 3600
        while not all(Path(path).exists() for path in wait_paths) and time.monotonic() < deadline:
            time.sleep(1)
    return _finalize_run(workflow, root)


def _finalize_run(workflow: CatalystScreenWorkflow, root: Path) -> dict[str, Any]:
    reference_dir = _branch_dir(root, "references")
    local_library = ReferenceLibrary(reference_dir).initialize()
    shared_library = workflow._shared_library(initialize=True)
    computed_path = reference_dir / "computed.parquet"
    computed = pd.read_parquet(computed_path) if computed_path.exists() else pd.DataFrame()
    computed_frames: list[pd.DataFrame] = []
    computed_sources: list[Path] = []
    reference_targets = workflow.children()["references"].targets()
    for target in reference_targets:
        target_dir = reference_dir / target.tag
        result_path = _deepest_parquet(target_dir)
        if result_path is None:
            continue
        frame = pd.read_parquet(result_path)
        try:
            local_record = local_library.publish(
                frame,
                target,
                workflow.method,
                protocol=workflow._reference_protocol(),
                source_run=root,
                source_target_dir=target_dir,
            )
        except ValueError:
            continue
        frame = local_record.dataframe().copy()
        frame["reference_id"] = local_record.reference_id
        frame["reference_source"] = "calculated"
        computed_frames.append(frame)
        computed_sources.append(local_record.path / "result.parquet")
        if shared_library is not None:
            shared_library.publish(
                frame.drop(columns=["reference_id", "reference_source"], errors="ignore"),
                target,
                workflow.method,
                protocol=workflow._reference_protocol(),
                source_run=root,
                source_target_dir=target_dir,
            )
    if computed_frames:
        computed = _concat_reference_results(
            computed_frames,
            source_files=computed_sources,
        )
        computed.to_parquet(computed_path, index=False)
    reused_path = reference_dir / "reused.parquet"
    reused = pd.read_parquet(reused_path) if reused_path.exists() else pd.DataFrame()
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
    combined.to_parquet(reference_dir / "merged.parquet", index=False)

    analysis_report = build_analysis(root)
    branch_reports: dict[str, Any] = {}
    for branch in workflow.children():
        report_path = _branch_dir(root, branch) / "collection_report.json"
        if report_path.exists():
            branch_reports[branch] = json.loads(report_path.read_text())
    report = {
        "schema_version": 1,
        "finalized_at": _utc_now(),
        "analysis": analysis_report,
        "branches": branch_reports,
        "n_references_calculated": int(len(computed)),
        "n_references_reused": int(len(reused)),
    }
    (root / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    return report


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

    attrs = merge_dataframe_attrs(frames, source_files=source_files)
    if not isinstance(attrs.get("frust_results"), dict):
        raise ValueError(
            "Reference results have incompatible canonical frust_results contracts"
        )
    merged = pd.concat(frames, ignore_index=True)
    merged.attrs.clear()
    merged.attrs.update(attrs)
    return merged


def _branch_dir(root: Path, branch: str) -> Path:
    mapping = {
        "transition_states": root / "calculations" / "transition_states",
        "references": root / "calculations" / "references",
        "cycle_molecules": root / "calculations" / "full_cycle" / "molecular_states",
        "int3": root / "calculations" / "full_cycle" / "int3",
    }
    return mapping[branch]


def _deepest_parquet(target_dir: Path) -> Path | None:
    files = list(target_dir.glob("*.parquet")) if target_dir.is_dir() else []
    return max(files, key=lambda path: (path.stem.count("."), path.stat().st_mtime), default=None)


def _coerce_method(method: MethodPlan | str | None) -> MethodPlan:
    if method is None:
        return method_preset("wb97xd3-631g")
    if isinstance(method, MethodPlan):
        return method
    return method_preset(str(method))


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
