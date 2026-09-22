from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import frust as ft
from frust.artifacts import compact_result_dataframe
from frust.results import attach_result_contract, free_energy_components
from frust.screen.cleanup import cleanup_submitit_jobs, initialize_submitit_directory
from frust.workflows.core import (
    BaseWorkflow,
    ExecutionOptions,
    StageDef,
    WorkflowTarget,
    _collect_expected_outputs,
    _collect_expected_outputs_submitted,
    _run_stage_group_job,
)
from frust.workflows.screening import _cleanup_screening_targets, _finalize_run


def _full_ts_frame(structure_id: str = "TS1:system:r2") -> pd.DataFrame:
    modes = [
        {"frequency": -185.66, "displacements": np.ones((3, 3)).tolist()},
        {"frequency": 46.89, "displacements": np.zeros((3, 3)).tolist()},
        {"frequency": 74.99, "displacements": np.zeros((3, 3)).tolist()},
    ]
    frame = pd.DataFrame(
        {
            "structure_id": [structure_id],
            "state_id": ["TS1"],
            "state_kind": ["transition_state"],
            "system_name": ["system"],
            "substrate_name": ["substrate_000"],
            "catalyst_name": ["catalyst_000"],
            "rpos": [2],
            "cid": [4],
            "atoms": [["C", "H", "H"]],
            "coords_embedded": [(np.ones((3, 3)) * 9).tolist()],
            "connectivity_bonds": [[(0, 1), (0, 2)]],
            "constraint_roles": [{"reactive_c": 0}],
            "constraint_spec": [{"distances": []}],
            "dft_hessian-input.hess": [b"large-hessian"],
            "dft_ts_opt-oc": [np.arange(9, dtype=float).reshape(3, 3).tolist()],
            "dft_ts_opt-NT": [True],
            "dft_freq-EE": [-640.947196255019],
            "dft_freq-GE": [-640.69441387],
            "dft_freq-vibs": [modes],
            "dft_freq-NT": [True],
            "dft_solv_sp-EE": [-640.971996885853],
            "dft_solv_sp-NT": [True],
            "intermediate-oc": [np.ones((3, 3)).tolist()],
        }
    )
    attach_result_contract(
        frame,
        "transition_state",
        dft=True,
        calculation_level="full",
    )
    frame.attrs["frust_workflow"] = {
        "workflow": "screen_ts",
        "method": "test-method",
    }
    return frame


def test_compact_ts_schema_preserves_scientific_values():
    full = _full_ts_frame()
    recipe = {"mode": "electronic_plus_thermal"}
    before = free_energy_components(full, thermochemistry=recipe)

    compact = compact_result_dataframe(full)
    after = free_energy_components(compact, thermochemistry=recipe)

    assert compact["dft_freq-frequencies_cm1"].iloc[0] == pytest.approx(
        [-185.66, 46.89, 74.99]
    )
    assert np.array_equal(compact["dft_ts_opt-oc"].iloc[0], full["dft_ts_opt-oc"].iloc[0])
    assert compact["dft_solv_sp-EE"].iloc[0] == full["dft_solv_sp-EE"].iloc[0]
    assert after["free_energy_hartree"].iloc[0] == pytest.approx(
        before["free_energy_hartree"].iloc[0]
    )
    assert "dft_freq-vibs" not in compact
    assert "dft_hessian-input.hess" not in compact
    assert "coords_embedded" not in compact
    assert "intermediate-oc" not in compact
    assert compact.attrs["frust_artifacts"]["policy"] == "screening"
    assert compact.attrs["frust_artifacts"]["vibration_displacements"] is False


@pytest.mark.parametrize(
    ("state_id", "profile", "level", "optimized_column"),
    [
        ("ligand", "minimum", "low_cost", "xtb_opt-oc"),
        ("dimer", "minimum", "dft_ranked", "xtb_opt-oc"),
        ("ligand", "minimum", "full", "dft_opt-oc"),
        ("TS1", "transition_state", "full", "dft_ts_opt-oc"),
        ("int1", "minimum", "full", "dft_opt-oc"),
        ("INT3", "constrained_minimum", "full", "dft_opt-oc"),
    ],
)
def test_compact_schema_covers_every_screening_result_family(
    state_id,
    profile,
    level,
    optimized_column,
):
    frame = pd.DataFrame(
        {
            "structure_id": [f"{state_id}:system:r2"],
            "state_id": [state_id],
            "state_kind": [profile],
            "system_name": ["system"],
            "atoms": [["H", "H"]],
            "xtb_opt-oc": [[[0, 0, 0], [0, 0, 1]]],
            "xtb_opt-EE": [-1.0],
            "xtb_opt-NT": [True],
            "dft_rank_sp-EE": [-1.1],
            "dft_rank_sp-NT": [True],
            "dft_opt-oc": [[[0, 0, 0], [0, 0, 0.9]]],
            "dft_opt-NT": [True],
            "dft_ts_opt-oc": [[[0, 0, 0], [0, 0, 0.8]]],
            "dft_ts_opt-NT": [True],
            "dft_freq-EE": [-1.2],
            "dft_freq-GE": [-1.15],
            "dft_freq-vibs": [[{"frequency": 20.0, "mode": [[0, 0, 0], [0, 0, 0]]}]],
            "dft_freq-NT": [True],
            "dft_solv_sp-EE": [-1.25],
            "dft_solv_sp-NT": [True],
            "discarded-stage-oc": [[[9, 9, 9], [9, 9, 9]]],
        }
    )
    attach_result_contract(
        frame,
        profile,
        dft=level == "full",
        calculation_level=level,
    )

    compact = compact_result_dataframe(frame)

    assert optimized_column in compact
    assert "discarded-stage-oc" not in compact
    if level == "full":
        assert "dft_freq-frequencies_cm1" in compact
        assert "dft_freq-vibs" not in compact
    else:
        assert "dft_freq-frequencies_cm1" not in compact


def test_frequency_quality_is_identical_in_standard_and_screening_rows():
    full = _full_ts_frame()
    compact = compact_result_dataframe(full)

    full_report = ft.inspect_ts_vibrations(full)
    compact_report = ft.inspect_ts_vibrations(compact)

    pd.testing.assert_frame_equal(full_report, compact_report)
    with pytest.raises(ValueError, match="displacement vectors"):
        ft.plot_vibs(compact)


def test_screening_collector_streams_compact_rows_and_returns_small_status(tmp_path):
    targets = [
        WorkflowTarget("target_0", payload=None),
        WorkflowTarget("target_1", payload=None),
    ]
    expected: dict[str, str] = {}
    for index, target in enumerate(targets):
        target_dir = tmp_path / target.tag
        target_dir.mkdir()
        frame = _full_ts_frame(f"TS1:system:r{index}")
        frame.at[0, "constraint_roles"] = {
            "reactive_c": 0,
            f"state_specific_{index}": 1,
        }
        frame.to_parquet(target_dir / "final.parquet", index=False)
        (target_dir / "intermediate.parquet").write_bytes(b"old")
        expected[target.tag] = "final.parquet"

    workflow = SimpleNamespace(workflow_name="screen_ts")
    output = tmp_path / "merged.parquet"
    report = tmp_path / "collection_report.json"
    result = _collect_expected_outputs(
        workflow,
        targets,
        tmp_path,
        expected,
        output,
        report,
        True,
        target_retention="compact_success",
        artifact_policy="screening",
        defer_screening_cleanup=True,
    )

    assert len(result) == 2
    assert "dft_freq-vibs" not in result
    assert set(result["constraint_roles"].iloc[0]) == {
        "reactive_c",
        "state_specific_0",
        "state_specific_1",
    }
    assert json.loads(report.read_text())["n_rows"] == 2
    assert (tmp_path / "target_0/intermediate.parquet").exists()

    status = _collect_expected_outputs_submitted(
        workflow,
        targets,
        tmp_path,
        expected,
        output,
        report,
        True,
        target_retention="compact_success",
        artifact_policy="screening",
        defer_screening_cleanup=True,
    )
    assert status.row_count == 2
    assert not any(isinstance(value, pd.DataFrame) for value in status.__dict__.values())


def test_submitit_cleanup_requires_ownership_and_success(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    managed = initialize_submitit_directory(run_dir)
    (managed / "jobs/job.pkl").write_bytes(b"jobs")
    (managed / "control/finalizer.log").write_bytes(b"control")

    jobs_report = cleanup_submitit_jobs(run_dir)
    assert jobs_report["removed_bytes"] == 4
    assert not (managed / "jobs").exists()
    assert (managed / "control/finalizer.log").exists()

    with pytest.raises(ValueError, match="successful run report"):
        ft.screen.cleanup_submitit(run_dir)
    cleanup = ft.screen.cleanup_submitit(run_dir, allow_incomplete=True)
    assert cleanup["removed_bytes"] >= 7
    assert not managed.exists()

    unowned = tmp_path / "unowned"
    (unowned / ".submitit").mkdir(parents=True)
    with pytest.raises(ValueError, match="unowned"):
        ft.screen.cleanup_submitit(unowned, allow_incomplete=True)


def test_standalone_screening_collector_removes_targets_only_on_complete_success(tmp_path):
    workflow = SimpleNamespace(workflow_name="screen_ts")
    target = WorkflowTarget("target", payload=None)
    target_dir = tmp_path / target.tag
    target_dir.mkdir()
    _full_ts_frame().to_parquet(target_dir / "final.parquet", index=False)

    _collect_expected_outputs(
        workflow,
        [target],
        tmp_path,
        {target.tag: "final.parquet"},
        tmp_path / "merged.parquet",
        tmp_path / "report.json",
        True,
        artifact_policy="screening",
        return_dataframe=False,
    )

    assert (tmp_path / "merged.parquet").exists()
    assert not target_dir.exists()


def test_finalizer_rejects_missing_collection_report(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    workflow = SimpleNamespace()

    with pytest.raises(RuntimeError, match="complete collection reports"):
        _finalize_run(
            workflow,
            root,
            expected_report_paths=[str(root / "missing.json")],
            artifact_policy="screening",
        )

    report = json.loads((root / "run_report.json").read_text())
    assert report["overall_status"] == "failed"
    assert report["incomplete_collections"][0]["reason"] == "missing"


class _HessianWorkflow(BaseWorkflow):
    workflow_name = "hessian_test"

    def __init__(self):
        super().__init__(dft=False)

    def _build_targets(self):
        return [WorkflowTarget("target", payload=None)]

    def _stage_defs(self):
        return [StageDef("dft_ts_opt", "TS optimization")]


def test_hessian_is_deleted_only_after_successful_ts_optimization(tmp_path, monkeypatch):
    workflow = _HessianWorkflow()
    target = workflow.targets()[0]
    input_frame = pd.DataFrame(
        {
            "atoms": [["H", "H"]],
            "coords_embedded": [[[0, 0, 0], [0, 0, 1]]],
            "dft_hessian-input.hess": [b"hessian"],
            "dft_hessian-NT": [True],
        }
    )
    input_frame.to_parquet(tmp_path / "hessian.parquet", index=False)
    hessian_file = tmp_path / "FRUST_results/dft_hessian/row_0/input.hess"
    hessian_file.parent.mkdir(parents=True)
    hessian_file.write_bytes(b"hessian")

    def successful_calculation(*args, **kwargs):
        frame = args[3]
        assert "dft_hessian-input.hess" in frame
        out = frame.copy()
        out["dft_ts_opt-NT"] = True
        return out

    monkeypatch.setattr(
        "frust.workflows.core._run_stage_calculation",
        successful_calculation,
    )
    result = _run_stage_group_job(
        workflow,
        target,
        ["dft_ts_opt"],
        "hessian.parquet",
        "optts.parquet",
        tmp_path,
        ExecutionOptions(artifact_policy="screening"),
    )

    assert "dft_hessian-input.hess" not in result
    assert not hessian_file.exists()
    assert not (tmp_path / "hessian.parquet").exists()
    assert (tmp_path / "optts.parquet").exists()


def test_failed_ts_optimization_retains_hessian_evidence(tmp_path, monkeypatch):
    workflow = _HessianWorkflow()
    target = workflow.targets()[0]
    input_frame = pd.DataFrame(
        {
            "atoms": [["H", "H"]],
            "coords_embedded": [[[0, 0, 0], [0, 0, 1]]],
            "dft_hessian-input.hess": [b"hessian"],
            "dft_hessian-NT": [True],
        }
    )
    input_frame.to_parquet(tmp_path / "hessian.parquet", index=False)
    hessian_file = tmp_path / "FRUST_results/dft_hessian/row_0/input.hess"
    hessian_file.parent.mkdir(parents=True)
    hessian_file.write_bytes(b"hessian")

    def failed_calculation(*args, **kwargs):
        out = args[3].copy()
        out["dft_ts_opt-NT"] = False
        return out

    monkeypatch.setattr(
        "frust.workflows.core._run_stage_calculation",
        failed_calculation,
    )
    result = _run_stage_group_job(
        workflow,
        target,
        ["dft_ts_opt"],
        "hessian.parquet",
        "optts.parquet",
        tmp_path,
        ExecutionOptions(artifact_policy="screening"),
    )

    assert "dft_hessian-input.hess" in result
    assert hessian_file.exists()
    assert (tmp_path / "hessian.parquet").exists()


def test_successful_target_cleanup_leaves_failed_target_untouched(tmp_path):
    targets = [WorkflowTarget("success", None), WorkflowTarget("failed", None)]
    child = SimpleNamespace(targets=lambda: targets)
    workflow = SimpleNamespace(children=lambda: {"transition_states": child})
    branch = tmp_path / "calculations/transition_states"
    for target, normal in zip(targets, [True, False]):
        target_dir = branch / target.tag
        calculator_dir = target_dir / "FRUST_results/dft_freq/row_0"
        calculator_dir.mkdir(parents=True)
        (calculator_dir / "orca.out").write_bytes(b"large")
        pd.DataFrame({"dft_freq-NT": [normal]}).to_parquet(
            target_dir / "final.parquet",
            index=False,
        )

    cleanup = _cleanup_screening_targets(workflow, tmp_path)

    assert cleanup["n_removed_targets"] == 1
    assert not (branch / "success").exists()
    assert (branch / "failed/FRUST_results/dft_freq/row_0/orca.out").exists()
