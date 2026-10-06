"""Optional ωB97 follow-on calculations for portable UMA candidates."""

from __future__ import annotations

from contextlib import nullcontext
from shutil import copytree
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.constraints import render_orca_constraints, validate_dataframe_constraints
from frust.screen.runs import ScreenRun
from frust.workflows.factories import MolsWorkflow

from test_uma_full_analysis import _FakeStepper, _prepared, _run_full_uma


def test_comparison_seed_constraints_survive_parquet(tmp_path):
    seed = pd.DataFrame({
        "atoms": [["B", "H", "C"]],
        "constraint_roles": [{"cat_B": 0, "transfer_H": 1, "substrate_C": 2}],
        "constraint_spec": [[
            {"kind": "distance", "roles": ("cat_B", "transfer_H"), "value": 1.2},
            {"kind": "distance", "roles": ("transfer_H", "substrate_C"), "value": 1.3},
        ]],
    })
    path = tmp_path / "comparison_seed.parquet"
    seed.to_parquet(path)
    restored = pd.read_parquet(path)

    validate_dataframe_constraints(restored)
    assert "{B 0 1 1.2 C}" in render_orca_constraints(restored.iloc[0])


class _FakeComparisonStepper(_FakeStepper):
    def gxtb(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    @staticmethod
    def _calculate(df, name, kwargs):
        out = _FakeStepper._calculate(df, name, kwargs)
        if name.startswith("dft_"):
            out[f"{name}-oc"] = [
                [[float(x) + 1.0, float(y), float(z)] for x, y, z in coords]
                for coords in out["coords_embedded"]
            ]
        if name == "dft_hessian":
            out["dft_hessian-input.hess"] = ["mock-hessian"] * len(out)
        if name == "dft_freq":
            out["dft_freq-GE"] = out["dft_freq-EE"] + 0.02
            out["dft_freq-vibs"] = [
                [
                    {
                        "frequency": -250.0 if state == "TS1" else 35.0,
                        "mode": [[0.0, 0.0, 0.0]] * len(atoms),
                    }
                ]
                for state, atoms in zip(out["state_id"], out["atoms"])
            ]
        return out


def _run_comparison(
    tmp_path, monkeypatch, candidate_ids=None, artifact_policy="standard"
):
    parent = _run_full_uma(tmp_path, monkeypatch)
    selected = "selected" if candidate_ids is None else candidate_ids
    workflow = ft.workflows.wb97_comparison(parent, candidates=selected)
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", _prepared),
        patch("frust.workflows.core.Stepper", _FakeComparisonStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        comparison = workflow.run(
            out_dir=tmp_path / "comparison",
            n_cores=1,
            mem_gb=2,
            artifact_policy=artifact_policy,
        )
    return parent, workflow, comparison


def test_selected_comparison_uses_independent_barriers(tmp_path, monkeypatch):
    parent, workflow, comparison = _run_comparison(tmp_path, monkeypatch)
    assert len(workflow.children()["transition_states"].targets()) == 1
    assert [
        stage.id for stage in workflow.children()["transition_states"]._stage_defs()
    ] == [
        "prepare",
        "dft_preopt",
        "dft_hessian",
        "dft_ts_opt",
        "dft_freq",
        "dft_solv_sp",
    ]
    assert workflow.children()["transition_states"]._stage_defs()[1].lowest is None
    paired = comparison.method_comparison()
    assert len(paired) == 1
    assert paired.iloc[0]["match_status"] == "matched"
    assert (
        paired.iloc[0]["parent_uma_result_id"]
        == parent.barriers().iloc[0]["ts_result_id"]
    )
    assert paired.iloc[0]["wb97_ts_result_id"] != paired.iloc[0]["parent_uma_result_id"]
    assert (
        paired.iloc[0]["uma_method_fingerprint"]
        != paired.iloc[0]["wb97_method_fingerprint"]
    )
    assert (
        paired.iloc[0]["uma_delta_g_kcal_mol"]
        != paired.iloc[0]["wb97_delta_g_kcal_mol"]
    )
    assert (
        paired.iloc[0]["wb97_reference_result_ids"]
        != paired.iloc[0]["uma_reference_result_ids"]
    )
    pd.testing.assert_frame_equal(
        paired, ScreenRun(comparison.path).method_comparison()
    )
    relocated = tmp_path / "relocated_comparison"
    copytree(comparison.path, relocated)
    pd.testing.assert_frame_equal(paired, ScreenRun(relocated).method_comparison())


def test_explicit_candidates_and_missing_result(tmp_path, monkeypatch):
    parent = _run_full_uma(tmp_path, monkeypatch)
    candidate_ids = parent.candidate_barriers()["ts_result_id"].tolist()
    workflow = ft.workflows.wb97_comparison(parent, candidates=candidate_ids)
    assert len(workflow.children()["transition_states"].targets()) == 2
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", _prepared),
        patch("frust.workflows.core.Stepper", _FakeComparisonStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        comparison = workflow.run(out_dir=tmp_path / "comparison", n_cores=1, mem_gb=2)
    assert len(comparison.method_comparison()) == 2
    ts_path = comparison.path / "calculations/transition_states/merged.parquet"
    ts = pd.read_parquet(ts_path)
    ts = ts[ts["parent_uma_result_id"] != candidate_ids[1]]
    ts.to_parquet(ts_path)
    comparison.refresh_analysis()
    paired = comparison.method_comparison().set_index("parent_uma_result_id")
    assert paired.loc[candidate_ids[1], "match_status"] == "missing_wb97"
    assert pd.isna(paired.loc[candidate_ids[1], "wb97_delta_g_kcal_mol"])
    assert pd.notna(paired.loc[candidate_ids[1], "uma_delta_g_kcal_mol"])


def test_missing_wb97_reference_does_not_borrow_uma_energy(tmp_path, monkeypatch):
    _, _, comparison = _run_comparison(tmp_path, monkeypatch)
    reference_path = comparison.path / "calculations/references/merged.parquet"
    references = pd.read_parquet(reference_path)
    references = references[references["state_id"].ne("ligand")]
    references.to_parquet(reference_path)
    comparison.refresh_analysis()
    paired = comparison.method_comparison().iloc[0]
    assert paired["uma_quality_status"] == "review"
    assert pd.notna(paired["uma_delta_g_kcal_mol"])
    assert paired["wb97_quality_status"] == "incomplete"
    assert pd.isna(paired["wb97_delta_g_kcal_mol"])
    assert "missing:ligand" in paired["wb97_quality_issues"]


def test_compact_bundle_keeps_parent_candidate_link(tmp_path, monkeypatch):
    parent, _, comparison = _run_comparison(
        tmp_path, monkeypatch, artifact_policy="screening"
    )
    raw = pd.read_parquet(
        comparison.path / "calculations/transition_states/merged.parquet"
    )
    assert raw["parent_uma_result_id"].tolist() == [
        parent.barriers().iloc[0]["ts_result_id"]
    ]
    assert comparison.method_comparison().iloc[0]["match_status"] == "matched"


def test_comparison_submit_plan_and_reference_reuse(tmp_path, monkeypatch):
    parent = _run_full_uma(tmp_path, monkeypatch)
    store = tmp_path / "reference_store"
    first = ft.workflows.wb97_comparison(
        parent, reference_store=store, reuse_policy="auto_valid"
    )
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", _prepared),
        patch("frust.workflows.core.Stepper", _FakeComparisonStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        first.run(out_dir=tmp_path / "first_comparison", n_cores=1, mem_gb=2)
    second = ft.workflows.wb97_comparison(
        parent, reference_store=store, reuse_policy="auto_valid"
    )
    assert all(
        item.action == "reuse"
        for item in second.targets()
        if item.branch == "references"
    )

    class FakeExecutor:
        def __init__(self):
            self.submissions = []

        def update_parameters(self, **kwargs):
            pass

        def submit(self, function, *args, **kwargs):
            self.submissions.append((function, args, kwargs))
            return SimpleNamespace(job_id="mock-job")

    fake = FakeExecutor()
    cluster = ft.ClusterConfig(
        backend="slurm", partition="kemi1", log_dir=tmp_path / "logs"
    )
    with (
        patch("frust.workflows.core.create_executor", return_value=fake),
        patch("frust.workflows.screening.create_executor", return_value=fake),
    ):
        submission = second.submit(
            out_dir=tmp_path / "submitted",
            cluster=cluster,
            execution="dft_staged",
        )
        second.submit(
            out_dir=tmp_path / "submitted",
            cluster=cluster,
            execution="dft_staged",
        )
    assert set(submission.child_submissions) == {"transition_states"}
    stage_groups = [
        args[2]
        for function, args, _ in fake.submissions
        if function.__name__ == "_run_stage_group_submitted_job"
    ]
    assert any("dft_preopt" in group for group in stage_groups)
    assert any("dft_hessian" in group for group in stage_groups)
    assert any("dft_ts_opt" in group for group in stage_groups)
    assert any("dft_freq" in group for group in stage_groups)
    assert any("dft_solv_sp" in group for group in stage_groups)
    manifest = ScreenRun(tmp_path / "submitted").manifest
    assert manifest["comparison"]["candidate_result_ids"] == [
        parent.barriers().iloc[0]["ts_result_id"]
    ]
    assert (tmp_path / "submitted/comparison/seeds").exists()
    from test_public_arrays import Scheduler
    scheduler = Scheduler()
    with (
        patch("frust.workflows.core.create_executor", side_effect=scheduler.create),
        patch("frust.workflows.screening.create_executor", side_effect=scheduler.create),
    ):
        arrays = second.submit(out_dir=tmp_path / "submitted_arrays", cluster=cluster,
                               execution="dft_staged", array=True, array_parallelism=2)
    assert set(arrays.child_submissions) == {"transition_states"}
    assert any(fn.__name__ == "_run_stage_array_submitted_job" for fn, _, _, _ in scheduler.calls)
    assert arrays.child_submissions["transition_states"].records
    with pytest.raises(FileExistsError, match="different catalyst-screen"):
        ft.workflows.wb97_comparison(
            parent,
            candidates=parent.candidate_barriers()["ts_result_id"].tolist(),
            reference_store=store,
            reuse_policy="auto_valid",
        ).submit(out_dir=tmp_path / "submitted", cluster=cluster)
