"""Method-independent candidate and validation contracts for ad10."""

from dataclasses import replace
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import frust as ft
from frust.artifacts import compact_result_dataframe
from frust.workflows.factories import RawMolsWorkflow
from test_modular_workflow_validation import (
    FakeStepper,
    components,
    run_mocked,
    workflow,
)


@pytest.mark.parametrize(
    "screening,ranking,method",
    [
        ("gxtb-default", "uma-gas", "uma-gas"),
        ("gxtb-default", "uma-gas", "r2scan-3c"),
        ("gxtb-default", "r2scan-3c", "uma-gas"),
        ("gxtb-default", "r2scan-3c", "r2scan-3c"),
        ("uma-gas", None, "uma-gas"),
        ("uma-gas", None, "r2scan-3c"),
        ("gxtb-default", None, "uma-gas"),
        ("gxtb-default", None, "r2scan-3c"),
    ],
)
def test_validation_count_survives_all_stages(tmp_path, screening, ranking, method):
    wf = workflow(
        screening=screening,
        ranking=ranking,
        method=method,
        ranking_top_n=None,
        validation_top_n=2,
        scope="full_cycle",
    )
    for child in wf.children().values():
        stages = child.show_stages(detail="full")
        validation = stages[stages.phase.eq("validation") & stages.kind.eq("calc")]
        assert validation.lowest.isna().all()
        assert not validation.empty
    run = run_mocked(wf, tmp_path, artifact_policy="screening")
    for state, name, cids, coords, kwargs in FakeStepper.calls:
        if name in {
            "uma_min_opt",
            "uma_preopt",
            "uma_hessian",
            "uma_ts_opt",
            "uma_freq",
            "dft_preopt",
            "dft_opt",
            "dft_hessian",
            "dft_ts_opt",
            "dft_freq",
        }:
            assert set(cids) == {0, 1}, (state, name, cids)
    assert len(run.candidate_barriers()) == 2
    assert run.candidate_barriers().selected.sum() == 1
    assert set(run.states().method_family) == {"uma" if method == "uma-gas" else "dft"}
    frame = pd.read_parquet(run.path / "calculations/transition_states/merged.parquet")
    assert ("uma_freq-vibs" if method == "uma-gas" else "dft_freq-vibs") in frame
    assert frame.attrs["frust_workflow"]["candidate_policy"]["count"] == 2
    assert run.barriers().delta_g_kcal_mol.notna().all()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan, np.inf])
def test_validation_count_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        workflow(validation_top_n=value)


def test_alias_conflicts_and_identity():
    with pytest.raises(ValueError, match="disagree"):
        workflow(validation_top_n=3, ranking_top_n=2)
    with pytest.raises(ValueError, match="disagree"):
        workflow(validation_top_n=2, ranking_top_n=None, ts_refine_n=3)
    explicit = workflow(validation_top_n=2, ranking_top_n=None)
    alias = workflow(ranking_top_n=2)
    assert explicit._reference_protocol("full") == alias._reference_protocol("full")
    assert explicit._reference_protocol("low_cost") == alias._reference_protocol(
        "low_cost"
    )
    assert explicit.method.fingerprint() == alias.method.fingerprint()
    assert explicit._reference_protocol("full") != workflow(
        ranking_top_n=1
    )._reference_protocol("full")


@pytest.mark.parametrize("factory", ["mols", "raw_mols", "screen_ts", "int3"])
@pytest.mark.parametrize("method", ["uma-gas", "r2scan-3c"])
def test_public_factories_expose_validation_count(factory, method):
    options = dict(method=method, validation_top_n=3, top_n=4, dft=True)
    if method == "uma-gas":
        if factory == "raw_mols":
            pass
        else:
            options["screening"] = "uma-gas"
    if factory in {"mols", "raw_mols"}:
        options["smiles"] = ["CC"]
    else:
        options["dataframe"] = components()
    wf = getattr(ft.workflows, factory)(**options)
    stages = wf.show_stages(detail="full")
    boundary = stages[
        stages.operation.eq("candidate_selection") & stages.phase.eq("validation")
    ]
    assert boundary.lowest.tolist() == [3]
    assert wf.targets()
    assert pickle.loads(pickle.dumps(wf)).show_stages().equals(wf.show_stages())


def test_capability_error_precedes_preparation():
    method = ft.workflows.methods.preset("r2scan-3c").with_stage(
        "dft_ts_opt", ft.workflows.methods.gxtb(job="opt")
    )
    wf = workflow(method=method, validation_top_n=1)
    with patch(
        "frust.workflows.factories.ScreenTSWorkflow._prepare_initial_df"
    ) as prepare:
        with pytest.raises(ValueError, match="ts_search"):
            wf.show_stages()
        prepare.assert_not_called()


@pytest.mark.parametrize("method", ["uma-gas", "r2scan-3c"])
def test_staged_grouping_uses_validation_roles(method):
    child = workflow(method=method).children()["transition_states"]
    groups = child._stage_groups("dft_staged")
    assert groups[0][-1].operation == "constrained_preoptimization"
    assert [g[0].operation for g in groups[1:]] == [
        "hessian_seed",
        "ts_search",
        "frequencies",
    ] + (["analysis_single_point"] if child.method.include_terminal_solv_sp else [])


def test_final_winner_can_differ_from_screening_and_ranking(tmp_path):
    wf = workflow(method="r2scan-3c", ranking_top_n=None, validation_top_n=2)
    original = FakeStepper.calculate.__func__

    def reversed_energy(cls, df, name, kwargs):
        out = original(cls, df, name, kwargs)
        if name in {"dft_freq", "dft_solv_sp"}:
            out[name + "-EE"] = [-40.0 - cid for cid in out.cid]
            if name == "dft_freq":
                out[name + "-GE"] = out[name + "-EE"] + 0.01
        return out

    with patch.object(FakeStepper, "calculate", classmethod(reversed_energy)):
        run = run_mocked(wf, tmp_path)
    assert run.barriers().iloc[0].ts_cid == 1
    assert run.barriers(level="low_cost").iloc[0].ts_cid == 0


@pytest.mark.parametrize("method", ["uma-gas", "r2scan-3c"])
@pytest.mark.parametrize("execution", ["dft_staged", "fully_staged"])
def test_local_staged_execution_and_repeated_run_preserve_candidate_results(
    tmp_path, method, execution
):
    wf = workflow(method=method, validation_top_n=2, ranking_top_n=None)
    run = run_mocked(wf, tmp_path, execution=execution)
    before = run.states().copy()
    again = run_mocked(pickle.loads(pickle.dumps(wf)), tmp_path, execution=execution)
    pd.testing.assert_frame_equal(before, again.states())
    assert set(before[before.state_id.eq("TS1")].cid) == {0, 1}


def test_custom_role_keys_drive_execution_results_and_reference_publication(tmp_path):
    base = ft.workflows.methods.preset("r2scan-3c")
    roles = {
        role: "custom_" + stage
        for role, stage in base.resolved_validation_roles().items()
    }
    stages = {
        key: value
        for key, value in base.stages.items()
        if key in {"xtb_preopt", "xtb_sp", "xtb_opt", "dft_rank_sp"}
    }
    stages.update(
        {
            roles[role]: base.for_stage(stage)
            for role, stage in base.resolved_validation_roles().items()
        }
    )
    method = replace(base, name="custom-roles", stages=stages, validation_roles=roles)
    wf = workflow(
        method=method, validation_top_n=2, ranking_top_n=None, scope="full_cycle"
    )
    planned = wf.show_stages(detail="full")
    assert set(
        planned[planned.phase.eq("validation") & planned.kind.eq("calc")].stage
    ) == set(roles.values())
    assert "custom_dft_hessian" in planned.group.tolist()
    original = FakeStepper.calculate.__func__

    def renamed(cls, df, name, kwargs):
        canonical = name.removeprefix("custom_")
        out = original(cls, df, canonical, kwargs)
        return (
            out.rename(
                columns={
                    col: name + col[len(canonical) :]
                    for col in out
                    if col.startswith(canonical + "-")
                }
            )
            if name != canonical
            else out
        )

    with patch.object(FakeStepper, "calculate", classmethod(renamed)):
        run = run_mocked(
            wf, tmp_path, execution="dft_staged", artifact_policy="screening"
        )
    assert run.barriers().delta_g_kcal_mol.notna().all()
    frame = pd.read_parquet(run.path / "calculations/transition_states/merged.parquet")
    contract = frame.attrs["frust_results"]
    assert contract["columns"]["optimized"]["coords"] == "custom_dft_ts_opt-oc"
    assert (
        contract["columns"]["frequency"]["frequencies"]
        == "custom_dft_freq-frequencies_cm1"
    )
    assert "custom_dft_freq-vibs" in frame
    report = __import__("json").loads(
        (run.path / "calculations/references/publication_report.json").read_text()
    )
    assert report["n_not_published"] == 0


def test_old_submitted_objects_restore_historical_defaults():
    wf = workflow()
    child = wf.children()["transition_states"]
    del wf.validation_top_n
    del child.validation_top_n
    object.__delattr__(child.method, "validation_roles")
    restored = pickle.loads(pickle.dumps(wf))
    assert restored.validation_top_n is None
    assert restored.children()["transition_states"].validation_top_n is None
    assert restored.show_stages().shape[0] > 0


def test_nonfinite_ranking_cannot_enter_validation(tmp_path):
    wf = workflow(validation_top_n=3, ranking_top_n=None)
    run = run_mocked(wf, tmp_path)
    assert set(run.states()[run.states().state_id.eq("TS1")].cid) == {0, 1}
    for state, name, cids, *_ in FakeStepper.calls:
        if name in {"uma_preopt", "uma_ts_opt", "uma_freq"}:
            assert set(cids) == {0, 1}


@pytest.mark.parametrize("method", ["uma-gas", "r2scan-3c"])
def test_quality_and_missing_thermal_data_override_lower_final_energy(tmp_path, method):
    wf = workflow(
        screening="uma-gas",
        ranking=None,
        method=method,
        validation_top_n=2,
        ranking_top_n=None,
    )
    original = FakeStepper.calculate.__func__

    def faulty_candidate(cls, df, name, kwargs):
        out = original(cls, df, name, kwargs)
        if name in {"uma_freq", "dft_freq", "dft_solv_sp"}:
            out[name + "-EE"] = [-100.0 - cid for cid in out.cid]
        if name in {"uma_freq", "dft_freq"}:
            out[name + "-GE"] = out[name + "-EE"] + 0.01
            for index, row in out.iterrows():
                if row.cid == 1:
                    if row.state_id.startswith("TS"):
                        out.at[index, name + "-GE"] = np.nan
                    else:
                        out.at[index, name + "-vibs"] = [
                            {
                                "frequency": -200.0,
                                "mode": [[0.0, 0.0, 0.0]] * len(row.atoms),
                            }
                        ]
        return out

    with patch.object(FakeStepper, "calculate", classmethod(faulty_candidate)):
        run = run_mocked(wf, tmp_path)
    assert run.barriers().iloc[0].ts_cid == 0
    candidates = run.candidate_barriers().set_index("cid")
    assert candidates.loc[1, "quality_status"] == "incomplete"
    refs = pd.read_parquet(run.path / "calculations/references/computed.parquet")
    published = refs[refs.reference_id.notna()]
    assert set(published.cid) == {0}


def test_missing_roles_reject_inspection_and_unused_roles_are_optional():
    method = ft.workflows.methods.preset("r2scan-3c")
    minimal = replace(
        method,
        validation_roles={
            "minimum_optimization": "dft_opt",
            "frequencies": "dft_freq",
            "analysis_single_point": "dft_solv_sp",
        },
    )
    assert (
        ft.workflows.raw_mols(smiles=["CC"], method=minimal, dft=True)
        .show_stages()
        .shape[0]
    )
    with pytest.raises(ValueError, match="hessian_seed"):
        workflow(method=minimal).show_stages()


def test_raw_full_result_can_omit_explicit_thermochemistry_recipe():
    from frust.workflows.core import _attach_workflow_attrs

    base = ft.workflows.methods.preset("r2scan-3c")
    wf = ft.workflows.raw_mols(
        smiles=["CC"], method=replace(base, thermochemistry=None), dft=True
    )
    frame = _attach_workflow_attrs(
        pd.DataFrame({"atoms": [["C", "C"]]}), workflow=wf, target=wf.targets()[0]
    )
    assert frame.attrs["frust_results"]["energy_protocol"]["thermochemistry"] is None


def test_alias_preserves_run_signature(tmp_path):
    first = workflow(validation_top_n=2, ranking_top_n=None)
    second = workflow(ranking_top_n=2)
    for wf, folder in [(first, tmp_path / "first"), (second, tmp_path / "second")]:
        folder.mkdir()
        wf._write_manifest(folder)
    import json

    manifests = [
        json.loads((folder / "manifest.json").read_text())
        for folder in [tmp_path / "first", tmp_path / "second"]
    ]
    assert manifests[0]["run_signature"] == manifests[1]["run_signature"]


def test_old_uma_submitted_objects_keep_their_frozen_grouping():
    wf = workflow(
        screening="uma-gas", ranking=None, method="uma-gas", ranking_top_n=None
    )
    child = wf.children()["transition_states"]
    assert len(child._stage_groups("dft_staged")) == 4
    del wf.validation_top_n
    del child.validation_top_n
    restored = pickle.loads(pickle.dumps(wf))
    assert (
        len(restored.children()["transition_states"]._stage_groups("dft_staged")) == 1
    )
    restored._children_cache = None
    assert (
        len(restored.children()["transition_states"]._stage_groups("dft_staged")) == 1
    )


def test_explicit_validation_boundary_reoptimizes_a_changed_uma_environment(tmp_path):
    wf = workflow(
        screening="uma-gas",
        ranking=None,
        method="uma-alpb-chloroform",
        validation_top_n=2,
        ranking_top_n=None,
    )
    stages = wf.children()["transition_states"].show_stages().set_index("stage")
    assert stages.loc["uma_preopt", "constraint"]
    assert "uma_refine_prune" not in stages.index
    run = run_mocked(wf, tmp_path)
    assert run.barriers().delta_g_kcal_mol.notna().all()
