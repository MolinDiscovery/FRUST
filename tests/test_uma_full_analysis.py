"""Portable full UMA reference and barrier contracts without live chemistry."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.stepper import Stepper
from frust.screen.runs import ScreenRun
from frust.screen.references import ReferenceLibrary, reference_identity
from frust.workflows.factories import MolsWorkflow, ScreenTSWorkflow


_FORMULAS = {
    "ligand": {"C": 5, "H": 7, "N": 1},
    "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
    "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
    "HH": {"H": 2},
    "TS1": {"C": 13, "H": 19, "B": 1, "N": 2},
}
_ENERGIES = {
    "ligand": -10.0,
    "dimer": -40.0,
    "HBpin-mol": -5.0,
    "HH": -1.0,
    "TS1": -29.9,
}


def _components() -> pd.DataFrame:
    return pd.DataFrame({
        "role": ["substrate", "catalyst"],
        "smiles": ["CN1C=CC=C1", "BC1=C(N(C)C)C=CC=C1"],
        "compound_name": ["pyrrole", "NMe"],
        "rpos": ["2", None],
    })


def _prepared(self, target, *, save_dir, options):
    del self, save_dir, options
    state = target.state_id
    atoms = [atom for atom, count in _FORMULAS[state].items() for _ in range(count)]
    cids = [0, 1] if state == "TS1" else [0]
    coords = [[float(index) * 0.1, 0.0, 0.0] for index in range(len(atoms))]
    return pd.DataFrame({
        "structure_id": [target.target_id] * len(cids),
        "state_id": [state] * len(cids),
        "state_kind": ["transition_state" if state == "TS1" else "minimum"] * len(cids),
        "system_name": [target.system.system_name] * len(cids),
        "substrate_name": [target.system.substrate_name] * len(cids),
        "catalyst_name": [target.system.catalyst_name] * len(cids),
        "rpos": [target.rpos] * len(cids),
        "cid": cids,
        "atoms": [atoms] * len(cids),
        "coords_embedded": [coords] * len(cids),
        "constraint_roles": [{"a": 0, "b": 1}] * len(cids),
        "constraint_spec": [{"distances": []}] * len(cids),
    })


class _FakeStepper:
    def __init__(self, **kwargs):
        pass

    def prune_conformers(self, df, *, name, **kwargs):
        return df.copy()

    def xtb(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    def orca(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    @staticmethod
    def _calculate(df, name, kwargs):
        if kwargs.get("lowest"):
            df = df.iloc[: kwargs["lowest"]].copy()
        out = df.copy()
        out[f"{name}-EE"] = [
            _ENERGIES[state] - (0.1 if name == "uma_freq" else 0.0) + 0.01 * cid
            for state, cid in zip(out["state_id"], out["cid"])
        ]
        out[f"{name}-oc"] = out["coords_embedded"]
        out[f"{name}-NT"] = True
        if name == "uma_hessian":
            out["uma_hessian-input.hess"] = ["mock-hessian"] * len(out)
        if name == "uma_freq":
            out["uma_freq-GE"] = [
                _ENERGIES[state] + 0.01 * cid
                for state, cid in zip(out["state_id"], out["cid"])
            ]
            out["uma_freq-vibs"] = [
                [{"frequency": -250.0 if state == "TS1" else 35.0,
                  "mode": [[0.0, 0.0, 0.0]] * len(atoms)}]
                for state, atoms in zip(out["state_id"], out["atoms"])
            ]
        return out


def _run_full_uma(
    tmp_path, monkeypatch, *, environment="uma-gas", artifact_policy="standard"
) -> ScreenRun:
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(), ts_types=["TS1"], screening=environment,
        method=environment, level="full", dimer_reference="dimer",
        top_n=2, ts_refine_n=2, prune_initial=False,
    )
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", _prepared),
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", _prepared),
        patch("frust.workflows.core.Stepper", _FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        return workflow.run(
            out_dir=tmp_path / "run", n_cores=1, mem_gb=2,
            artifact_policy=artifact_policy,
        )


@pytest.mark.parametrize("environment", ["uma-gas", "uma-alpb-chloroform"])
def test_full_uma_local_run_and_portable_barrier(tmp_path, monkeypatch, environment):
    run = _run_full_uma(tmp_path, monkeypatch, environment=environment)
    states = run.states()
    barrier = run.barriers().iloc[0]
    assert set(states["method_family"]) == {"uma"}
    assert set(states["energy_stage"]) == {"uma_freq"}
    assert set(states["geometry_stage"]) == {"uma_ts_opt", "uma_min_opt"}
    assert set(states["energy_method"]) == {"UMA-S 1.2.1 (OMol)"}
    assert set(states["solvation_model"]) == ({"alpb"} if "alpb" in environment else {None})
    assert barrier["method_family"] == "uma"
    assert barrier["energy_model"] == "omol@uma-s-1p2p1"
    assert barrier["ts_guess_profile"] == "omol-uma-s-1p2p1/gas"
    assert barrier["solvation_model"] == ("alpb" if "alpb" in environment else None)
    assert barrier["quality_status"] == "review"
    assert barrier["delta_e_kcal_mol"] == pytest.approx(0.15 * 627.5094740631)
    assert barrier["delta_g_kcal_mol"] == pytest.approx(0.1 * 627.5094740631)
    assert run.manifest["resolved_ts_spec_profile"] == "omol-uma-s-1p2p1/gas"
    assert run.manifest["ts_refine_n"] == 2
    reopened = ScreenRun(run.path).refresh_analysis()
    pd.testing.assert_frame_equal(run.barriers(), reopened.barriers())


def test_missing_reference_and_missing_gibbs_are_incomplete(tmp_path, monkeypatch):
    run = _run_full_uma(tmp_path, monkeypatch)
    reference_path = run.path / "calculations/references/merged.parquet"
    references = pd.read_parquet(reference_path)
    references = references[references["state_id"] != "ligand"].copy()
    references.to_parquet(reference_path)
    run.refresh_analysis()
    barrier = run.barriers().iloc[0]
    assert barrier["quality_status"] == "incomplete"
    assert pd.isna(barrier["delta_g_kcal_mol"])

    run2 = _run_full_uma(tmp_path / "second", monkeypatch)
    ts_path = run2.path / "calculations/transition_states/merged.parquet"
    ts = pd.read_parquet(ts_path)
    ts = ts.drop(columns=["uma_freq-GE"])
    ts.to_parquet(ts_path)
    run2.refresh_analysis()
    barrier = run2.barriers().iloc[0]
    assert barrier["quality_status"] == "incomplete"
    assert pd.isna(barrier["delta_g_kcal_mol"])


def test_compact_uma_bundle_retains_reviewable_ts_mode(tmp_path, monkeypatch):
    run = _run_full_uma(
        tmp_path, monkeypatch, artifact_policy="screening"
    )
    raw_ts = pd.read_parquet(
        run.path / "calculations/transition_states/merged.parquet"
    )
    assert "uma_freq-vibs" in raw_ts
    assert "uma_freq-frequencies_cm1" in raw_ts
    assert "uma_hessian-input.hess" not in raw_ts
    assert run.barriers().iloc[0]["quality_status"] == "review"


def test_bad_frequency_and_mixed_environment_are_rejected(tmp_path, monkeypatch):
    run = _run_full_uma(tmp_path, monkeypatch)
    ts_path = run.path / "calculations/transition_states/merged.parquet"
    ts = pd.read_parquet(ts_path)
    ts["uma_freq-vibs"] = [
        [{"frequency": -250.0}, {"frequency": -80.0}]
        for _ in range(len(ts))
    ]
    ts.to_parquet(ts_path)
    run.refresh_analysis()
    assert run.barriers().iloc[0]["quality_status"] == "invalid"

    run2 = _run_full_uma(tmp_path / "second", monkeypatch)
    reference_path = run2.path / "calculations/references/merged.parquet"
    references = pd.read_parquet(reference_path)
    contract = deepcopy(references.attrs["frust_results"])
    contract["energy_protocol"]["calculator"]["kwargs"]["uma_xtb_alpb"] = "chloroform"
    references.attrs["frust_results"] = contract
    references.to_parquet(reference_path)
    run2.refresh_analysis()
    barrier = run2.barriers().iloc[0]
    assert barrier["quality_status"] == "invalid"
    assert "mixed_electronic_energy_protocols" in barrier["quality_issues"]
    assert pd.isna(barrier["delta_e_kcal_mol"])
    assert pd.isna(barrier["delta_g_kcal_mol"])


def test_unbalanced_uma_barrier_has_no_energy(tmp_path, monkeypatch):
    run = _run_full_uma(tmp_path, monkeypatch)
    ts_path = run.path / "calculations/transition_states/merged.parquet"
    ts = pd.read_parquet(ts_path)
    ts["atoms"] = [list(atoms) + ["H"] for atoms in ts["atoms"]]
    ts.to_parquet(ts_path)

    run.refresh_analysis()
    barrier = run.barriers().iloc[0]
    assert barrier["quality_status"] == "invalid"
    assert "unbalanced_composition" in barrier["quality_issues"]
    assert pd.isna(barrier["delta_e_kcal_mol"])
    assert pd.isna(barrier["delta_g_kcal_mol"])


def test_full_uma_rejects_unsupported_scope_and_mixed_plan():
    with pytest.raises(ValueError, match="scope='barriers'"):
        ft.workflows.catalyst_screen(
            dataframe=_components(), screening="uma-gas", method="uma-gas",
            level="full", scope="full_cycle",
        )
    with pytest.raises(ValueError, match="one model and environment"):
        ft.workflows.catalyst_screen(
            dataframe=_components(), screening="uma-gas",
            method="uma-alpb-chloroform", level="full",
        )


def test_uma_reference_identity_separates_gas_and_alpb():
    common = dict(dataframe=_components(), ts_types=["TS1"], level="full")
    gas = ft.workflows.catalyst_screen(
        **common, screening="uma-gas", method="uma-gas"
    )
    alpb = ft.workflows.catalyst_screen(
        **common, screening="uma-alpb-chloroform",
        method="uma-alpb-chloroform",
    )
    gas_target = gas.children()["references"].targets()[0]
    alpb_target = alpb.children()["references"].targets()[0]
    gas_key, gas_identity = reference_identity(
        gas_target, gas.method, protocol=gas._reference_protocol()
    )
    alpb_key, alpb_identity = reference_identity(
        alpb_target, alpb.method, protocol=alpb._reference_protocol()
    )
    assert gas_key != alpb_key
    assert gas_identity["active_method"]["result_family"] == "uma"
    assert gas_identity["active_method"]["stages"]["uma_freq"]["kwargs"].get(
        "uma_xtb_alpb"
    ) is None
    assert alpb_identity["active_method"]["stages"]["uma_freq"]["kwargs"][
        "uma_xtb_alpb"
    ] == "chloroform"


def test_uma_reference_publication_rejects_wrong_environment(tmp_path, monkeypatch):
    gas_run = _run_full_uma(tmp_path, monkeypatch)
    gas_references = pd.read_parquet(
        gas_run.path / "calculations/references/computed.parquet"
    )
    gas_ligand = gas_references[
        gas_references["state_id"].eq("ligand")
    ].copy()
    alpb = ft.workflows.catalyst_screen(
        dataframe=_components(), ts_types=["TS1"],
        screening="uma-alpb-chloroform", method="uma-alpb-chloroform",
        level="full", dimer_reference="dimer",
    )
    ligand_target = next(
        target for target in alpb.children()["references"].targets()
        if target.state_id == "ligand"
    )
    library = ReferenceLibrary(tmp_path / "wrong_environment").initialize()
    with pytest.raises(ValueError, match="full UMA model and environment"):
        library.publish(
            gas_ligand, ligand_target, alpb.method,
            protocol=alpb._reference_protocol(), calculation_level="full",
        )

    gas_ligand = gas_ligand.copy()
    gas_ligand.at[gas_ligand.index[0], "uma_freq-vibs"] = [
        {"frequency": -120.0, "mode": []},
        {"frequency": -40.0, "mode": []},
    ]
    gas_workflow = ft.workflows.catalyst_screen(
        dataframe=_components(), ts_types=["TS1"], screening="uma-gas",
        method="uma-gas", level="full", dimer_reference="dimer",
    )
    gas_target = next(
        target for target in gas_workflow.children()["references"].targets()
        if target.state_id == "ligand"
    )
    with pytest.raises(ValueError, match="imaginary frequencies"):
        library.publish(
            gas_ligand, gas_target, gas_workflow.method,
            protocol=gas_workflow._reference_protocol(), calculation_level="full",
        )


def test_approved_uma_mode_selects_reviewed_candidate(tmp_path, monkeypatch):
    run = _run_full_uma(tmp_path, monkeypatch)
    candidates = run.states().query("state_id == 'TS1'")
    assert set(candidates["cid"]) == {0, 1}
    approved = candidates[candidates["cid"].eq(1)].iloc[0]
    run.set_review(str(approved["result_id"]), "approved", note="Intended transfer mode")
    selected = run.states().query("state_id == 'TS1' and quality_status == 'ready'")
    assert selected["cid"].tolist() == [1]
    assert run.barriers().iloc[0]["quality_status"] == "ready"


def test_full_uma_submitted_plan_and_matching_restart(tmp_path):
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(), ts_types=["TS1"], screening="uma-gas",
        method="uma-gas", level="full", dimer_reference="dimer",
    )

    class FakeExecutor:
        def __init__(self):
            self.parameters = []
            self.submissions = []

        def update_parameters(self, **kwargs):
            self.parameters.append(kwargs)

        def submit(self, function, *args, **kwargs):
            self.submissions.append((function, args, kwargs))
            return SimpleNamespace(job_id=f"job-{len(self.submissions)}")

    fake = FakeExecutor()
    cluster = ft.ClusterConfig(
        backend="slurm", partition="kemi1", log_dir=tmp_path / "logs"
    )
    stages = workflow.show_stages(execution="dft_staged")
    assert set(stages["group"]) == {"init"}
    assert "uma_min_opt" in set(stages["stage"])
    assert "uma_ts_opt" in set(stages["stage"])
    assert not any(stage.startswith("dft_") for stage in stages["stage"])
    with (
        patch("frust.workflows.core.create_executor", return_value=fake),
        patch("frust.workflows.screening.create_executor", return_value=fake),
    ):
        first = workflow.submit(
            out_dir=tmp_path / "submitted", cluster=cluster,
            execution="dft_staged",
        )
        second = workflow.submit(
            out_dir=tmp_path / "submitted", cluster=cluster,
            execution="dft_staged",
        )
    assert set(first.child_submissions) == {"transition_states", "references"}
    assert set(second.child_submissions) == {"transition_states", "references"}
    jobs = [
        (fn, args) for fn, args, _ in fake.submissions
        if fn.__name__ == "_run_stage_group_submitted_job"
    ]
    assert jobs
    assert any("uma_ts_opt" in args[2] for _, args in jobs)
    assert any("uma_min_opt" in args[2] for _, args in jobs)


def test_numfreq_parser_values_reach_uma_thermochemistry(tmp_path):
    from tooltoad.orca import read_gibbs_energy

    assert read_gibbs_energy(
        ["Final Gibbs free energy ... -12.345678 Eh\n"]
    ) == pytest.approx(-12.345678)
    oet = tmp_path / "oet"
    (oet / "bin").mkdir(parents=True)
    (oet / "bin" / "oet_uma").write_text("#!/bin/sh\n")
    step = Stepper(
        n_cores=1, memory_gb=2, output_base=tmp_path, save_output_dir=False
    )

    def parser_result(atoms, coords, n_cores, scr, data2file, options, xtra_inp_str, memory, read_files):
        assert "NumFreq" in options
        return {
            "normal_termination": True,
            "electronic_energy": -12.5,
            "gibbs_energy": -12.345678,
            "vibs": [{"frequency": 38.0, "mode": [[0.0, 0.0, 0.0]] * 2}],
        }

    step.orca_fn = parser_result
    frame = pd.DataFrame({
        "substrate_name": ["HH"], "atoms": [["H", "H"]],
        "coords_embedded": [[[0, 0, 0], [0, 0, 0.75]]],
    })
    with patch("frust.utils.uma.get_oet_tools", return_value=oet):
        result = step.orca(
            frame, name="uma_freq", options={"ExtOpt": None, "NumFreq": None},
            uma="omol@uma-s-1p2p1", uma_server=False,
        )
    assert result["uma_freq-GE"].iloc[0] == pytest.approx(-12.345678)
    assert result["uma_freq-EE"].iloc[0] == pytest.approx(-12.5)
    assert result["uma_freq-vibs"].iloc[0][0]["frequency"] == 38.0
