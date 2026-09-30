"""Mocked g-xTB geometry screening, UMA reranking, and DFT validation."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.screen.references import reference_identity
from frust.screen.runs import ScreenRun
from frust.workflows.factories import MolsWorkflow, ScreenTSWorkflow


FORMULAS = {
    "ligand": {"C": 5, "H": 7, "N": 1},
    "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
    "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
    "HH": {"H": 2},
    "TS1": {"C": 13, "H": 19, "B": 1, "N": 2},
}
ENERGIES = {
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
    atoms = [atom for atom, count in FORMULAS[state].items() for _ in range(count)]
    cids = [0, 1, 2, 3] if state == "TS1" else [0]
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
        "coords_embedded": [[[float(cid), 0.0, 0.0]] * len(atoms) for cid in cids],
        "constraint_roles": [{"a": 0, "b": 1}] * len(cids),
        "constraint_spec": [{"distances": []}] * len(cids),
    })


class _FakeStepper:
    calls: list[tuple[str, str, tuple[int, ...]]] = []
    ranking_energies = {0: -30.0, 1: -32.0, 2: float("nan"), 3: -40.0}

    def __init__(self, **kwargs):
        del kwargs

    def xtb(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    def gxtb(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    def orca(self, df, *, name, **kwargs):
        return self._calculate(df, name, kwargs)

    @classmethod
    def _calculate(cls, df, name, kwargs):
        state = str(df["state_id"].iloc[0])
        cls.calls.append((state, name, tuple(int(cid) for cid in df["cid"])))
        out = df.copy()
        if name == "uma_rank_sp":
            assert "xtb_opt-oc" in out
            assert all(
                coords[0][0] == float(cid) + 10.0
                for coords, cid in zip(out["xtb_opt-oc"], out["cid"])
            )
            out[f"{name}-EE"] = [
                cls.ranking_energies[int(cid)] if state == "TS1" else ENERGIES[state]
                for cid in out["cid"]
            ]
        elif name == "xtb_sp" and state == "TS1":
            out[f"{name}-EE"] = [-4.0, -3.0, -2.0, -1.0]
        elif name == "xtb_opt" and state == "TS1":
            out[f"{name}-EE"] = [-4.0, -3.0, -2.0][: len(out)]
        else:
            out[f"{name}-EE"] = [
                ENERGIES[state] + 0.01 * int(cid) for cid in out["cid"]
            ]
        out[f"{name}-oc"] = [
            [[float(cid) + (10.0 if name == "xtb_opt" else 20.0), 0.0, 0.0]]
            * len(atoms)
            for cid, atoms in zip(out["cid"], out["atoms"])
        ]
        out[f"{name}-NT"] = True
        if name == "dft_freq":
            out["dft_freq-GE"] = [
                ENERGIES[state] + 0.01 * int(cid) for cid in out["cid"]
            ]
            out["dft_freq-vibs"] = [
                [{"frequency": -250.0 if state == "TS1" else 35.0,
                  "mode": [[0.0, 0.0, 0.0]] * len(atoms)}]
                for atoms in out["atoms"]
            ]
        if kwargs.get("lowest"):
            out = out.sort_values(f"{name}-EE", kind="stable").head(kwargs["lowest"])
        return out


def _workflow(*, ranking="uma-gas", level="full", top_n=3, uma_rank_top_n=2):
    return ft.workflows.catalyst_screen(
        dataframe=_components(), ts_types=["TS1"], screening="gxtb-default",
        ranking=ranking, method="wb97xd3-631g", level=level,
        dimer_reference="dimer", top_n=top_n,
        uma_rank_top_n=uma_rank_top_n, prune_initial=False,
    )


def _run(tmp_path, *, level="full", ranking="uma-gas", ranking_energies=None) -> ScreenRun:
    workflow = _workflow(level=level, ranking=ranking)
    _FakeStepper.calls = []
    _FakeStepper.ranking_energies = ranking_energies or {
        0: -30.0, 1: -32.0, 2: float("nan"), 3: -40.0
    }
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", _prepared),
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", _prepared),
        patch("frust.workflows.core.Stepper", _FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        return workflow.run(out_dir=tmp_path / "run", n_cores=1, mem_gb=2)


@pytest.mark.parametrize(
    "ranking, solvent", [("uma-gas", None), ("uma-alpb-chloroform", "ALPB(chloroform)")]
)
def test_hybrid_plan_and_reference_identity(ranking, solvent):
    workflow = _workflow(ranking=ranking)
    assert workflow._analysis_levels() == ("low_cost", "uma_ranked", "full")
    assert workflow.plan().attrs["uma_rank_top_n"] == 2
    for child in workflow.children().values():
        stages = child.show_stages()
        ids = stages["stage"].tolist()
        assert ids.index("xtb_sp") < ids.index("xtb_sp_filter") < ids.index("xtb_opt")
        assert ids.index("xtb_opt") < ids.index("uma_rank_sp") < ids.index("uma_rank_filter")
        assert ids.index("uma_rank_filter") < ids.index("dft_opt" if child.workflow_name == "mols" else "dft_preopt")
        assert not any(stage in ids for stage in ("uma_opt", "dft_rank_sp"))
        assert stages.set_index("stage").loc["xtb_sp_filter", "lowest"] == 3
        assert stages.set_index("stage").loc["uma_rank_filter", "lowest"] == 2
        assert stages.set_index("stage").loc["uma_rank_sp", "solvent"] == solvent
    target = workflow.children()["references"].targets()[0]
    _, ranked_identity = reference_identity(
        target, workflow.method, protocol=workflow._reference_protocol("uma_ranked"),
        calculation_level="uma_ranked",
    )
    _, full_identity = reference_identity(
        target, workflow.method, protocol=workflow._reference_protocol("full"),
        calculation_level="full",
    )
    assert "uma_rank_sp" in ranked_identity["active_method"]["stages"]
    assert "uma_rank_sp" in full_identity["active_method"]["stages"]
    assert ranked_identity["active_method"]["stages"]["uma_rank_sp"]["kwargs"].get(
        "uma_xtb_alpb"
    ) == ("chloroform" if solvent else None)


def test_ranking_environment_separates_cached_references():
    gas = _workflow(ranking="uma-gas")
    alpb = _workflow(ranking="uma-alpb-chloroform")
    gas_target = gas.children()["references"].targets()[0]
    alpb_target = alpb.children()["references"].targets()[0]
    gas_ranked, _ = reference_identity(
        gas_target, gas.method, protocol=gas._reference_protocol("uma_ranked"),
        calculation_level="uma_ranked",
    )
    alpb_ranked, _ = reference_identity(
        alpb_target, alpb.method, protocol=alpb._reference_protocol("uma_ranked"),
        calculation_level="uma_ranked",
    )
    gas_low, _ = reference_identity(
        gas_target, gas.method, protocol=gas._reference_protocol("low_cost"),
        calculation_level="low_cost",
    )
    alpb_low, _ = reference_identity(
        alpb_target, alpb.method, protocol=alpb._reference_protocol("low_cost"),
        calculation_level="low_cost",
    )
    assert gas_ranked != alpb_ranked
    assert gas_low == alpb_low


@pytest.mark.parametrize("ranking", ["uma-gas", "uma-alpb-chloroform"])
def test_hybrid_full_run_has_independent_uma_and_dft_tiers(tmp_path, ranking):
    run = _run(tmp_path, ranking=ranking)
    assert (run.path / "analysis/states_by_level.parquet").exists()
    assert run.manifest["ranking"]["stage_id"] == "uma_rank_sp"
    assert run.manifest["uma_rank_top_n"] == 2
    assert set(run.summary()["ranking_method"]) == {ranking}
    assert set(run.summary()["method"]) == {"wb97xd3-631g"}
    assert set(run.states(level="uma_ranked")["energy_stage"]) == {"uma_rank_sp"}
    assert set(run.states(level="uma_ranked")["geometry_stage"]) == {"xtb_opt"}
    assert set(run.states(level="uma_ranked")["method_family"]) == {"uma"}
    assert set(run.states(level="uma_ranked")["solvation_model"]) == (
        {"alpb"} if "alpb" in ranking else {None}
    )
    assert set(run.states(level="full")["method_family"]) == {"dft"}
    ranked_barrier = run.barriers(level="uma_ranked").iloc[0]
    full_barrier = run.barriers(level="full").iloc[0]
    assert ranked_barrier["method_family"] == "uma"
    assert pd.notna(ranked_barrier["delta_e_kcal_mol"])
    assert pd.isna(ranked_barrier["delta_g_kcal_mol"])
    assert full_barrier["method_family"] == "dft"
    assert pd.notna(full_barrier["delta_e_kcal_mol"])
    assert pd.notna(full_barrier["delta_g_kcal_mol"])
    assert len(run.compare_barriers()) == 1
    assert {"delta_e_low_cost_kcal_mol", "delta_e_uma_ranked_kcal_mol", "delta_e_full_kcal_mol"}.issubset(run.compare_barriers())
    assert (run.path / "calculations/transition_states/tiers/uma_ranked/merged.parquet").exists()
    assert (run.path / "calculations/references/tiers/uma_ranked/merged.parquet").exists()
    ranked_ts = pd.read_parquet(
        run.path / "calculations/transition_states/tiers/uma_ranked/merged.parquet"
    )
    assert {"xtb_opt-EE", "xtb_opt-oc", "uma_rank_sp-EE"}.issubset(ranked_ts)
    assert ranked_ts.attrs["frust_results"]["columns"]["analysis"]["electronic_energy"] == "uma_rank_sp-EE"
    assert ranked_ts.attrs["frust_results"]["columns"]["optimized"]["coords"] == "xtb_opt-oc"
    assert ranked_ts.attrs["frust_analysis_tier"]["selection_stage"] == "uma_rank_sp"
    raw_ts = pd.read_parquet(
        run.path / "calculations/transition_states/merged.parquet"
    )
    assert "dft_solv_sp-EE" in raw_ts and "uma_opt-EE" not in raw_ts
    steps = ft.show_steps(raw_ts)
    assert steps.loc["xtb_sp_filter", "lowest"] == 3
    assert steps.loc["uma_rank_filter", "lowest"] == 2
    ts_calls = [(name, cids) for state, name, cids in _FakeStepper.calls if state == "TS1"]
    assert ("uma_rank_sp", (0, 1, 2)) in ts_calls
    assert ("dft_preopt", (1, 0)) in ts_calls
    assert all(name != "uma_opt" for name, _ in ts_calls)

    reopened = ScreenRun(run.path).refresh_analysis()
    pd.testing.assert_frame_equal(run.compare_barriers(), reopened.compare_barriers())


def test_uma_ranked_terminal_ties_and_missing_energy(tmp_path):
    run = _run(
        tmp_path, level="uma_ranked",
        ranking_energies={0: -32.0, 1: -32.0, 2: float("nan"), 3: -40.0},
    )
    assert run.available_analysis_levels() == ("low_cost", "uma_ranked")
    assert run.states().query("state_id == 'TS1'")["cid"].tolist() == [0]
    assert run.states().query("state_id == 'TS1'")["energy_stage"].tolist() == ["uma_rank_sp"]
    assert run.barriers().iloc[0]["method_family"] == "uma"
    assert not any(name.startswith("dft_") for _, name, _ in _FakeStepper.calls)


def test_hybrid_rejects_incompatible_choices():
    with pytest.raises(ValueError, match="requires UMA ranking"):
        _workflow(ranking=None, level="uma_ranked")
    with pytest.raises(ValueError, match="g-xTB screening"):
        ft.workflows.catalyst_screen(
            dataframe=_components(), screening="uma-gas", ranking="uma-gas",
            level="full",
        )
    with pytest.raises(ValueError, match="include_dft_rank_sp=False"):
        ft.workflows.catalyst_screen(
            dataframe=_components(), ranking="uma-gas", level="full",
            include_dft_rank_sp=True,
        )


def test_hybrid_submit_plan_and_restart(tmp_path):
    workflow = _workflow()

    class FakeExecutor:
        def __init__(self):
            self.submissions = []

        def update_parameters(self, **kwargs):
            pass

        def submit(self, function, *args, **kwargs):
            self.submissions.append((function, args, kwargs))
            return SimpleNamespace(job_id="mock-job")

    cluster = ft.ClusterConfig(
        backend="slurm", partition="kemi1", log_dir=tmp_path / "logs"
    )
    fake = FakeExecutor()
    with (
        patch("frust.workflows.core.create_executor", return_value=fake),
        patch("frust.workflows.screening.create_executor", return_value=fake),
    ):
        first = workflow.submit(
            out_dir=tmp_path / "submitted", cluster=cluster, execution="dft_staged"
        )
        second = workflow.submit(
            out_dir=tmp_path / "submitted", cluster=cluster, execution="dft_staged"
        )
    assert set(first.child_submissions) == {"transition_states", "references"}
    assert set(second.child_submissions) == {"transition_states", "references"}
    manifest = ScreenRun(tmp_path / "submitted").manifest
    assert manifest["ranking"]["stage_id"] == "uma_rank_sp"
    assert manifest["analysis_levels"] == ["low_cost", "uma_ranked", "full"]
    jobs = [
        args for function, args, _ in fake.submissions
        if function.__name__ == "_run_stage_group_submitted_job"
    ]
    assert any("uma_rank_sp" in args[2] for args in jobs)
    with pytest.raises(FileExistsError, match="different catalyst-screen manifest"):
        _workflow(ranking="uma-alpb-chloroform").submit(
            out_dir=tmp_path / "submitted", cluster=cluster,
            execution="dft_staged",
        )
