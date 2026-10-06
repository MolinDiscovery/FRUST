"""Selected-geometry handoff and portable tiers without live calculators."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import frust as ft
from frust.screen.references import reference_identity
from frust.stepper import Stepper
from frust.workflows.factories import MolsWorkflow, ScreenTSWorkflow

FORMULAS = {
    "ligand": {"C": 5, "H": 7, "N": 1},
    "dimer": {"C": 16, "H": 24, "B": 2, "N": 2},
    "HBpin-mol": {"C": 6, "H": 13, "B": 1, "O": 2},
    "HH": {"H": 2},
    "TS1": {"C": 13, "H": 19, "B": 1, "N": 2},
}
for name in ["dimer_bh_bridged", "dimer_eight_membered"]:
    FORMULAS[name] = FORMULAS["dimer"]
ENERGIES = dict(
    ligand=-10.0,
    dimer=-40.0,
    dimer_bh_bridged=-39.9,
    dimer_eight_membered=-39.8,
    HH=-1.0,
    TS1=-29.9,
)
ENERGIES["HBpin-mol"] = -5.0


def components():
    return pd.DataFrame(
        dict(
            role=["substrate", "catalyst"],
            smiles=["CN1C=CC=C1", "BC1=C(N(C)C)C=CC=C1"],
            compound_name=["pyrrole", "NMe"],
            rpos=["2", None],
        )
    )


def workflow(**kwargs):
    options = dict(
        dataframe=components(),
        ts_types=["TS1"],
        screening="gxtb-default",
        ranking="uma-alpb-chloroform",
        method="uma-gas-opt-alpb-chloroform",
        level="full",
        ranking_top_n=1,
        top_n=3,
        prune_initial=False,
    )
    options.update(kwargs)
    return ft.workflows.catalyst_screen(**options)


def prepared(self, target, *, save_dir, options):
    atoms = [
        atom for atom, count in FORMULAS[target.state_id].items() for _ in range(count)
    ]
    return pd.DataFrame(
        dict(
            structure_id=[target.target_id] * 4,
            state_id=[target.state_id] * 4,
            state_kind=[target.state_kind] * 4,
            system_name=[target.system.system_name] * 4,
            substrate_name=[target.system.substrate_name] * 4,
            catalyst_name=[target.system.catalyst_name] * 4,
            rpos=[target.rpos] * 4,
            cid=[0, 1, 2, 3],
            atoms=[atoms] * 4,
            coords_embedded=[[[float(cid), 0.0, 0.0]] * len(atoms) for cid in range(4)],
            constraint_roles=[{"a": 0, "b": 1}] * 4,
            constraint_spec=[{"distances": []}] * 4,
        )
    )


class FakeStepper:
    calls = []

    def __init__(self, **kwargs):
        pass

    def xtb(self, df, *, name, **kwargs):
        return self.calculate(df, name, kwargs)

    def gxtb(self, df, *, name, **kwargs):
        return self.calculate(df, name, kwargs)

    def orca(self, df, *, name, **kwargs):
        return self.calculate(df, name, kwargs)

    @classmethod
    def calculate(cls, df, name, kwargs):
        state = df.state_id.iloc[0]
        coords = df[Stepper._last_coord_col(df)].tolist()
        cls.calls.append((state, name, df.cid.tolist(), coords, kwargs))
        out = df.copy()
        energy = ENERGIES[state]
        if name in ["uma_rank_sp", "dft_rank_sp"]:
            out[name + "-EE"] = [
                energy + [0.0, -0.2, np.nan, -10.0][cid] for cid in df.cid
            ]
        else:
            out[name + "-EE"] = [
                energy + 0.01 * cid - (1.0 if name == "uma_solv_sp" else 0.0)
                for cid in df.cid
            ]
        offset = {
            "xtb_preopt": 1.0,
            "xtb_opt": 10.0,
            "uma_preopt": 20.0,
            "uma_ts_opt": 30.0,
            "uma_min_opt": 40.0,
            "dft_preopt": 20.0,
            "dft_ts_opt": 30.0,
            "dft_opt": 40.0,
        }.get(name)
        out[name + "-oc"] = [
            (
                [[point[0] + offset, point[1], point[2]] for point in value]
                if offset is not None
                else [[-999.0, 0.0, 0.0]] * len(value)
            )
            for value in coords
        ]
        out[name + "-NT"] = True
        if name in ["uma_hessian", "dft_hessian"]:
            out[name + "-input.hess"] = ["seed"] * len(df)
        if name in ["uma_freq", "dft_freq"]:
            out[name + "-GE"] = out[name + "-EE"] + 0.01
            out[name + "-vibs"] = [
                [
                    {
                        "frequency": -250.0 if state == "TS1" else 35.0,
                        "mode": [[0.0, 0.0, 0.0]] * len(atoms),
                    }
                ]
                for atoms in df.atoms
            ]
        if kwargs.get("lowest"):
            out = out.sort_values(name + "-EE", kind="stable").head(kwargs["lowest"])
        return out


def run_mocked(wf, tmp_path):
    FakeStepper.calls = []
    with (
        patch.object(MolsWorkflow, "_prepare_initial_df", prepared),
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", prepared),
        patch("frust.workflows.core.Stepper", FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        return wf.run(out_dir=tmp_path / "run", n_cores=1, mem_gb=2)


def test_uma_ranked_geometry_reaches_constrained_preopt_and_unconstrained_validation(
    tmp_path,
):
    wf = workflow()
    stages = wf.show_stages(detail="full")
    ts = stages[stages.branch.eq("transition_states")].set_index("stage")
    assert ts.index.tolist() == [
        "prepare",
        "xtb_preopt",
        "xtb_sp",
        "xtb_sp_filter",
        "xtb_opt",
        "uma_rank_sp",
        "uma_rank_filter",
        "uma_preopt",
        "uma_hessian",
        "uma_ts_opt",
        "uma_freq",
        "uma_solv_sp",
    ]
    assert ts.loc["uma_preopt", "constraint"]
    assert not ts.loc[
        ["uma_hessian", "uma_ts_opt", "uma_freq", "uma_solv_sp"], "constraint"
    ].any()
    assert ts.loc["uma_preopt", "geometry_from"] == "xtb_opt"
    assert ts.loc["uma_solv_sp", "geometry_from"] == "uma_ts_opt"
    run = run_mocked(wf, tmp_path)
    calls = {
        name: (cids, coords, kwargs)
        for state, name, cids, coords, kwargs in FakeStepper.calls
        if state == "TS1"
    }
    assert calls["xtb_preopt"][0] == [0, 1, 2, 3]
    assert calls["xtb_opt"][0] == [0, 1, 2]
    assert calls["uma_preopt"][0] == [
        1
    ]  # Finite ranking winner, not g-xTB winner cid 0.
    assert calls["uma_preopt"][1][0][0][0] == 12.0
    assert calls["uma_preopt"][2]["constraint"]
    assert calls["uma_hessian"][1][0][0][0] == 32.0
    assert calls["uma_ts_opt"][2]["use_last_hess"]
    assert calls["uma_solv_sp"][1][0][0][0] == 62.0
    for state, name, cids, coords, kwargs in FakeStepper.calls:
        if name == "uma_min_opt":
            assert cids == [1] and coords[0][0][0] == 12.0 and not kwargs["constraint"]
        if name == "uma_solv_sp" and state != "TS1":
            assert coords[0][0][0] == 52.0
    ranked = run.states(level="uma_ranked")
    assert set(ranked.cid) == {1}
    assert set(ranked.geometry_stage) == {"xtb_opt"}
    assert set(ranked.energy_stage) == {"uma_rank_sp"}
    assert run.barriers(level="uma_ranked").delta_g_kcal_mol.isna().all()
    assert run.barriers(level="full").delta_g_kcal_mol.notna().all()
    actual = run.barriers().iloc[0]
    assert actual.delta_g_corrected_kcal_mol == pytest.approx(
        actual.delta_g_kcal_mol - 1.89
    )
    assert set(run.states().method_family) == {"uma"}
    assert run.available_analysis_levels() == ("low_cost", "uma_ranked", "full")
    assert len(run.dimer_references()) == 3
    frame = pd.read_parquet(run.path / "calculations/transition_states/merged.parquet")
    assert frame.attrs["frust_geometry_inputs"]["uma_preopt"] == "xtb_opt-oc"
    assert "uma_freq-vibs" in frame
    assert not any(column.startswith("dft_") for column in frame)
    assert (
        frame.attrs["frust_results"]["columns"]["ranking"]["electronic_energy"]
        == "uma_rank_sp-EE"
    )
    states = run.states()
    np.testing.assert_allclose(
        states.free_energy_hartree, states.electronic_energy_hartree + 0.01
    )


@pytest.mark.parametrize(
    "ranking,method,tier",
    [
        ("uma-alpb-chloroform", "wb97xd3-631g", "uma_ranked"),
        ("wb97xd3-631g", "uma-gas-opt-alpb-chloroform", "dft_ranked"),
        ("wb97xd3-631g", "r2scan-3c", "dft_ranked"),
    ],
)
def test_selection_and_validation_are_independent(tmp_path, ranking, method, tier):
    wf = workflow(ranking=ranking, method=method)
    run = run_mocked(wf, tmp_path)
    selected = run.states(level=tier)
    assert set(selected.cid) == {1}
    assert selected.free_energy_hartree.isna().all()
    assert run.barriers().delta_g_kcal_mol.notna().all()
    expected = "uma" if tier == "uma_ranked" else "dft"
    assert set(selected.method_family) == {expected}
    assert run.available_analysis_levels() == ("low_cost", tier, "full")


def test_alias_and_reference_identity_cover_candidate_policy():
    current = workflow(ranking_top_n=2)
    legacy = workflow(ranking_top_n=None, uma_rank_top_n=2)
    assert current.plan().equals(legacy.plan())
    assert current.method.fingerprint() == legacy.method.fingerprint()
    with pytest.raises(ValueError, match="disagree"):
        workflow(ranking_top_n=2, uma_rank_top_n=1)
    with pytest.raises(ValueError, match="positive"):
        workflow(ranking_top_n=0)
    target = current.children()["references"].targets()[0]

    def identity(wf, level):
        return reference_identity(
            target,
            wf.method,
            protocol=wf._reference_protocol(level),
            calculation_level=level,
        )[0]

    assert identity(current, "full") != identity(workflow(ranking_top_n=1), "full")
    assert identity(current, "low_cost") == identity(
        workflow(ranking_top_n=1), "low_cost"
    )
    assert identity(current, "full") != identity(
        workflow(ranking="uma-gas", ranking_top_n=2), "full"
    )
    other_validation = workflow(method="wb97xd3-631g", ranking_top_n=2)
    assert identity(current, "low_cost") == identity(other_validation, "low_cost")
    assert identity(current, "uma_ranked") == identity(other_validation, "uma_ranked")
    assert identity(current, "full") != identity(other_validation, "full")


@pytest.mark.parametrize("method", ["uma-gas-opt-alpb-chloroform", "wb97xd3-631g"])
def test_all_selected_candidates_receive_validation_and_remain_reviewable(
    tmp_path, method
):
    run = run_mocked(workflow(method=method, ranking_top_n=2), tmp_path)
    for state, name, cids, coords, kwargs in FakeStepper.calls:
        if name in {
            "uma_preopt",
            "uma_min_opt",
            "uma_freq",
            "dft_preopt",
            "dft_opt",
            "dft_freq",
        }:
            assert set(cids) == {0, 1}
    candidates = run.candidate_barriers()
    assert len(candidates) == 2 and candidates.selected.sum() == 1
    assert run.barriers().delta_g_kcal_mol.notna().all()


def test_native_arrays_keep_the_validation_chain_without_preparation(tmp_path):
    from types import SimpleNamespace
    from frust.screen.runs import ScreenRun

    class FakeExecutor:
        def __init__(self):
            self.submissions = []
            self.parameters = []
            self.array_count = 0

        def update_parameters(self, **kwargs):
            self.parameters.append(kwargs)

        def map_array(self, function, *columns):
            arguments = list(zip(*columns))
            self.submissions.extend((function, args, {}) for args in arguments)
            self.array_count += 1
            return [
                SimpleNamespace(job_id=f"{800+self.array_count}_{index}")
                for index in range(len(arguments))
            ]

        def submit(self, function, *args, **kwargs):
            self.submissions.append((function, args, kwargs))
            return SimpleNamespace(job_id="mock-job")

    fake = FakeExecutor()
    wf = workflow()
    cluster = ft.cluster.ClusterConfig(
        backend="slurm", partition="chem", log_dir=tmp_path / "logs"
    )
    with (
        patch("frust.workflows.core.create_executor", return_value=fake),
        patch("frust.workflows.screening.create_executor", return_value=fake),
    ):
        with (
            patch.object(ScreenTSWorkflow, "_prepare_initial_df") as ts_prepare,
            patch.object(MolsWorkflow, "_prepare_initial_df") as ref_prepare,
        ):
            submitted = wf.submit(
                out_dir=tmp_path / "submitted",
                cluster=cluster,
                execution="single_job",
                array=True,
                array_parallelism=4,
            )
        ts_prepare.assert_not_called()
        ref_prepare.assert_not_called()
    assert len(submitted.child_submissions) == 2 and fake.array_count == 2
    chains = [
        [stage.id for stage in args[0]._stage_defs()]
        for function, args, _ in fake.submissions
        if function.__name__ == "_run_target_submitted_job"
    ]
    assert any(
        parameters.get("slurm_array_parallelism") == 4 for parameters in fake.parameters
    )
    assert any(
        "uma_rank_sp" in chain and "uma_preopt" in chain and "uma_ts_opt" in chain
        for chain in chains
    )
    assert any("uma_rank_sp" in chain and "uma_min_opt" in chain for chain in chains)
    manifest = ScreenRun(tmp_path / "submitted").manifest
    assert manifest["ranking_top_n"] == 1
    assert manifest["analysis_levels"] == ["low_cost", "uma_ranked", "full"]
    assert manifest["reference_store_configured"] is False


def test_uma_validation_keeps_a_consistent_gas_potential():
    method = ft.workflows.methods.preset("uma-gas-opt-alpb-chloroform")
    mixed = method.with_stage(
        "uma_freq", ft.workflows.methods.uma(job="freq", xtb_alpb="chloroform")
    )
    with pytest.raises(ValueError, match="one model and environment"):
        workflow(method=mixed).show_stages()
    for options in [{"Opt": None}, {"OptTS": None}, {"NumFreq": None}]:
        with pytest.raises(ValueError, match="single-point"):
            ft.workflows.RankingPlan(
                name="bad",
                stage_id="dft_rank_sp",
                calculator=replace(
                    ft.workflows.methods.orca(method="PBE", job="sp"), options=options
                ),
            )


def test_full_cycle_selection_uses_the_same_limit_for_int3():
    wf = workflow(method="wb97xd3-631g", scope="full_cycle", ranking_top_n=2)
    stages = wf.children()["int3"]._stage_defs()
    selection = next(stage for stage in stages if stage.id == "uma_rank_filter")
    assert selection.lowest == 2
    assert all(
        stage.lowest is None
        for stage in stages
        if stage.id in {"dft_preopt", "dft_opt"}
    )


def test_p13_parent_coverage_is_lightweight_and_contains_no_dft_stages(monkeypatch):
    monkeypatch.delenv("FRUST_REFERENCE_STORE", raising=False)
    chemistry = pd.DataFrame(
        {
            "role": ["substrate"] * 2 + ["catalyst"] * 4,
            "smiles": [
                "COC1=CC=CC(OC)=C1",
                "CC1=CC=CC(N(C)C)=C1",
                "BC1=C(N(C)C)C=CC=C1",
                "BC1=C(N(CC)CC)C=CC=C1",
                "BC1=C(N2CCCCC2)C=CC=C1",
                "BC1=C(N2C(C)(C)CCCC2(C)C)C=CC=C1",
            ],
            "rpos": ["3;", "2;"] + [None] * 4,
        }
    )
    wf = workflow(
        dataframe=chemistry,
        ts_types=["TS1", "TS2", "TS3", "TS4"],
        n_confs=200,
        top_n=20,
    )
    with (
        patch.object(ScreenTSWorkflow, "_prepare_initial_df") as ts_prepare,
        patch.object(MolsWorkflow, "_prepare_initial_df") as ref_prepare,
    ):
        plan = wf.plan()
        stages = wf.show_stages(detail="full")
    ts_prepare.assert_not_called()
    ref_prepare.assert_not_called()
    assert len(wf.systems()) == 8
    assert plan.branch.value_counts().to_dict() == {
        "transition_states": 32,
        "references": 16,
    }
    assert plan.action.eq("calculate").all() and plan.reference_id.isna().all()
    assert not stages.stage.str.startswith("dft_").any()
    assert not any(stage.startswith("dft_") for stage in wf.method.stages)
    assert wf.n_confs == 200 and wf.top_n == 20 and wf.ranking_top_n == 1


def test_raw_molecule_validation_uses_explicit_ranking_without_dft_flags(tmp_path):
    from frust.workflows.factories import RawMolsWorkflow

    methods = ft.workflows.methods
    method = methods.apply_screening_plan(
        methods.preset("uma-gas-opt-alpb-chloroform"),
        methods.screening_preset("gxtb-default"),
    ).with_ranking(methods.ranking_preset("uma-alpb-chloroform"))
    wf = ft.workflows.raw_mols(
        smiles=["CN1C=CC=C1"], method=method, calculation_level="full", top_n=3
    )
    target = workflow().children()["references"].targets()[0]
    frame = prepared(None, target, save_dir=None, options=None)
    with (
        patch.object(RawMolsWorkflow, "_prepare_initial_df", return_value=frame),
        patch("frust.workflows.core.Stepper", FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        result = wf.run(out_dir=tmp_path)
    assert result.cid.tolist() == [1]
    assert (
        result.attrs["frust_results"]["columns"]["ranking"]["electronic_energy"]
        == "uma_rank_sp-EE"
    )
    assert result.attrs["frust_workflow"]["ranking_top_n"] == 1
    assert not any(column.startswith("dft_") for column in result)
