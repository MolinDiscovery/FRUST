"""Semantic result contracts for planned UMA method variants."""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

import frust as ft
from frust.results import (
    attach_result_contract,
    free_energy_components,
    result_column,
    result_contract,
)
from frust.workflows.core import ANALYSIS_TIER_FILES


@pytest.mark.parametrize(
    "profile, geometry_stage",
    [
        ("transition_state", "uma_ts_opt"),
        ("minimum", "uma_min_opt"),
        ("constrained_minimum", "uma_min_opt"),
    ],
)
def test_full_uma_contract_uses_uma_geometry_and_frequency(profile, geometry_stage):
    contract = result_contract(
        profile,
        dft=False,
        calculation_level="full",
        full_method="uma",
        screening_opt_stage="uma_opt",
        include_dft_rank_sp=False,
        thermochemistry=ft.workflows.ThermochemistrySpec(mode="frequency_gibbs"),
    )

    assert contract["dft"] is False
    assert contract["columns"]["analysis"]["electronic_energy"] == "uma_freq-EE"
    assert contract["columns"]["optimized"]["coords"] == f"{geometry_stage}-oc"
    assert contract["columns"]["frequency"]["gibbs_energy"] == "uma_freq-GE"
    assert contract["columns"]["frequency"]["frequencies"] == "uma_freq-frequencies_cm1"
    assert contract["thermochemistry"]["mode"] == "frequency_gibbs"


def test_uma_ranked_tier_uses_single_point_energy_and_gxtb_geometry():
    df = pd.DataFrame({"uma_rank_sp-EE": [-10.0], "xtb_opt-oc": [[(0, 0, 0)]]})
    attach_result_contract(
        df,
        "transition_state",
        dft=False,
        calculation_level="uma_ranked",
        screening_opt_stage="xtb_opt",
    )

    assert result_column(df) == "uma_rank_sp-EE"
    assert result_column(df, "coords", purpose="optimized") == "xtb_opt-oc"
    assert (
        df.attrs["frust_results"]["columns"]["ranking"]["electronic_energy"]
        == "uma_rank_sp-EE"
    )
    assert ANALYSIS_TIER_FILES["uma_ranked"] == "tier_uma_ranked.parquet"


def test_full_hybrid_contract_identifies_uma_ranking_and_dft_result():
    contract = result_contract(
        "transition_state",
        dft=True,
        calculation_level="full",
        screening_opt_stage="xtb_opt",
        include_dft_rank_sp=False,
        ranking_stage="uma_rank_sp",
    )

    assert contract["columns"]["ranking"]["electronic_energy"] == "uma_rank_sp-EE"
    assert contract["columns"]["analysis"]["electronic_energy"] == "dft_solv_sp-EE"
    assert contract["columns"]["optimized"]["coords"] == "dft_ts_opt-oc"


def test_full_uma_frequency_thermochemistry_uses_uma_values():
    df = pd.DataFrame({"uma_freq-EE": [-20.0], "uma_freq-GE": [-19.9]})
    attach_result_contract(
        df,
        "transition_state",
        dft=False,
        calculation_level="full",
        full_method="uma",
        thermochemistry=ft.workflows.ThermochemistrySpec(mode="frequency_gibbs"),
    )

    components = free_energy_components(df)
    assert components.loc[0, "free_energy_hartree"] == pytest.approx(-19.9)


def test_existing_dft_contract_and_method_fingerprint_shape_remain_unchanged():
    contract = result_contract("transition_state", dft=True)
    method = ft.workflows.methods.preset("wb97xd3-631g")

    assert contract["schema_version"] == 5
    assert contract["columns"]["analysis"]["electronic_energy"] == "dft_solv_sp-EE"
    assert contract["columns"]["frequency"]["frequencies"] == "dft_freq-frequencies_cm1"
    assert "result_family" not in method.to_dict()


def test_uma_method_family_is_serialized_without_changing_dft_identity():
    base = ft.workflows.methods.preset("wb97xd3-631g")
    uma_plan = ft.workflows.MethodPlan(
        name="uma-contract-example",
        stages={"uma_freq": ft.workflows.methods.uma(job="sp")},
        result_family="uma",
    )

    assert base.result_family == "dft"
    assert "result_family" not in base.to_dict()
    assert uma_plan.to_dict()["result_family"] == "uma"
    assert uma_plan.fingerprint() != base.fingerprint()


@pytest.mark.parametrize(
    "environment, solvent",
    [("uma-gas", None), ("uma-alpb-chloroform", "chloroform")],
)
def test_ranking_presets_have_independent_identity(environment, solvent):
    ranking = ft.workflows.methods.ranking_preset(environment)
    screening = ft.workflows.methods.screening_preset(environment)

    assert ranking.stage_id == "uma_rank_sp"
    assert ranking.calculator.kwargs["uma"] == "omol@uma-s-1p2p1"
    assert ranking.calculator.solvent == solvent
    assert ranking.fingerprint() != screening.fingerprint()
    assert pickle.loads(pickle.dumps(ranking)).fingerprint() == ranking.fingerprint()
    if solvent:
        assert ranking.to_dict()["calculator"]["kwargs"]["uma_xtb_alpb"] == solvent
    else:
        assert "uma_xtb_alpb" not in ranking.calculator.kwargs


def test_tier_and_family_mismatches_are_rejected():
    with pytest.raises(ValueError, match="UMA ranking"):
        result_contract(
            "transition_state",
            dft=False,
            calculation_level="uma_ranked",
            ranking_stage="dft_rank_sp",
        )
    with pytest.raises(ValueError, match="full_method"):
        result_contract(
            "transition_state", dft=False, calculation_level="full", full_method="other"
        )
    with pytest.raises(ValueError, match="include_dft_rank_sp=False"):
        result_contract(
            "transition_state",
            dft=True,
            calculation_level="full",
            ranking_stage="uma_rank_sp",
        )
    with pytest.raises(ValueError, match="single-point"):
        ft.workflows.RankingPlan(
            name="wrong-job",
            stage_id="uma_rank_sp",
            calculator=ft.workflows.methods.uma(job="opt"),
        )
