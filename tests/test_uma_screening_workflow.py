"""UMA screening stage graphs, provenance, and result-tier behavior."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import pandas as pd
import pytest

import frust as ft
from frust.results import result_column
from frust.screen.references import reference_identity
from frust.workflows.factories import ScreenTSWorkflow


def _components() -> pd.DataFrame:
    """Return a small substrate/catalyst input for workflow planning."""
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", "BC1=C(N(C)C)C=CC=C1"],
            "compound_name": ["pyrrole", "NMe"],
            "rpos": ["2", None],
        }
    )


@pytest.mark.parametrize(
    "screening, solvent", [("uma-gas", None), ("uma-alpb-chloroform", "chloroform")]
)
def test_uma_screening_presets_use_explicit_potential_and_stage_names(
    screening, solvent
):
    plan = ft.workflows.methods.screening_preset(screening)

    assert set(plan.stages) == {"xtb_preopt", "uma_sp", "uma_opt"}
    assert plan.stages["xtb_preopt"].options == {"gfnff": None, "opt": None}
    assert plan.stages["uma_sp"].kwargs["uma"] == "omol@uma-s-1p2p1"
    assert plan.stages["uma_sp"].solvent == solvent
    assert plan.stages["uma_opt"].solvation_model == ("alpb" if solvent else None)
    assert "SMD" not in plan.stages["uma_opt"].xtra_inp_str
    if solvent:
        assert plan.stages["uma_sp"].kwargs["uma_xtb_alpb"] == "chloroform"
        assert plan.stages["uma_opt"].kwargs["uma_xtb_alpb"] == "chloroform"


def test_full_uma_screen_uses_one_stage_plan_for_ts_references_and_int3():
    workflow = ft.workflows.catalyst_screen(
        dataframe=_components(),
        screening="uma-alpb-chloroform",
        method="wb97xd3-631g",
        level="full",
        scope="full_cycle",
        ts_types=["TS3"],
        spec_profile="omol-uma-s-1p2p1/gas",
        spec_match="exact",
        n_confs=1,
    )

    assert workflow._analysis_levels() == ("low_cost", "full")
    for child in workflow.children().values():
        stages = child.show_stages()
        ids = stages["stage"].tolist()
        assert ids.index("xtb_preopt") < ids.index("uma_sp")
        assert ids.index("uma_sp") < ids.index("uma_sp_filter") < ids.index("uma_opt")
        assert "xtb_sp" not in ids and "xtb_opt" not in ids
        assert "dft_rank_sp" not in ids
        assert stages.set_index("stage").loc["uma_sp", "solvent"] == "ALPB(chloroform)"
    ts = workflow.children()["transition_states"]
    assert ts.resolved_spec_profile == "omol-uma-s-1p2p1/gas"
    assert ts.targets()
    init_group = [stage.id for stage in ts._stage_groups("dft_staged")[0]]
    assert init_group[-4:] == ["uma_sp", "uma_sp_filter", "uma_opt", "dft_preopt"]
    assert workflow.children()["int3"].resolved_spec_profile == "omol-uma-s-1p2p1/gas"
    assert workflow.children()["int3"].targets()

    reference_target = workflow.children()["references"].targets()[0]
    _, identity = reference_identity(
        reference_target,
        workflow.method,
        protocol=workflow._reference_protocol("full"),
    )
    assert "uma_sp" in identity["active_method"]["stages"]
    assert "uma_opt" in identity["active_method"]["stages"]
    assert "dft_rank_sp" not in identity["active_method"]["stages"]


def test_uma_ranking_sp_switch_preserves_dft_ranked_level():
    base = dict(dataframe=_components(), screening="uma-gas", ts_types=["TS1"])
    ranked = ft.workflows.catalyst_screen(**base, level="dft_ranked")
    full = ft.workflows.catalyst_screen(**base, level="full", include_dft_rank_sp=True)

    assert (
        "dft_rank_sp"
        in ranked.children()["transition_states"].show_stages()["stage"].tolist()
    )
    assert (
        "dft_rank_sp"
        in full.children()["transition_states"].show_stages()["stage"].tolist()
    )
    assert (
        "dft_rank_filter"
        in full.children()["transition_states"].show_stages()["stage"].tolist()
    )
    assert full._analysis_levels() == ("low_cost", "dft_ranked", "full")
    assert (
        ranked.children()["transition_states"].resolved_spec_profile
        == "wb97xd3-631g/gas"
    )


class _FakeStepper:
    """Return one synthetic row per calculator call without external services."""

    def __init__(self, **kwargs):
        del kwargs

    def xtb(self, df, *, name, **kwargs):
        del kwargs
        return self._calculate(df, name)

    def orca(self, df, *, name, **kwargs):
        del kwargs
        return self._calculate(df, name)

    @staticmethod
    def _calculate(df, name):
        result = df.copy()
        result[f"{name}-EE"] = (
            result["cid"].map({0: -1.0, 1: -3.0, 2: -2.0}) if name == "uma_sp" else -1.0
        )
        result[f"{name}-oc"] = [[(0.0, 0.0, 0.0)] for _ in range(len(result))]
        result[f"{name}-NT"] = True
        return result


def _prepared_row(self, target, *, save_dir, options):
    """Supply one structure so the test exercises workflow dispatch only."""
    del self, target, save_dir, options
    return pd.DataFrame(
        {
            "structure_id": ["TS1:sample:r2"],
            "state_id": ["TS1"],
            "state_kind": ["transition_state"],
            "system_name": ["sample"],
            "substrate_name": ["pyrrole"],
            "catalyst_name": ["NMe"],
            "rpos": [2],
            "cid": [0],
            "atoms": [["H"]],
            "coords_embedded": [[(0.0, 0.0, 0.0)]],
        }
    )


def test_uma_full_run_writes_uma_low_cost_tier_and_skips_ranking_sp(tmp_path):
    workflow = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS1"],
        screening="uma-gas",
        calculation_level="full",
        prune_initial=False,
        n_confs=1,
    )
    assert isinstance(workflow, ScreenTSWorkflow)

    with (
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", _prepared_row),
        patch("frust.workflows.core.Stepper", _FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        result = workflow.run(targets=[0], out_dir=tmp_path, target_retention="all")

    assert "uma_sp-EE" in result and "uma_opt-EE" in result
    assert "dft_rank_sp-EE" not in result
    assert result_column(result, purpose="ranking") == "uma_opt-EE"
    assert result_column(result, purpose="analysis") == "dft_solv_sp-EE"
    assert result.attrs["frust_workflow"]["guess_profile"] == "wb97xd3-631g/gas"

    snapshot = pd.read_parquet(next(tmp_path.glob("*/tier_low_cost.parquet")))
    assert result_column(snapshot, purpose="analysis") == "uma_opt-EE"
    assert result_column(snapshot, "coords", purpose="optimized") == "uma_opt-oc"
    assert not list(tmp_path.glob("*/tier_dft_ranked.parquet"))


def test_uma_single_point_filters_before_constrained_optimization():
    workflow = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS1"],
        screening="uma-gas",
        calculation_level="low_cost",
        top_n=1,
        prune_initial=False,
    )

    def prepare_three(self, target, *, save_dir, options):
        del self, target, save_dir, options
        return pd.concat(
            [
                _prepared_row(None, None, save_dir=None, options=None).assign(cid=cid)
                for cid in range(3)
            ],
            ignore_index=True,
        )

    with (
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", prepare_three),
        patch("frust.workflows.core.Stepper", _FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        result = workflow.run(targets=[0])

    assert len(result) == 1
    assert result.iloc[0]["cid"] == 1
    assert "uma_opt-EE" in result


def test_uma_full_ranking_switch_writes_dft_ranked_tier(tmp_path):
    workflow = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS1"],
        screening="uma-gas",
        calculation_level="full",
        include_dft_rank_sp=True,
        prune_initial=False,
        n_confs=1,
    )

    with (
        patch.object(ScreenTSWorkflow, "_prepare_initial_df", _prepared_row),
        patch("frust.workflows.core.Stepper", _FakeStepper),
        patch("frust.workflows.core._uma_scope_for_stages", return_value=nullcontext()),
    ):
        result = workflow.run(targets=[0], out_dir=tmp_path, target_retention="all")

    assert result_column(result, purpose="ranking") == "dft_rank_sp-EE"
    ranked = pd.read_parquet(next(tmp_path.glob("*/tier_dft_ranked.parquet")))
    assert result_column(ranked, purpose="analysis") == "dft_rank_sp-EE"
    assert result_column(ranked, "coords", purpose="optimized") == "uma_opt-oc"
