"""Regressions for chemical identity, candidate counts and exact connections."""

import hashlib
import pandas as pd
import pytest
import frust as ft
from frust.structures.planner import plan_targets
from frust.workflows.bisarylation import _system_signature
from frust.workflows.bisarylation_analysis import (
    _analysis_case,
    _barrier_rows,
    _review_roles,
)
from test_bisarylation_barriers import fixture

SYSTEM = dict(
    system_name="tmp",
    substrate_name="pyrrole",
    catalyst_name="tmp",
    catalyst_smiles="CN(C)c1ccccc1B",
    substrate_smiles="CN1C=CC=C1",
    rpos=2,
)


def test_bound_site_is_part_of_signature_and_target_identity():
    a, b = ({**SYSTEM, "bound_rpos": s} for s in (2, 3))
    assert _system_signature(a, 2) != _system_signature(b, 2)
    assert (
        plan_targets(pd.DataFrame([a]), states=["TS5"])[0].target_id
        != plan_targets(pd.DataFrame([b]), states=["TS5"])[0].target_id
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"bound_rpos": 2},
        {"bound_substrate_smiles": SYSTEM["substrate_smiles"]},
        {"bound_substrate_smiles": pd.NA, "bound_rpos": pd.NA},
    ],
)
def test_equivalent_bound_inputs_share_normalized_identity(extra):
    explicit = {
        **SYSTEM,
        "bound_substrate_smiles": SYSTEM["substrate_smiles"],
        "bound_rpos": 2,
    }
    assert _system_signature({**SYSTEM, **extra}, 2) == _system_signature(explicit, 2)
    assert (
        plan_targets(pd.DataFrame([{**SYSTEM, **extra}]), states=["TS5"])[0].target_id
        == plan_targets(pd.DataFrame([explicit]), states=["TS5"])[0].target_id
    )


@pytest.mark.parametrize(
    "extra", [{"bound_rpos": 999}, {"bound_substrate_smiles": "c1ccoc1"}]
)
def test_invalid_bound_role_fails_before_embedding(extra):
    with pytest.raises(ValueError):
        plan_targets(pd.DataFrame([{**SYSTEM, **extra}]), states=["TS5"])


@pytest.mark.parametrize("ranking", [None, "uma-gas"])
@pytest.mark.parametrize("alias", ["validation_top_n", "ranking_top_n"])
def test_count_alias_is_resolved_without_requiring_ranking(ranking, alias):
    wf = ft.workflows.bisarylation_screen(
        systems=pd.DataFrame([SYSTEM]), ranking=ranking, **{alias: 3}
    )
    assert {c.validation_top_n for c in wf.children().values()} == {3}


def test_legacy_ts6_override_without_exact_saddle_does_not_approve_replacement(
    tmp_path,
):
    states, case = fixture()
    proof = tmp_path / "analysis/pathway_evidence/proof.json"
    proof.parent.mkdir(parents=True)
    proof.write_text("{}")
    record = dict(
        system_name=case["system_name"],
        rpos=2,
        system_signature=case["system_signature"],
        method_fingerprint=case["method_fingerprint"],
        variant="two_saddle",
        result_ids={r: r + "-result" for r in _review_roles("two_saddle")},
        minimum_result_ids={"TS6": {"INT4": "INT4-result"}},
        evidence=[
            dict(
                path=str(proof.relative_to(tmp_path)),
                sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
            )
        ],
    )
    states.loc[states.state_id.eq("TS6"), "result_id"] = "new-TS6"
    analyzed = _analysis_case(states, case, tmp_path, [record])
    rows = _barrier_rows(states, analyzed, quantity="electronic").set_index("ts_type")
    assert rows.loc["TS6", "quality_status"] != "ready"
    assert rows.loc["TS7", "quality_status"] == "ready"


def test_fractional_bound_site_is_rejected():
    with pytest.raises(ValueError):
        plan_targets(pd.DataFrame([{**SYSTEM, "bound_rpos": 2.5}]), states=["TS5"])


@pytest.mark.parametrize("alias", ["validation_top_n", "ranking_top_n"])
def test_alias_normalizes_before_changed_uma_environment_check(alias):
    wf = ft.workflows.catalyst_screen(
        systems=pd.DataFrame([SYSTEM]),
        method="uma-alpb-chloroform",
        screening="uma-gas",
        **{alias: 3},
    )
    assert {child.validation_top_n for child in wf.children().values()} == {3}
