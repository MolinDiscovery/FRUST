"""Connection evidence is exact, portable and independently invalidated."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
import shutil
import pandas as pd
import pytest
from frust.workflows.reactions import NETWORKS
from frust.workflows.reaction_analysis import reaction_tables
from frust.workflows.connections import save_connection
from test_bisarylation_barriers import fixture


def setup_run(tmp_path):
    states, case = fixture()
    definition = NETWORKS["bisarylation"]
    case.update(
        network_id="bisarylation",
        definition_version=1,
        reactions=[
            {**asdict(r), "expressions": r.expressions("full_cycle")}
            for r in definition.reactions
        ],
        profile_terms={},
        reference="M",
    )
    manifest = dict(
        calculation_level="full",
        reaction_plan=dict(cases=[case]),
        g_corrections_kcal_mol={},
    )
    run = SimpleNamespace(
        manifest=manifest,
        path=tmp_path,
        analysis_dir=tmp_path / "analysis",
        states=lambda: states,
        refresh_analysis=lambda: None,
    )
    return run, states, case


@pytest.mark.parametrize("ts", ["TS5", "TS6", "TS7", "TS8"])
def test_exact_connection_replacement_revocation_and_portable_proof(tmp_path, ts):
    run, states, case = setup_run(tmp_path)
    reaction = next(r for r in case["reactions"] if r["ts_state"] == ts)
    roles = set().union(*(set(e) for e in reaction["expressions"].values())) - {ts}
    ids = {role: role + "-result" for role in roles}
    proof = tmp_path / f"{ts}-proof.json"
    proof.write_text('{"reviewed":true}')

    def record(decision="approved"):
        save_connection(
            run,
            system_name=case["system_name"],
            rpos=2,
            reaction_id=reaction["reaction_id"],
            saddle_result_id=ts + "-result",
            reference_result_ids=ids,
            endpoint_result_ids=ids,
            direction_sides={"forward": "reactants", "backward": "products"},
            evidence_files=[proof],
            decision=decision,
        )

    def rows(frame=states, root=tmp_path, candidates=False):
        return reaction_tables(
            root,
            frame,
            run.manifest,
            pd.DataFrame(),
            calculation_level="full",
            candidates=candidates,
        )[0]

    if ts == 'TS6':
        neighbor=states.loc[states.state_id.eq('INT4')].iloc[0].copy()
        neighbor.result_id='INT4-reviewed-neighbor'
        neighbor.electronic_energy_hartree+=.03
        neighbor.free_energy_hartree+=.03
        states.loc[len(states)] = neighbor
        ids['INT4']=neighbor.result_id
    record()
    original = rows().set_index("ts_type")
    assert original.loc[ts, "connection_status"] == "approved"
    assert original.loc[ts, "quality_status"] == "ready"
    alternative = states.loc[states.state_id.eq(ts)].iloc[0].copy()
    alternative.result_id = ts + "-other"
    alternative.electronic_energy_hartree -= 0.01
    both = pd.concat([states, pd.DataFrame([alternative])], ignore_index=True)
    candidate = (
        rows(both, candidates=True)
        .loc[lambda df: df.ts_result_id.eq(alternative.result_id)]
        .iloc[0]
    )
    assert candidate.connection_status == "unreviewed_candidate"
    if ts == "TS6":
        assert original.loc[ts,"uses_alternate_minimum"]
        assert not candidate.uses_alternate_minimum
    assert candidate.quality_status != "ready" and pd.isna(candidate.delta_e_kcal_mol)
    changed = states.copy()
    changed.loc[changed.state_id.eq(ts), "result_id"] = "replacement"
    altered = rows(changed).set_index("ts_type")
    assert altered.loc[ts, "quality_status"] != "ready"
    # TS5/6 are independent of exchange approval; the other keeps its diagnostic behavior.
    unaffected = "TS5" if ts != "TS5" else "TS6"
    assert (
        altered.loc[unaffected, "quality_status"]
        == original.loc[unaffected, "quality_status"]
    )
    relocated = tmp_path / "moved"
    relocated.mkdir()
    shutil.copytree(tmp_path / "analysis", relocated / "analysis")
    assert (
        rows(root=relocated).set_index("ts_type").loc[ts, "connection_status"]
        == "approved"
    )
    copied = next((relocated / "analysis/pathway_evidence").iterdir())
    copied.write_text("changed")
    assert (
        rows(root=relocated).set_index("ts_type").loc[ts, "connection_status"]
        == "changed_evidence"
    )
    record("rejected")
    assert rows().set_index("ts_type").loc[ts, "quality_status"] != "ready"


@pytest.mark.parametrize("mutation", ["result", "protocol", "quality", "missing_proof"])
def test_endpoint_and_reference_changes_invalidate_only_their_connection(
    tmp_path, mutation
):
    run, states, case = setup_run(tmp_path)
    reaction = case["reactions"][1]
    proof = tmp_path / "proof"
    proof.write_text("review")
    ids = {r: r + "-result" for r in ["INT4", "INT5", "HH"]}
    save_connection(
        run,
        system_name=case["system_name"],
        rpos=2,
        reaction_id=reaction["reaction_id"],
        saddle_result_id="TS6-result",
        reference_result_ids=ids,
        endpoint_result_ids=ids,
        direction_sides={"backward": "reactants", "forward": "products"},
        evidence_files=[proof],
    )
    if mutation == "result":
        states.loc[states.state_id.eq("INT4"), "result_id"] = "new-endpoint"
    elif mutation == "protocol":
        states.loc[states.state_id.eq("INT4"), "energy_protocol_fingerprint"] = (
            "other-surface"
        )
    elif mutation == "quality":
        states.loc[states.state_id.eq("INT4"), "quality_status"] = "invalid"
    else:
        next((tmp_path / "analysis/pathway_evidence").iterdir()).unlink()
    rows = reaction_tables(
        tmp_path, states, run.manifest, pd.DataFrame(), calculation_level="full"
    )[0].set_index("ts_type")
    assert rows.loc["TS6", "quality_status"] != "ready"
    assert pd.isna(rows.loc["TS6", "delta_e_kcal_mol"])


def test_missing_reverse_fragment_does_not_revoke_forward_connection(tmp_path):
    run, states, case = setup_run(tmp_path)
    proof = tmp_path / "proof"
    proof.write_text("review")
    save_connection(
        run,
        system_name=case["system_name"],
        rpos=2,
        reaction_id="bisarylation.ts6",
        saddle_result_id="TS6-result",
        reference_result_ids={r: r + "-result" for r in ["INT4", "INT5", "HH"]},
        endpoint_result_ids={r: r + "-result" for r in ["INT4", "INT5"]},
        direction_sides={"forward": "reactants", "backward": "products"},
        evidence_files=[proof],
    )
    without_h2 = states.loc[~states.state_id.eq("HH")]
    rows = reaction_tables(
        tmp_path, without_h2, run.manifest, pd.DataFrame(), calculation_level="full"
    )[0].set_index("ts_type")
    assert rows.loc["TS6", "quality_status"] == "ready"
    assert rows.loc["TS6", "connection_status"] == "approved"
    assert rows.loc["TS6", "reverse_quality_status"] != "ready"
    assert pd.isna(rows.loc["TS6", "reverse_delta_e_kcal_mol"])


def test_connection_requires_minimum_endpoints_on_both_sides(tmp_path):
    run, states, case = setup_run(tmp_path)
    proof = tmp_path / "proof"
    proof.write_text("review")
    with pytest.raises(ValueError, match="both chemical sides"):
        save_connection(
            run,
            system_name=case["system_name"],
            rpos=2,
            reaction_id="bisarylation.ts6",
            saddle_result_id="TS6-result",
            reference_result_ids={"INT4": "INT4-result"},
            endpoint_result_ids={"INT4": "INT4-result"},
            direction_sides={"forward": "reactants", "backward": "products"},
            evidence_files=[proof],
        )


def test_declared_node_gibbs_corrections_are_applied_once_to_each_quantity(tmp_path):
    run, states, case = setup_run(tmp_path)
    run.manifest["g_corrections_kcal_mol"] = {
        "INT4": 1.2,
        "TS6": 0.4,
        "INT5": -0.2,
        "M": 2.1,
    }
    rows = reaction_tables(
        tmp_path, states, run.manifest, pd.DataFrame(), calculation_level="full"
    )[0].set_index("ts_type")
    assert rows.loc["TS5", "delta_g_corrected_kcal_mol"] - rows.loc[
        "TS5", "delta_g_kcal_mol"
    ] == pytest.approx(-2.1)
    assert rows.loc["TS6", "delta_g_corrected_kcal_mol"] - rows.loc[
        "TS6", "delta_g_kcal_mol"
    ] == pytest.approx(-0.8)
    assert rows.loc["TS6", "reverse_delta_g_corrected_kcal_mol"] - rows.loc[
        "TS6", "reverse_delta_g_kcal_mol"
    ] == pytest.approx(0.6)
    assert rows.loc["TS6", "reaction_delta_g_corrected_kcal_mol"] - rows.loc[
        "TS6", "reaction_delta_g_kcal_mol"
    ] == pytest.approx(-1.4)
    run.manifest["scope"] = "full_cycle"
    _, profile = reaction_tables(
        tmp_path, states, run.manifest, pd.DataFrame(), calculation_level="full"
    )
    ts = profile.loc[profile.profile_state.eq("TS6")].iloc[0]
    assert ts.relative_g_corrected_kcal_mol - ts.relative_g_kcal_mol == pytest.approx(
        -0.8
    )


def test_common_analysis_preserves_neighbor_and_dimer_inspection_fields(tmp_path):
    run, states, case = setup_run(tmp_path)
    case["profile_terms"] = NETWORKS["bisarylation"].profile_terms
    case["profile_order"] = list(case["profile_terms"])
    rows, _ = reaction_tables(
        tmp_path, states, run.manifest, pd.DataFrame(), calculation_level="full"
    )
    ts6 = rows.set_index("ts_type").loc["TS6"]
    assert pd.notna(ts6.dimer_relative_g_kcal_mol)
    energies = states.set_index("state_id").free_energy_hartree
    expected = (
        energies["TS6"]
        + energies["HH"]
        - 0.5 * energies["dimer"]
        - energies["incoming_ligand"]
        - energies["bound_ligand"]
    ) * 627.5094740631
    assert ts6.dimer_relative_g_kcal_mol == pytest.approx(expected)
    assert ts6.dimer_elevation_quality_status == "ready"
    assert not ts6.uses_alternate_minimum
    missing_h2 = states.loc[~states.state_id.eq("HH")]
    rows, _ = reaction_tables(
        tmp_path, missing_h2, run.manifest, pd.DataFrame(), calculation_level="full"
    )
    assert (
        "missing:HH" in rows.set_index("ts_type").loc["TS6", "reverse_quality_issues"]
    )
