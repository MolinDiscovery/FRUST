"""Generic planning, native dispatch and expression contracts across chemistry."""

from dataclasses import replace
import json
import pandas as pd
import pytest
import frust as ft
from frust.workflows.reaction_planner import plan_reactions
from frust.workflows.reaction_analysis import reaction_tables
from frust.workflows.reactions import (
    ReactionDefinition,
    NetworkDefinition,
    NETWORKS,
    register_network,
)
from frust.workflows.expressions import evaluate_expression
from test_reaction_contracts import SYSTEM


def plan(ts_types, scope="full_cycle"):
    return plan_reactions(
        pd.DataFrame([SYSTEM]),
        ts_types=ts_types,
        scope=scope,
        method_fingerprint="method",
    )


def test_ts6_contract_exact_dependencies_and_coefficients():
    forward = plan(["TS6"], "barriers")
    full = plan(["TS6"])
    assert {t.state_id for t in forward["targets"]} == {"TS6", "INT4"}
    assert {t.state_id for t in full["targets"]} == {"TS6", "INT4", "INT5", "HH"}
    expressions = full["cases"][0]["reactions"][0]["expressions"]
    assert expressions == dict(
        forward_barrier={"TS6": 1.0, "INT4": -1.0},
        reverse_barrier={"TS6": 1.0, "INT5": -1.0, "HH": -1.0},
        reaction_energy={"INT5": 1.0, "HH": 1.0, "INT4": -1.0},
    )


@pytest.mark.parametrize(
    "ts_types,expected,excluded",
    [
        (
            ["TS1", "TS2", "TS3", "TS4"],
            {"int1", "int2", "INT3", "HBpin-ligand"},
            {"INT4", "INT5", "INT6", "TS5"},
        ),
        (
            ["TS5", "TS6", "TS7", "TS8"],
            {"INT4", "INT5", "INT6"},
            {"TS1", "TS2", "TS3", "TS4", "INT3"},
        ),
        (["TS7"], {"TS7", "INT5", "INT6", "HBpin-mol"}, {"TS8", "TS6", "TS5", "INT4"}),
        ([f"TS{i}" for i in range(1, 9)], {"INT3", "INT4", "INT5", "INT6"}, set()),
    ],
)
def test_selected_coverage_never_traverses_unrequested_reactions(
    ts_types, expected, excluded
):
    result = plan(ts_types)
    actual = {t.state_id for t in result["targets"]}
    assert expected <= actual and not actual & excluded
    assert all(result["reasons"][t.target_id] for t in result["targets"])


@pytest.mark.parametrize("selection", [[], ["TS99"]])
def test_empty_unknown_selection_fails_early(selection):
    with pytest.raises(ValueError):
        ft.workflows.catalyst_screen(systems=pd.DataFrame([SYSTEM]), ts_types=selection)


@pytest.mark.parametrize(
    "method,screening", [("uma-gas", "uma-gas"), ("r2scan-3c", "gxtb-default")]
)
@pytest.mark.parametrize("ranking", [None, "uma-gas", "r2scan-3c"])
@pytest.mark.parametrize("alias", ["validation_top_n", "ranking_top_n"])
def test_unified_and_wrapper_count_method_and_pickle_parity(
    method, screening, ranking, alias, monkeypatch
):
    import pickle
    import frust.structures.builders

    monkeypatch.setattr(
        frust.structures.builders,
        "build",
        lambda *a, **k: pytest.fail("Planning embedded"),
    )
    options = dict(
        systems=pd.DataFrame([SYSTEM]),
        method=method,
        screening=screening,
        ranking=ranking,
        spec_profile="uma-gas",
        **{alias: 3},
    )
    unified = ft.workflows.catalyst_screen(
        ts_types=["TS5", "TS6", "TS7", "TS8"], scope="full_cycle", **options
    )
    wrapper = ft.workflows.bisarylation_screen(**options)
    assert type(unified) is type(wrapper)
    assert [t.target.target_id for t in unified.targets()] == [
        t.target.target_id for t in wrapper.targets()
    ]
    for child in unified.children().values():
        assert child.validation_top_n == 3
        assert [t.as_dict() for t in pickle.loads(pickle.dumps(child)).targets()] == [
            t.as_dict() for t in child.targets()
        ]
        assert child.method.resolved_validation_roles()["frequencies"] in [
            s.method_stage or s.id for s in child._stage_defs()
        ]
    with pytest.raises(ValueError, match="agree|conflict"):
        ft.workflows.catalyst_screen(
            **{**options, "validation_top_n": 2, "ranking_top_n": 3}
        )


def test_synthetic_third_network_uses_same_planner_dispatch_collection_and_analysis(
    tmp_path, monkeypatch
):
    from frust.structures.specs import STATE_SPECS, StateSpec
    from test_bisaryl_ts_candidates import candidate
    from frust.workflows.screening import CatalystScreenWorkflow

    monkeypatch.setitem(
        STATE_SPECS,
        "TS90",
        StateSpec(
            "TS90", "transition_state", "connected_graph::TS90::test", "system_rpos"
        ),
    )
    recipe = ReactionDefinition(
        "synthetic.transfer",
        "synthetic",
        "TS90",
        {"int1": 1.0},
        {"int2": 1.0},
        {s: (s, "incoming") for s in ("TS90", "int1", "int2")},
    )
    definition = NetworkDefinition(
        "synthetic",
        (recipe,),
        {"Start": {"int1": 1.0}, "TS90": {"TS90": 1.0}, "End": {"int2": 1.0}},
        "Start",
    )
    try:
        register_network(definition)
        spec = replace(candidate("TS5"), name="TS90", spec_id="TS90::test")
        wf = ft.workflows.catalyst_screen(
            systems=pd.DataFrame([SYSTEM]),
            ts_types=["TS90"],
            scope="full_cycle",
            method="uma-gas",
            screening="uma-gas",
            candidate_specs={"TS90": spec},
        )
        assert type(wf) is CatalystScreenWorkflow
        assert {t.target.state_id for t in wf.targets()} == {"TS90", "int1", "int2"}
        wf._write_manifest(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        case = manifest["reaction_plan"]["cases"][0]
        states = pd.DataFrame(
            [
                dict(
                    structure_id=identity,
                    result_id=role,
                    quality_status="ready",
                    formula={"H": 2},
                    electronic_energy_hartree=e,
                    free_energy_hartree=e,
                    energy_protocol_fingerprint="test",
                )
                for role, identity in case["bindings"].items()
                for e in [dict(TS90=-0.8, int1=-1.0, int2=-1.1)[role]]
            ]
        )
        barriers, profiles = reaction_tables(
            tmp_path, states, manifest, pd.DataFrame(), calculation_level="full"
        )
        assert barriers.quality_status.tolist() == ["ready"]
        assert set(profiles.network_id) == {"synthetic"}
        assert barriers.iloc[0].delta_e_kcal_mol == pytest.approx(0.2 * 627.5094740631)
        # The persisted recipe stays analyzable without a live registry entry.
        NETWORKS.pop("synthetic")
        assert reaction_tables(
            tmp_path, states, manifest, pd.DataFrame(), calculation_level="full"
        )[0].equals(barriers)
    finally:
        NETWORKS.pop("synthetic", None)


def test_expression_cancels_missing_spectators_and_checks_surface_inventory():
    from frust.workflows.reactions import difference

    ts = pd.Series(
        dict(
            result_id="ts",
            quality_status="ready",
            formula={"H": 2},
            electronic_energy_hartree=-1.0,
            free_energy_hartree=-1.0,
            energy_protocol_fingerprint="surface",
        )
    )
    minimum = ts.copy()
    minimum.result_id = "minimum"
    minimum.electronic_energy_hartree = -1.1
    terms = difference({"TS": 1, "spectator": 1}, {"minimum": 1, "spectator": 1})
    assert (
        evaluate_expression({"TS": ts, "minimum": minimum}, terms)["quality_status"]
        == "ready"
    )
    minimum.energy_protocol_fingerprint = "different"
    assert (
        evaluate_expression({"TS": ts, "minimum": minimum}, terms)["quality_status"]
        == "invalid"
    )
    minimum.energy_protocol_fingerprint = "surface"
    minimum["formula"] = {"H": 1}
    assert (
        evaluate_expression({"TS": ts, "minimum": minimum}, terms)["quality_status"]
        == "invalid"
    )


def test_third_network_native_mocked_execution_and_portable_collection(
    tmp_path, monkeypatch
):
    from frust.structures.specs import STATE_SPECS, StateSpec
    import test_modular_workflow_validation as helper
    from test_bisaryl_ts_candidates import candidate
    from frust.screen.runs import ScreenRun

    monkeypatch.setitem(
        STATE_SPECS,
        "TS90",
        StateSpec(
            "TS90", "transition_state", "connected_graph::TS90::test", "system_rpos"
        ),
    )
    monkeypatch.setitem(helper.FORMULAS, "TS90", helper.FORMULAS["int1"])
    monkeypatch.setitem(helper.FORMULAS, "int2", helper.FORMULAS["int1"])
    monkeypatch.setitem(helper.ENERGIES, "TS90", -29.8)
    roles = {s: (s, "incoming") for s in ("TS90", "int1", "int2")}
    reaction = ReactionDefinition(
        "third.native",
        "third",
        "TS90",
        {"int1": 1.0},
        {"int2": 1.0},
        roles,
        report_review_energy=True,
    )
    network = NetworkDefinition(
        "third",
        (reaction,),
        {"Start": {"int1": 1.0}, "TS90": {"TS90": 1.0}, "End": {"int2": 1.0}},
        "Start",
    )
    try:
        register_network(network)
        spec = replace(candidate("TS5"), name="TS90", spec_id="TS90::mock")
        wf = ft.workflows.catalyst_screen(
            systems=pd.DataFrame([SYSTEM]),
            ts_types=["TS90"],
            scope="full_cycle",
            method="uma-gas",
            screening="uma-gas",
            candidate_specs={"TS90": spec},
            validation_top_n=2,
            n_confs=4,
            prune_initial=False,
        )
        run = helper.run_mocked(wf, tmp_path)
        assert set(run.states().state_id) == {"TS90", "int1", "int2"}
        assert len(run.candidate_barriers()) == 2
        assert run.barriers().delta_e_kcal_mol.notna().all()
        assert run.profile(quantity="electronic").network_id.tolist() == ["third"] * 3
        NETWORKS.pop("third")
        moved = tmp_path / "relocated"
        import shutil

        shutil.copytree(run.path, moved)
        reopened = ScreenRun(moved).refresh_analysis()
        pd.testing.assert_frame_equal(run.barriers(), reopened.barriers())
    finally:
        NETWORKS.pop("third", None)


@pytest.mark.parametrize(
    "method,screening", [("uma-gas", "uma-gas"), ("r2scan-3c", "gxtb-default")]
)
def test_mixed_native_run_collects_both_named_branches(
    tmp_path, monkeypatch, method, screening
):
    from collections import Counter
    import test_modular_workflow_validation as helper
    from frust.workflows.bisarylation_analysis import _review_roles

    def add(*parts):
        out = Counter()
        for part, weight in parts:
            for atom, count in part.items():
                out[atom] += weight * count
        return dict(out)

    adduct = add((helper.FORMULAS["int2"], 1), (helper.FORMULAS["ligand"], 1))
    diaryl = add((adduct, 1), (helper.FORMULAS["HH"], -1))
    exchange = add((diaryl, 1), (helper.FORMULAS["HBpin-mol"], 1))
    for state, formula in [
        ("TS5", adduct),
        ("INT4", adduct),
        ("TS6", adduct),
        ("INT5", diaryl),
        ("TS7", exchange),
        ("INT6", exchange),
        ("TS8", exchange),
    ]:
        monkeypatch.setitem(helper.FORMULAS, state, formula)
        monkeypatch.setitem(helper.ENERGIES, state, -30.0 + len(helper.ENERGIES) * 0.01)
    wf = ft.workflows.catalyst_screen(
        systems=pd.DataFrame([SYSTEM]),
        ts_types=[f"TS{i}" for i in range(1, 9)],
        scope="full_cycle",
        method=method,
        screening=screening,
        spec_profile="uma-gas",
        validation_top_n=2,
        top_n=3,
        prune_initial=False,
    )
    run = helper.run_mocked(wf, tmp_path)
    states = run.states()
    assert {f"TS{i}" for i in range(1, 9)} <= set(states.state_id)
    assert {"INT3", "INT4", "INT5", "INT6"} <= set(states.state_id)
    assert run.candidate_barriers().groupby("ts_type").size().eq(2).all()
    pd.DataFrame(
        [
            dict(
                result_id=r,
                decision="approved",
                reviewer="test",
                note="mocked numerical fixture",
                reviewed_at="2026-10-29",
            )
            for r in states.result_id
        ]
    ).to_csv(run.reviews_path, index=False)
    run.refresh_analysis()
    pool = run.pathway_candidates(
        system_name=SYSTEM["system_name"], rpos=2, variant="two_saddle"
    )
    ids = {
        role: pool.loc[pool.role.eq(role)]
        .sort_values("free_energy_hartree")
        .iloc[0]
        .result_id
        for role in _review_roles("two_saddle")
    }
    proof = tmp_path / "mocked-path.json"
    proof.write_text('{"fixture":true}')
    run.set_pathway_review(
        system_name=SYSTEM["system_name"],
        rpos=2,
        variant="two_saddle",
        result_ids=ids,
        evidence_files=[proof],
    )
    assert len(run.barriers()) == 8
    assert set(run.profile(network_id="productive").network_id) == {"productive"}
    assert set(run.profile(network_id="bisarylation").network_id) == {"bisarylation"}
    with pytest.raises(ValueError, match="network_id"):
        run.plot_profile(system_name=SYSTEM["system_name"], rpos=2)


def test_ts6_partial_manifest_does_not_claim_unscheduled_dimers(tmp_path):
    wf = ft.workflows.catalyst_screen(
        systems=pd.DataFrame([SYSTEM]),
        ts_types=["TS6"],
        scope="full_cycle",
        method="uma-gas",
        screening="uma-gas",
    )
    wf._write_manifest(tmp_path)
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["dimer_candidates"] == []
    )
    run = ft.screen.open_run(tmp_path).refresh_analysis()
    assert run.dimer_references().empty
    assert run.profile(
        include_invalid=True, quantity="electronic"
    ).profile_state.tolist() == ["TS6:reactants", "TS6", "TS6:products"]


@pytest.mark.parametrize(
    "variant,ts,absent",
    [("ts7_only", "TS7", {"TS8", "INT6"}), ("ts8_only", "TS8", {"TS7", "INT6"})],
)
def test_generic_alternatives_update_sides_and_do_not_invent_missing_states(
    variant, ts, absent
):
    from frust.workflows.bisarylation import _system_signature

    signature = _system_signature(SYSTEM, 2)
    review = dict(
        variant=variant,
        system_signature=signature,
        method_fingerprint="method",
        evidence_sha256=["a" * 64],
    )
    result = plan_reactions(
        pd.DataFrame([SYSTEM]),
        ts_types=["TS5", "TS6", "TS7", "TS8"],
        scope="full_cycle",
        method_fingerprint="method",
        pathway_reviews={signature: review},
    )
    assert not {t.state_id for t in result["targets"]} & absent
    reaction = next(r for r in result["cases"][0]["reactions"] if r["ts_state"] == ts)
    assert reaction["products"] == {"int2": 1.0, "bound_product": 1.0}
    assert reaction["reactants"] == {"INT5": 1.0, "HBpin-mol": 1.0}


def test_registry_rejects_conflicting_roles_and_invalid_stoichiometry():
    first = ReactionDefinition(
        "third.one",
        "third",
        "TS1",
        {"int1": 1.0},
        {"int2": 1.0},
        {s: (s, "incoming") for s in ["TS1", "int1", "int2"]},
    )
    with pytest.raises(ValueError, match="positive"):
        replace(first, reactants={"int1": -1.0}).validate()
    second = replace(
        first, reaction_id="third.two", roles={**first.roles, "int2": ("int2", "bound")}
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        NetworkDefinition(
            "third", (first, second), {"start": {"int1": 1.0}}, "start"
        ).validate()


def test_reference_applicability_does_not_follow_only_optional_core_states():
    result = plan(["TS6"])
    assert result["cases"][0]["applicability"]["HH"] == "required"
    assert result["cases"][0]["applicability"]["TS8"] == "not_requested"
