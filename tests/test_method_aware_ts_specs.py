from __future__ import annotations

import pandas as pd
import pytest

import frust as ft
from frust.constraints import render_orca_constraints, render_xtb_constraints
from frust.tsguess2 import resolve_profile_spec


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": [
                "CN1C=CC=C1",
                "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
            ],
            "compound_name": ["pyrrole", "cat"],
            "rpos": ["2", None],
        }
    )


def test_profile_catalog_exposes_all_four_method_environment_slots():
    profiles = ft.show_spec_profiles()

    assert set(profiles["profile"]) == {
        "wb97xd3-631g/gas",
        "wb97xd3-631g/smd-chloroform",
        "r2scan-3c/gas",
        "r2scan-3c/smd-chloroform",
    }
    r2_solv = profiles[profiles["profile"].eq("r2scan-3c/smd-chloroform")]
    assert set(r2_solv.loc[r2_solv["status"].eq("active"), "state"]) == {
        "TS1",
        "TS2",
        "TS4",
        "INT3",
    }
    assert r2_solv.set_index("state").loc["TS3", "status"] == "quarantined"


def test_profile_resolution_falls_back_only_within_method_family():
    r2 = resolve_profile_spec("TS4", "r2scan-3c/gas")
    wb97 = resolve_profile_spec("TS4", "wb97xd3-631g-solv")

    assert r2.resolved_profile == "r2scan-3c/smd-chloroform"
    assert r2.match == "same_method_environment_fallback"
    assert wb97.resolved_profile == "wb97xd3-631g/gas"
    assert wb97.spec.spec_id.startswith("TS4::tsguess2-v2::wb97xd3-631g")
    with pytest.raises(ValueError, match="No selectable TS4"):
        resolve_profile_spec("TS4", "r2scan-3c/gas", match="exact")


def test_quarantined_r2scan_ts3_is_never_selected():
    for profile in ("r2scan-3c/smd-chloroform", "r2scan-3c/gas"):
        with pytest.raises(ValueError, match="wrong mode"):
            resolve_profile_spec("TS3", profile)


def test_workflow_method_plan_selects_geometry_profile_upstream():
    solv = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS4"],
        method="r2scan-3c-solv",
    )
    gas = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS4"],
        method="wb97xd3-631g",
    )

    assert solv.resolved_spec_profile == "r2scan-3c/smd-chloroform"
    assert gas.resolved_spec_profile == "wb97xd3-631g/gas"
    assert len(solv.targets()) == 1
    assert len(gas.targets()) == 1


def test_workflow_validates_missing_or_quarantined_profiles_before_embedding():
    exact_missing = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS4"],
        method="r2scan-3c",
        spec_match="exact",
    )
    quarantined = ft.workflows.screen_ts(
        dataframe=_components(),
        ts_types=["TS3"],
        method="r2scan-3c-solv",
    )

    with pytest.raises(ValueError, match="has not been supplied"):
        exact_missing.targets()
    with pytest.raises(ValueError, match="wrong mode"):
        quarantined.targets()


def test_explicit_legacy_upgrade_creates_self_describing_constraints():
    old = pd.DataFrame(
        {
            "structure_type": ["TS4"],
            "constraint_atoms": [[10, 11, 12, 13, 14, 15]],
            "atoms": [["H"] * 16],
            "coords_embedded": [[(0.0, 0.0, 0.0)] * 16],
        }
    )

    modern = ft.upgrade_legacy_constraints(
        old,
        spec_profile="wb97xd3-631g/gas",
    )

    assert "constraint_atoms" in old
    assert "constraint_atoms" not in modern
    assert modern.loc[0, "constraint_roles"] == {
        "cat_B": 10,
        "transfer_H": 13,
        "pin_B": 14,
        "substrate_C": 15,
    }
    assert len(modern.loc[0, "constraint_spec"]) == 8
    assert "distance: 11, 15, 2.21926" in render_xtb_constraints(modern.loc[0])
    assert "{B 10 14 2.21926 C}" in render_orca_constraints(modern.loc[0])
    assert modern.attrs["frust_constraint_upgrade"]["source"] == "constraint_atoms"


def test_stepper_rejects_positional_constraints_without_explicit_upgrade():
    positional = pd.DataFrame(
        {
            "atoms": [["H", "H"]],
            "coords_embedded": [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.7)]],
            "constraint_atoms": [[0, 1, 0, 1, 0, 1]],
        }
    )
    step = ft.Stepper(step_type="TS1", debug=True, save_output_dir=False)

    with pytest.raises(ValueError, match="role-based row constraints"):
        step._validate_constraint_request(positional)


def test_stepper_rejects_malformed_role_constraint_before_calculator_dispatch():
    malformed = pd.DataFrame(
        {
            "atoms": [["H", "H", "H"]],
            "coords_embedded": [[(0.0, 0.0, 0.0)] * 3],
            "constraint_roles": [{"a": 0, "b": 1, "c": 2}],
            "constraint_spec": [
                [{"kind": "distance", "roles": ["a", "b", "c"], "value": 1.0}]
            ],
        }
    )
    step = ft.Stepper(debug=True, save_output_dir=False)

    with pytest.raises(ValueError, match="exactly 2 roles"):
        step._validate_constraint_request(malformed)
