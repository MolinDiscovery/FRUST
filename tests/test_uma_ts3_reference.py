"""Regression checks for the recovered incoming gas UMA TS3 reference."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import frust as ft
from frust.tsguess2.profiles.uma_omol_s_1p2p1_gas import GEOMETRIES

REFERENCE = json.loads(
    (Path(__file__).parent / "data/uma_ts3_c17_reference.json").read_text()
)
PROFILE = "omol-uma-s-1p2p1/gas"


def test_uma_ts3_revision2_matches_frozen_native_source():
    spec = ft.tsguess2.resolve_profile_spec("TS3", PROFILE, match="exact").spec
    assert spec.spec_id.endswith("::r2")
    assert GEOMETRIES["TS3"].reference.source_sha256 == REFERENCE["source_sha256"]
    assert GEOMETRIES["TS3"].reference.substrate_name == "1-methylpyrrole"
    assert GEOMETRIES["TS3"].reference.coordinates_column == "recover_ts-oc"
    assert GEOMETRIES["TS3"].reference.vibrations_column == "final_freq-vibs"
    assert GEOMETRIES["TS3"].reference.negative_frequencies == (-183.34,)
    coords = np.asarray(REFERENCE["coordinates"])
    for role, point in spec.role_coordinates.items():
        np.testing.assert_array_equal(point, coords[REFERENCE["roles"][role]])
    for constraint in spec.constraints:
        points = [coords[REFERENCE["roles"][r]] for r in constraint.roles]
        if constraint.kind == "distance":
            value = np.linalg.norm(points[0] - points[1])
        else:
            u, v = points[0] - points[1], points[2] - points[1]
            value = np.degrees(
                np.arccos(
                    np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
                )
            )
        assert constraint.value == pytest.approx(value, abs=1e-10)


def test_uma_ts3_core_is_incoming_and_distinct_from_retained_ts4():
    specs = {
        s: ft.tsguess2.resolve_profile_spec(s, PROFILE, match="exact").spec
        for s in ["TS3", "TS4"]
    }
    values = {s: GEOMETRIES[s].constraint_values for s in specs}
    assert values["TS3"]["pinB_substrateC"] > 2.1
    assert values["TS4"]["pinB_substrateC"] < 1.7
    assert values["TS3"]["transferH_pinB"] < values["TS3"]["transferH_catB"]
    assert specs["TS4"].spec_id.endswith("::r1")
    assert (
        GEOMETRIES["TS4"].reference.source_sha256
        == "ab73112876d2c3e85937c7809a7cc46b57bc0c72b6632560d090a78968547533"
    )
    mode = np.asarray(REFERENCE["imaginary_mode"])
    coords = np.asarray(REFERENCE["coordinates"])
    r = REFERENCE["roles"]
    rates = []
    for a, b in [("pin_B", "substrate_C"), ("pin_B", "transfer_H")]:
        delta = coords[r[a]] - coords[r[b]]
        rates.append(
            abs(np.dot(delta / np.linalg.norm(delta), mode[r[a]] - mode[r[b]]))
        )
    assert rates[0] > rates[1]


@pytest.mark.slow
@pytest.mark.parametrize("catalyst", ["CC1(C)CCCC(C)(C)N1c1ccccc1B", "CN(C)c1ccccc1B"])
@pytest.mark.parametrize("substrate,rpos", [("CN1C=CC=C1", 2), ("c1ccsc1", 0)])
def test_uma_ts3_assemblies_preserve_incoming_core_and_valid_fragment_frames(
    catalyst, substrate, rpos
):
    components = pd.DataFrame(
        [
            {"role": "substrate", "smiles": substrate, "rpos": str(rpos)},
            {"role": "catalyst", "smiles": catalyst},
        ]
    )
    rows = ft.screen.create_ts_guesses(
        ft.screen.expand(components),
        ts_types=["TS3"],
        n_confs=1,
        n_cores=1,
        spec_profile=PROFILE,
        spec_match="exact",
    )["TS3"]
    assert len(rows) == 1
    row = rows.iloc[0]
    r = row["constraint_roles"]
    xyz = np.asarray(row["coords_embedded"])
    mol = Chem.AddHs(Chem.MolFromSmiles(row["smiles"]))
    assert row["ts_spec_id"].endswith("::r2")
    assert [a.GetSymbol() for a in mol.GetAtoms()] == list(row["atoms"])
    for role, element in [
        ("cat_B", "B"),
        ("pin_B", "B"),
        ("transfer_H", "H"),
        ("substrate_C", "C"),
    ]:
        assert row["atoms"][r[role]] == element
    oxygens = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(r["pin_B"]).GetNeighbors()
        if a.GetAtomicNum() == 8
    ]
    assert len(oxygens) == 2
    for i in oxygens:
        assert 1.1 < np.linalg.norm(xyz[i] - xyz[r["pin_B"]]) < 1.8
    assert (
        np.linalg.norm(
            np.cross(
                xyz[oxygens[0]] - xyz[r["pin_B"]], xyz[oxygens[1]] - xyz[r["pin_B"]]
            )
        )
        > 0.5
    )
    neighbors = [
        a.GetIdx()
        for a in mol.GetAtomWithIdx(r["substrate_C"]).GetNeighbors()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in [r["cat_B"], r["pin_B"]]
    ]
    assert len(neighbors) == 2
    assert (
        np.linalg.norm(
            np.cross(
                xyz[neighbors[0]] - xyz[r["substrate_C"]],
                xyz[neighbors[1]] - xyz[r["substrate_C"]],
            )
        )
        > 0.5
    )
    assert np.linalg.norm(xyz[r["transfer_H"]] - xyz[r["pin_B"]]) < np.linalg.norm(
        xyz[r["transfer_H"]] - xyz[r["cat_B"]]
    )
    assert np.linalg.norm(xyz[r["substrate_C"]] - xyz[r["cat_B"]]) < np.linalg.norm(
        xyz[r["substrate_C"]] - xyz[r["pin_B"]]
    )
    assert all(np.isfinite(metric["measured"]) for metric in row["ts_core_metrics"])
    spec = ft.tsguess2.resolve_profile_spec("TS3", PROFILE, match="exact").spec
    assert row["constraint_spec"] == spec.constraint_dicts()
