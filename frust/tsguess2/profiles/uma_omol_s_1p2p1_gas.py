"""Reviewed gas-phase UMA-S 1.2.1 (OMol) ``tsguess2`` geometries."""

from __future__ import annotations

from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec

GEOMETRY_KEY = GeometryKey(method="omol-uma-s-1p2p1")


def _reference(state: str) -> ReferenceRecord:
    """Return provenance for one reviewed gas-phase UMA reference."""
    records = {
        "TS1": (
            "1-methylpyrrole",
            "NMe",
            "uma_ts_opt-oc",
            (-944.65,),
            "aafe5d1891a43859316085bd0b70ff939035451d8cc9ab35ced1856349af8d5b",
        ),
        "TS2": (
            "1-methylpyrrole",
            "NMe",
            "uma_ts_opt-oc",
            (-278.92,),
            "dec536ae81e06c5754adbc7f854254366a79b54966347f80ed8315528dda5219",
        ),
        "TS3": (
            "thiophene",
            "TMP",
            "uma_ts_opt-oc",
            (-94.46,),
            "edbe7400d196574f3a0f27f998bcd110dbb394cb6fbef9963b07db20a1693d6e",
        ),
        "TS4": (
            "thiophene",
            "TMP",
            "uma_ts_opt-oc",
            (-65.68,),
            "ab73112876d2c3e85937c7809a7cc46b57bc0c72b6632560d090a78968547533",
        ),
        "INT3": (
            "thiophene",
            "TMP",
            "uma_opt-oc",
            (),
            "cf0ddea0d8059c3fbe8daa95dcc4b8149e0eb18acf08656333d1452b7d639c18",
        ),
    }
    substrate, catalyst, coordinates_column, negatives, source_sha256 = records[state]
    notes = (
        "Gas-phase UMA OptTS and NumFreq; TS3 started from an ALPB seed and "
        "its reactive mode was accepted after user review."
        if state == "TS3"
        else "Gas-phase UMA optimization and numerical frequency; reactive mode reviewed."
    )
    return ReferenceRecord(
        substrate_name=substrate,
        catalyst_name=catalyst,
        method="UMA-S 1.2.1 (OMol)",
        basis=None,
        solvation_model=None,
        solvent=None,
        coordinates_column=coordinates_column,
        vibrations_column="uma_freq-vibs",
        negative_frequencies=negatives,
        mode_reviewed=True,
        source_sha256=source_sha256,
        notes=notes,
    )


GEOMETRIES: dict[str, StateGeometrySpec] = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.361732, -0.297923, 1.690497),
            "cat_N": (-2.420861, -1.106491, 0.729885),
            "substrate_C": (-0.676943, 0.997887, 1.328876),
            "transfer_H": (-1.55902, 0.081966, 1.047264),
        },
        constraint_values={
            "catB_catN": 3.0527661929497647,
            "catB_substrateC": 1.6996290946456523,
            "catN_transferH": 1.5019751595053095,
            "transferH_substrateC": 1.3024117746373456,
        },
        reference=_reference("TS1"),
    ),
    "TS2": StateGeometrySpec(
        state="TS2",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "B_transfer_H": (5.570947, 2.671637, -0.84327),
            "N_transfer_H": (5.487836, 1.884918, -0.900533),
            "cat_B": (4.512828, 2.680796, 0.417464),
            "cat_N": (5.17351, 0.047613, -1.010728),
        },
        constraint_values={
            "catB_BtransferH": 1.645949550927367,
            "catB_BtransferH_catN": 87.57256103806513,
            "catB_catN": 3.067553716803831,
            "catN_NtransferH": 1.867252911452008,
        },
        reference=_reference("TS2"),
    ),
    "TS3": StateGeometrySpec(
        state="TS3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.286156, 0.024608, 0.790853),
            "pin_B": (2.760429, -1.129134, 1.623895),
            "substrate_C": (2.119071, 0.09762, 2.47372),
            "transfer_H": (1.859982, -0.957609, 0.324649),
        },
        constraint_values={
            "catB_substrateC": 1.8791273211408535,
            "catB_substrateC_pinB": 71.16329601133123,
            "catB_transferH_pinB": 92.32969531592302,
            "pinB_catB": 2.0490389100397772,
            "pinB_substrateC": 1.6243361694258367,
            "transferH_catB": 1.2293789826497767,
            "transferH_pinB": 1.5900521356075088,
            "transferH_substrateC": 2.408140675999432,
        },
        reference=_reference("TS3"),
    ),
    "TS4": StateGeometrySpec(
        state="TS4",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (-0.892814, 0.54166, 1.899577),
            "pin_B": (0.91375, 1.185656, 2.489847),
            "substrate_C": (-0.068087, 0.406795, 3.546682),
            "transfer_H": (-0.016732, 1.144139, 1.268593),
        },
        constraint_values={
            "catB_pinB": 2.006694547511404,
            "catB_substrateC": 1.8469753852661384,
            "catB_substrateC_pinB": 69.99157703821759,
            "catB_transferH": 1.2363823940921352,
            "catB_transferH_pinB": 92.10045577492279,
            "pinB_substrateC": 1.6393683445507297,
            "pinB_transferH": 1.5358977004113914,
            "substrateC_transferH": 2.394995407570127,
        },
        reference=_reference("TS4"),
    ),
    "INT3": StateGeometrySpec(
        state="INT3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.186159, 0.003889, 0.802436),
            "pin_B": (2.57596, -1.283524, 1.219349),
            "substrate_C": (1.695224, 0.071802, 2.357273),
            "transfer_H": (1.78901, -1.016826, 0.217445),
        },
        constraint_values={
            "catB_substrateC": 1.6374606701728747,
            "catB_substrateC_pinB": 64.11714870820788,
            "catB_transferH": 1.3219315071163862,
            "catB_transferH_pinB": 95.35327592452325,
            "pinB_substrateC": 1.9767335424249775,
            "pinB_transferH": 1.3016258106383722,
        },
        reference=_reference("INT3"),
    ),
}
