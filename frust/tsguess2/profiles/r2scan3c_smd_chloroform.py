"""Built-in r2SCAN-3c/SMD(chloroform) ``tsguess2`` geometries."""

from __future__ import annotations

from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec


GEOMETRY_KEY = GeometryKey(
    method="r2scan-3c",
    solvation_model="smd",
    solvent="chloroform",
)
TS_SOURCE_SHA256 = "de0b267965013ad013287765e9dbb7161bf14b9196ef51b55724f84657c4f84b"
INT3_SOURCE_SHA256 = "70fc89bcaa9311a719580d7022949b23e1a79e93cd90f5ceb0277b7e49a74e00"


def _reference(
    *,
    state: str,
    negative_frequencies: tuple[float, ...],
) -> ReferenceRecord:
    """Return provenance for one supplied r2SCAN-3c reference row."""
    return ReferenceRecord(
        substrate_name="1-methylpyrrole",
        catalyst_name="NMe",
        method="r2SCAN-3c",
        basis=None,
        solvation_model="SMD",
        solvent="chloroform",
        coordinates_column="dft_ts_opt-oc" if state.startswith("TS") else "dft_opt-oc",
        vibrations_column="dft_freq-vibs",
        negative_frequencies=negative_frequencies,
        mode_reviewed=state != "TS3",
        source_sha256=TS_SOURCE_SHA256 if state.startswith("TS") else INT3_SOURCE_SHA256,
        notes="Extracted from the supplied 2026 r2SCAN-3c specification calculations.",
    )


GEOMETRIES: dict[str, StateGeometrySpec] = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (-0.378783, 1.673933, 0.492673),
            "cat_N": (1.236509, -0.429423, -0.777936),
            "substrate_C": (0.264459, 0.544575, 1.585717),
            "transfer_H": (0.817089, -0.001524, 0.637900),
        },
        constraint_values={
            "catN_transferH": 1.5374011414,
            "transferH_substrateC": 1.2255533853,
            "catB_substrateC": 1.6982211130,
            "catB_catN": 2.9407009265,
        },
        reference=_reference(state="TS1", negative_frequencies=(-125.47,)),
    ),
    "TS2": StateGeometrySpec(
        state="TS2",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "B_transfer_H": (0.564600, -1.820645, 0.197384),
            "N_transfer_H": (-0.222899, -1.756405, 0.342390),
            "cat_B": (0.667659, -0.607809, -0.939775),
            "cat_N": (-1.972516, -1.269625, 0.298000),
        },
        constraint_values={
            "catB_BtransferH": 1.6657499547,
            "catN_NtransferH": 1.8166135767,
            "catB_catN": 2.9900855170,
            "catB_BtransferH_catN": 86.1260118463,
        },
        reference=_reference(state="TS2", negative_frequencies=(-330.66,)),
    ),
    "TS4": StateGeometrySpec(
        state="TS4",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.826020, -0.811711, 1.021226),
            "pin_B": (-1.049345, -0.498024, 0.389794),
            "substrate_C": (-0.237905, -1.870112, 0.111216),
            "transfer_H": (0.247668, 0.281246, 0.961567),
        },
        constraint_values={
            "catB_pinB": 2.0035218461,
            "pinB_transferH": 1.6175378878,
            "substrateC_transferH": 2.3637299350,
            "catB_transferH": 1.2379843440,
            "catB_substrateC": 1.7550690250,
            "pinB_substrateC": 1.6182292963,
            "catB_transferH_pinB": 88.0692436429,
            "catB_substrateC_pinB": 72.7456097869,
        },
        reference=_reference(state="TS4", negative_frequencies=(-52.26,)),
    ),
    "INT3": StateGeometrySpec(
        state="INT3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.068633, 0.958031, 0.624149),
            "pin_B": (-0.652786, 0.321493, 0.209640),
            "substrate_C": (-0.047062, 1.895878, -0.161757),
            "transfer_H": (0.486527, -0.160149, 0.853871),
        },
        constraint_values={
            "catB_transferH": 1.2813844485,
            "catB_substrateC": 1.6558926805,
            "pinB_transferH": 1.3946493142,
            "pinB_substrateC": 1.7272884600,
            "catB_transferH_pinB": 89.2511744246,
            "catB_substrateC_pinB": 67.5420721389,
        },
        reference=_reference(state="INT3", negative_frequencies=()),
    ),
}


QUARANTINED_STATES = {
    "TS3": "Reference has two negative frequencies (-93.72 and -69.89 cm^-1) and the wrong mode.",
}
