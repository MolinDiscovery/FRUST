"""Built-in gas-phase r2SCAN-3c ``tsguess2`` geometries."""

from __future__ import annotations

from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec


GEOMETRY_KEY = GeometryKey(method="r2scan-3c")
TS_SOURCE_SHA256 = "966555c888fa6a41969f791d93bcf97fa9898df89bb2d0402f7c0088819c6e31"
INT3_SOURCE_SHA256 = "d316deaefc39d41683a99df28819dabbc9e194c012688f2fe264c5016d43e3ee"


def _reference(
    *,
    state: str,
    negative_frequencies: tuple[float, ...],
) -> ReferenceRecord:
    """Return provenance for one reviewed gas-phase reference row."""
    return ReferenceRecord(
        substrate_name="1-methylpyrrole",
        catalyst_name="NMe",
        method="r2SCAN-3c",
        basis=None,
        solvation_model=None,
        solvent=None,
        coordinates_column="dft_ts_opt-oc" if state.startswith("TS") else "dft_opt-oc",
        vibrations_column="dft_freq-vibs",
        negative_frequencies=negative_frequencies,
        mode_reviewed=True,
        source_sha256=TS_SOURCE_SHA256 if state.startswith("TS") else INT3_SOURCE_SHA256,
        notes="Extracted from reviewed 2026 gas-phase r2SCAN-3c specification calculations.",
    )


GEOMETRIES: dict[str, StateGeometrySpec] = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.210596, 1.289281, -1.309434),
            "cat_N": (-0.720317, -1.294644, -0.349124),
            "substrate_C": (1.456758, 0.206029, -0.914079),
            "transfer_H": (0.539469, -0.625080, -0.660208),
        },
        constraint_values={
            "catN_transferH": 1.4601883341,
            "transferH_substrateC": 1.2635710364,
            "catB_substrateC": 1.6978398634,
            "catB_catN": 2.9095468230,
        },
        reference=_reference(state="TS1", negative_frequencies=(-205.46,)),
    ),
    "TS2": StateGeometrySpec(
        state="TS2",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "B_transfer_H": (0.393184, -1.487301, -0.376741),
            "N_transfer_H": (-0.446299, -1.471168, -0.364664),
            "cat_B": (0.651686, 0.038738, -0.503130),
            "cat_N": (-2.060499, -1.062265, -0.358241),
        },
        constraint_values={
            "catB_BtransferH": 1.5529302923,
            "catN_NtransferH": 1.6651980538,
            "catB_catN": 2.9307248084,
            "catB_BtransferH_catN": 89.8221920519,
        },
        reference=_reference(state="TS2", negative_frequencies=(-328.55,)),
    ),
    "TS3": StateGeometrySpec(
        state="TS3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.194741, -1.254048, 0.565149),
            "pin_B": (-0.368805, 0.427510, 0.575883),
            "substrate_C": (-0.008737, -2.128511, 0.094383),
            "transfer_H": (0.770041, 0.209869, 0.944329),
        },
        constraint_values={
            "transferH_catB": 1.5707324869,
            "transferH_pinB": 1.2165896069,
            "transferH_substrateC": 2.6070911761,
            "catB_substrateC": 1.5603414567,
            "pinB_substrateC": 2.6257826580,
            "pinB_catB": 2.2961769575,
            "catB_transferH_pinB": 110.2871104844,
            "catB_substrateC_pinB": 60.3236100841,
        },
        reference=_reference(state="TS3", negative_frequencies=(-82.84,)),
    ),
    "TS4": StateGeometrySpec(
        state="TS4",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.807881, -0.791511, 1.069424),
            "pin_B": (-1.097932, -0.505187, 0.385269),
            "substrate_C": (-0.263789, -1.841267, 0.083125),
            "transfer_H": (0.236240, 0.297947, 0.991191),
        },
        constraint_values={
            "catB_pinB": 2.0450361092,
            "pinB_transferH": 1.6709819322,
            "substrateC_transferH": 2.3771515305,
            "catB_transferH": 1.2328067849,
            "catB_substrateC": 1.7953411837,
            "pinB_substrateC": 1.6038065057,
            "catB_transferH_pinB": 88.1943189838,
            "catB_substrateC_pinB": 73.7312981956,
        },
        reference=_reference(state="TS4", negative_frequencies=(-72.05,)),
    ),
    "INT3": StateGeometrySpec(
        state="INT3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (0.476934, 0.320552, -1.563391),
            "pin_B": (-0.960656, -0.033384, -0.399932),
            "substrate_C": (-0.545237, 1.519276, -0.972105),
            "transfer_H": (0.167153, -0.716658, -0.890704),
        },
        constraint_values={
            "catB_transferH": 1.2744711264,
            "catB_substrateC": 1.6826740371,
            "pinB_transferH": 1.4070087617,
            "pinB_substrateC": 1.7060797060,
            "catB_transferH_pinB": 89.0673204356,
            "catB_substrateC_pinB": 67.5072314576,
        },
        reference=_reference(state="INT3", negative_frequencies=()),
    ),
}
