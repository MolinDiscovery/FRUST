"""Built-in wB97X-D3/6-31G** gas-phase ``tsguess2`` geometries."""

from __future__ import annotations

from frust.tsguess2.models import GeometryKey, ReferenceRecord, StateGeometrySpec


GEOMETRY_KEY = GeometryKey(method="wb97xd3-631g", basis="6-31g**")


def _reference(
    *,
    catalyst: str,
) -> ReferenceRecord:
    """Return provenance for a migrated built-in wB97 geometry."""
    return ReferenceRecord(
        substrate_name="1-methylpyrrole",
        catalyst_name=catalyst,
        method="wB97X-D3",
        basis="6-31G**",
        solvation_model=None,
        solvent=None,
        coordinates_column="reference geometry",
        vibrations_column="reference frequencies",
        negative_frequencies=None,
        mode_reviewed=None,
        notes="Migrated from the original built-in tsguess2 v2 specification.",
    )


GEOMETRIES: dict[str, StateGeometrySpec] = {
    "TS1": StateGeometrySpec(
        state="TS1",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "transfer_H": (-1.558624, 0.103600, 1.047895),
            "cat_B": (0.373318, -0.291503, 1.700016),
            "cat_N": (-2.416144, -1.101445, 0.730403),
            "substrate_C": (-0.659429, 0.986440, 1.328243),
        },
        constraint_values={
            "catN_transferH": 1.51270,
            "transferH_substrateC": 1.29095,
            "catB_substrateC": 1.68461,
            "catB_catN": 3.06223,
        },
        reference=_reference(catalyst="NMe"),
    ),
    "TS2": StateGeometrySpec(
        state="TS2",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (4.505621, 2.687572, 0.441993),
            "B_transfer_H": (5.642044, 2.651080, -0.820176),
            "N_transfer_H": (5.529646, 1.878196, -0.907186),
            "cat_N": (5.173259, 0.041777, -1.003704),
        },
        constraint_values={
            "catB_BtransferH": 1.656,
            "catN_NtransferH": 1.961,
            "catB_catN": 3.080,
            "catB_BtransferH_catN": 87.38739,
        },
        reference=_reference(catalyst="NMe"),
    ),
    "TS3": StateGeometrySpec(
        state="TS3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.201563, 0.080366, 0.660199),
            "transfer_H": (1.676962, -1.004507, -0.052686),
            "pin_B": (2.532308, -1.428336, 0.777578),
            "substrate_C": (1.976672, 0.248906, 2.068494),
        },
        constraint_values={
            "transferH_catB": 1.37581,
            "transferH_pinB": 1.26409,
            "transferH_substrateC": 2.47686,
            "catB_substrateC": 1.61593,
            "pinB_substrateC": 2.17986,
            "pinB_catB": 2.00709,
            "catB_transferH_pinB": 98.89,
            "catB_substrateC_pinB": 61.75,
        },
        reference=_reference(catalyst="TMP"),
    ),
    "TS4": StateGeometrySpec(
        state="TS4",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (-0.930038, 0.590384, 1.929793),
            "transfer_H": (-0.087884, 1.262005, 1.344826),
            "pin_B": (0.999483, 1.217369, 2.683538),
            "substrate_C": (0.013065, 0.446161, 3.676874),
        },
        constraint_values={
            "catB_pinB": 2.21926,
            "pinB_transferH": 1.86758,
            "substrateC_transferH": 2.48888,
            "catB_transferH": 1.21598,
            "catB_substrateC": 1.94626,
            "pinB_substrateC": 1.58475,
            "catB_transferH_pinB": 89.48,
            "catB_substrateC_pinB": 77.13,
        },
        reference=_reference(catalyst="TMP"),
    ),
    "INT3": StateGeometrySpec(
        state="INT3",
        geometry_key=GEOMETRY_KEY,
        revision=1,
        role_coordinates={
            "cat_B": (1.239315, 0.051379, 0.884948),
            "transfer_H": (1.922198, -0.860414, 0.276733),
            "pin_B": (2.609655, -1.114321, 1.414101),
            "substrate_C": (1.789172, 0.052638, 2.481973),
        },
        constraint_values={
            "catB_transferH": 1.279,
            "catB_substrateC": 1.688,
            "pinB_transferH": 1.378,
            "pinB_substrateC": 1.749,
            "catB_transferH_pinB": 89.85,
            "catB_substrateC_pinB": 66.22,
        },
        reference=_reference(catalyst="TMP"),
    ),
}
