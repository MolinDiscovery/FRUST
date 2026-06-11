"""Built-in tsguess2 role-coordinate and constraint specifications."""

from __future__ import annotations

from dataclasses import dataclass

from frust.tsguess.specs import ConstraintEntry


@dataclass(frozen=True)
class TSGuess2Spec:
    """TS guess instructions for the SMILES-roundtrip backend.

    Parameters
    ----------
    name : str
        Structure type, for example ``"TS1"``.
    spec_id : str
        Versioned identifier stored on generated rows.
    builder_key : str
        Connected-SMILES builder family. Supported values are ``"ts1_ts2"``
        and ``"ts3_ts4"``.
    core_smarts : str
        SMARTS pattern used to recover role atom indices from generated SMILES.
    role_coordinates : dict
        Mapping from V2 role name to Cartesian coordinate tuple.
    constraints : tuple of ConstraintEntry
        V2 role-level distance and angle constraints.
    constraint_order : tuple of str
        Best-effort role order for legacy ``constraint_atoms`` projection.
    """

    name: str
    spec_id: str
    builder_key: str
    core_smarts: str
    role_coordinates: dict[str, tuple[float, float, float]]
    constraints: tuple[ConstraintEntry, ...]
    constraint_order: tuple[str, ...]

    def constraint_dicts(self) -> list[dict[str, object]]:
        """Return constraints as dataframe-friendly dictionaries."""
        return [entry.as_dict() for entry in self.constraints]


BUILTIN_TS_SPECS_V2: dict[str, TSGuess2Spec] = {
    "TS1": TSGuess2Spec(
        name="TS1",
        spec_id="TS1::builtin::methylpyrrole_v2",
        builder_key="ts1_ts2",
        core_smarts="[#1]~[#7]~[*]~[*]~[#5]~[#6]",
        role_coordinates={
            "transfer_H": (-1.558624, 0.103600, 1.047895),
            "cat_B": (0.373318, -0.291503, 1.700016),
            "cat_N": (-2.416144, -1.101445, 0.730403),
            "substrate_C": (-0.659429, 0.986440, 1.328243),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_N", "transfer_H"), 1.51270),
            ConstraintEntry("distance", ("transfer_H", "substrate_C"), 1.29095),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.68461),
            ConstraintEntry("distance", ("cat_B", "cat_N"), 3.06223),
        ),
        constraint_order=("cat_B", "cat_N", "transfer_H", "substrate_C"),
    ),
    "TS2": TSGuess2Spec(
        name="TS2",
        spec_id="TS2::builtin::methylpyrrole_v2",
        builder_key="ts1_ts2",
        core_smarts="[#1]~[#7]~[*]~[*]~[#5]~[#6]",
        role_coordinates={
            "cat_B": (4.505621, 2.687572, 0.441993),
            "B_transfer_H": (5.642044, 2.651080, -0.820176),
            "N_transfer_H": (5.529646, 1.878196, -0.907186),
            "cat_N": (5.17325900, 0.04177700, -1.00370400),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_B", "B_transfer_H"), 1.656),
            ConstraintEntry("distance", ("cat_N", "N_transfer_H"), 1.961),
            ConstraintEntry("distance", ("cat_B", "cat_N"), 3.080),
            ConstraintEntry("angle", ("cat_B", "B_transfer_H", "cat_N"), 87.38739),
        ),
        constraint_order=("cat_B", "cat_N", "B_transfer_H", "N_transfer_H", "substrate_C"),
    ),
    "TS3": TSGuess2Spec(
        name="TS3",
        spec_id="TS3::builtin::methylpyrrole_tmp_v2",
        builder_key="ts3_ts4",
        core_smarts="[#5]~[#1]~[#5]~[#6]",
        role_coordinates={
            "cat_B": (1.201563, 0.080366, 0.660199),
            "transfer_H": (1.676962, -1.004507, -0.052686),
            "pin_B": (2.532308, -1.428336, 0.777578),
            "substrate_C": (1.976672, 0.248906, 2.068494),
        },
        constraints=(
            ConstraintEntry("distance", ("transfer_H", "cat_B"), 1.37581),
            ConstraintEntry("distance", ("transfer_H", "pin_B"), 1.26409),
            ConstraintEntry("distance", ("transfer_H", "substrate_C"), 2.47686),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.61593),
            ConstraintEntry("distance", ("pin_B", "substrate_C"), 2.17986),
            ConstraintEntry("distance", ("pin_B", "cat_B"), 2.00709),
            ConstraintEntry("angle", ("cat_B", "transfer_H", "pin_B"), 98.89),
            ConstraintEntry("angle", ("cat_B", "substrate_C", "pin_B"), 61.75),
        ),
        constraint_order=("cat_B", "pin_B", "transfer_H", "substrate_C"),
    ),
    "TS4": TSGuess2Spec(
        name="TS4",
        spec_id="TS4::builtin::methylpyrrole_tmp_v2",
        builder_key="ts3_ts4",
        core_smarts="[#5]~[#1]~[#5]~[#6]",
        role_coordinates={
            "cat_B": (-0.930038, 0.590384, 1.929793),
            "transfer_H": (-0.087884, 1.262005, 1.344826),
            "pin_B": (0.999483, 1.217369, 2.683538),
            "substrate_C": (0.013065, 0.446161, 3.676874),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_B", "pin_B"), 2.21926),
            ConstraintEntry("distance", ("pin_B", "transfer_H"), 1.86758),
            ConstraintEntry("distance", ("substrate_C", "transfer_H"), 2.48888),
            ConstraintEntry("distance", ("cat_B", "transfer_H"), 1.21598),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.94626),
            ConstraintEntry("distance", ("pin_B", "substrate_C"), 1.58475),
            ConstraintEntry("angle", ("cat_B", "transfer_H", "pin_B"), 89.48),
            ConstraintEntry("angle", ("cat_B", "substrate_C", "pin_B"), 77.13),
        ),
        constraint_order=("cat_B", "pin_B", "transfer_H", "substrate_C"),
    ),
}
