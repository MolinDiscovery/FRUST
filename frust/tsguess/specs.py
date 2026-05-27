"""Built-in transition-state geometry specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintEntry:
    """One role-based distance or angle constraint.

    Parameters
    ----------
    kind : str
        Constraint kind. Supported values are ``"distance"`` and ``"angle"``.
    roles : tuple of str
        Role names used by this constraint.
    value : float
        Distance in Angstrom or angle in degrees.
    """

    kind: str
    roles: tuple[str, ...]
    value: float

    def as_dict(self) -> dict[str, object]:
        """Return a dataframe-friendly representation."""
        return {"kind": self.kind, "roles": list(self.roles), "value": self.value}


@dataclass(frozen=True)
class TSSpec:
    """Built-in TS core geometry and role constraints.

    Parameters
    ----------
    name : str
        Structure type, for example ``"TS1"``.
    spec_id : str
        Versioned identifier for provenance.
    role_coordinates : dict
        Mapping from role name to Cartesian coordinate tuple.
    constraints : tuple of ConstraintEntry
        Role-level constraints rendered by :class:`frust.stepper.Stepper`.
    constraint_order : tuple of str
        Role order used to project legacy ``constraint_atoms``.
    extra_fragment : str or None
        Optional built-in fragment added during assembly.
    """

    name: str
    spec_id: str
    role_coordinates: dict[str, tuple[float, float, float]]
    constraints: tuple[ConstraintEntry, ...]
    constraint_order: tuple[str, ...]
    extra_fragment: str | None = None

    def constraint_dicts(self) -> list[dict[str, object]]:
        """Return constraints as dataframe-friendly dictionaries."""
        return [entry.as_dict() for entry in self.constraints]


BUILTIN_TS_SPECS: dict[str, TSSpec] = {
    "TS1": TSSpec(
        name="TS1",
        spec_id="TS1::builtin::methylpyrrole_v1",
        role_coordinates={
            "transfer_H": (-1.558624, 0.103600, 1.047895),
            "cat_B": (0.373318, -0.291503, 1.700016),
            "cat_N": (-2.416144, -1.101445, 0.730403),
            "substrate_C": (-0.659429, 0.986440, 1.328243),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_B", "transfer_H"), 2.07696),
            ConstraintEntry("distance", ("cat_N", "transfer_H"), 1.51270),
            ConstraintEntry("distance", ("transfer_H", "substrate_C"), 1.29095),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.68461),
            ConstraintEntry("distance", ("cat_B", "cat_N"), 3.06223),
            ConstraintEntry("angle", ("cat_N", "transfer_H", "substrate_C"), 170.1342),
            ConstraintEntry("angle", ("transfer_H", "substrate_C", "cat_B"), 87.4870),
        ),
        constraint_order=("cat_B", "cat_N", "transfer_H", "substrate_C"),
        extra_fragment="H",
    ),
    "TS2": TSSpec(
        name="TS2",
        spec_id="TS2::builtin::methylpyrrole_v1",
        role_coordinates={
            "cat_B": (4.505621, 2.687572, 0.441993),
            "cat_N": (5.173259, 0.041777, -1.003704),
            "cat_H": (5.416981, 2.966094, 1.179912),
            "transfer_H": (5.642044, 2.651080, -0.820176),
            "n_transfer_H": (5.529646, 1.878196, -0.907186),
            "substrate_C": (3.638645, 3.836721, -0.134114),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_B", "transfer_H"), 1.656),
            ConstraintEntry("distance", ("cat_N", "n_transfer_H"), 1.961),
            ConstraintEntry("distance", ("cat_B", "cat_N"), 3.080),
            ConstraintEntry("angle", ("cat_B", "transfer_H", "cat_N"), 86.58),
        ),
        constraint_order=("cat_B", "cat_N", "cat_H", "transfer_H", "n_transfer_H", "substrate_C"),
        extra_fragment="H2",
    ),
    "TS3": TSSpec(
        name="TS3",
        spec_id="TS3::builtin::methylpyrrole_tmp_v1",
        role_coordinates={
            "cat_B": (1.201563, 0.080366, 0.660199),
            "transfer_H": (1.676962, -1.004507, -0.052686),
            "cat_H": (1.583281, 0.889608, -0.137320),
            "pin_B": (2.532308, -1.428336, 0.777578),
            "substrate_C": (1.976672, 0.248906, 2.068494),
            "cat_N": (-0.976510, 1.970566, 0.015729),
        },
        constraints=(
            ConstraintEntry("distance", ("transfer_H", "cat_B"), 1.376),
            ConstraintEntry("distance", ("transfer_H", "pin_B"), 1.264),
            ConstraintEntry("distance", ("transfer_H", "substrate_C"), 2.477),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.616),
            ConstraintEntry("distance", ("pin_B", "substrate_C"), 2.180),
            ConstraintEntry("distance", ("pin_B", "cat_B"), 2.007),
            ConstraintEntry("angle", ("cat_B", "transfer_H", "pin_B"), 98.89),
            ConstraintEntry("angle", ("cat_B", "substrate_C", "pin_B"), 61.75),
        ),
        constraint_order=("cat_B", "cat_N", "cat_H", "pin_B", "transfer_H", "substrate_C"),
        extra_fragment="HBpin",
    ),
    "TS4": TSSpec(
        name="TS4",
        spec_id="TS4::builtin::methylpyrrole_tmp_v1",
        role_coordinates={
            "cat_N": (-3.611411, -0.705527, 1.302784),
            "cat_B": (-0.930038, 0.590384, 1.929793),
            "cat_H": (-1.848264, 1.263084, 2.280216),
            "transfer_H": (-0.087884, 1.262005, 1.344826),
            "pin_B": (0.999483, 1.217369, 2.683538),
            "substrate_C": (0.013065, 0.446161, 3.676874),
        },
        constraints=(
            ConstraintEntry("distance", ("cat_B", "pin_B"), 2.219),
            ConstraintEntry("distance", ("pin_B", "transfer_H"), 1.868),
            ConstraintEntry("distance", ("substrate_C", "transfer_H"), 2.489),
            ConstraintEntry("distance", ("cat_B", "transfer_H"), 1.216),
            ConstraintEntry("distance", ("cat_B", "substrate_C"), 1.946),
            ConstraintEntry("distance", ("pin_B", "substrate_C"), 1.585),
            ConstraintEntry("angle", ("cat_B", "transfer_H", "pin_B"), 89.48),
            ConstraintEntry("angle", ("cat_B", "substrate_C", "pin_B"), 77.13),
        ),
        constraint_order=("cat_B", "cat_N", "cat_H", "transfer_H", "pin_B", "substrate_C"),
        extra_fragment="HBpin",
    ),
}
