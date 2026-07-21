"""Typed models for method-aware ``tsguess2`` specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal, Mapping

from frust.tsguess.specs import ConstraintEntry


ConstraintKind = Literal["distance", "angle"]
SpecStatus = Literal["active", "candidate", "quarantined"]


@dataclass(frozen=True)
class ConstraintDef:
    """Define one named, method-independent internal-coordinate constraint.

    Parameters
    ----------
    name : str
        Stable identifier used by geometry profiles to provide the numerical
        value.
    kind : {"distance", "angle"}
        Constraint type. Distances require two roles and angles require three.
    roles : tuple of str
        Ordered chemical roles participating in the internal coordinate.
    """

    name: str
    kind: ConstraintKind
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = 2 if self.kind == "distance" else 3 if self.kind == "angle" else None
        if expected is None:
            raise ValueError(f"Unsupported constraint kind: {self.kind!r}")
        if len(self.roles) != expected:
            raise ValueError(
                f"{self.kind} constraint {self.name!r} requires {expected} roles"
            )
        if not self.name:
            raise ValueError("constraint names must be non-empty")


@dataclass(frozen=True)
class CoreTopology:
    """Describe method-independent construction and constraint topology.

    Parameters
    ----------
    state : str
        Structure state such as ``"TS4"`` or ``"INT3"``.
    builder_key : str
        Connected-SMILES builder family.
    core_smarts : str
        SMARTS used to recover chemical roles after the SMILES round trip.
    constraints : tuple of ConstraintDef
        Named internal coordinates used for constrained optimization.
    """

    state: str
    builder_key: str
    core_smarts: str
    constraints: tuple[ConstraintDef, ...]

    def __post_init__(self) -> None:
        names = [constraint.name for constraint in self.constraints]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate constraint names for {self.state}: {names}")


@dataclass(frozen=True)
class GeometryKey:
    """Identify an optimization method and environment.

    Parameters
    ----------
    method : str
        Canonical method family, for example ``"r2scan-3c"``.
    basis : str or None
        Canonical basis label. Composite methods use ``None``.
    solvation_model : str or None
        Solvation model, for example ``"smd"``. Gas-phase profiles use
        ``None``.
    solvent : str or None
        Solvent name. Gas-phase profiles use ``None``.
    """

    method: str
    basis: str | None = None
    solvation_model: str | None = None
    solvent: str | None = None

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        basis = None if self.basis is None else str(self.basis).strip().lower()
        model = (
            None
            if self.solvation_model is None
            else str(self.solvation_model).strip().lower()
        )
        solvent = None if self.solvent is None else str(self.solvent).strip().lower()
        if not method:
            raise ValueError("geometry method must be non-empty")
        if (model is None) != (solvent is None):
            raise ValueError("solvation_model and solvent must both be set or both be None")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "basis", basis or None)
        object.__setattr__(self, "solvation_model", model or None)
        object.__setattr__(self, "solvent", solvent or None)

    @property
    def environment(self) -> str:
        """Return the canonical environment label."""
        if self.solvation_model is None:
            return "gas"
        return f"{self.solvation_model}-{self.solvent}"

    @property
    def profile_id(self) -> str:
        """Return the public profile identifier."""
        return f"{self.method}/{self.environment}"


@dataclass(frozen=True)
class ReferenceRecord:
    """Describe the optimized reference used to derive a state geometry.

    Parameters
    ----------
    substrate_name, catalyst_name : str
        Reference chemical system.
    method : str
        Human-readable electronic-structure method.
    basis : str or None
        Basis used for the optimized geometry.
    solvation_model, solvent : str or None
        Reference environment.
    coordinates_column, vibrations_column : str
        Source dataframe columns used for extraction and validation.
    negative_frequencies : tuple of float or None
        Final negative frequencies in inverse centimetres. ``None`` means the
        migrated source did not preserve this audit information.
    mode_reviewed : bool or None
        Whether the relevant imaginary mode was visually reviewed. ``None``
        means the review status is unknown.
    source_sha256 : str or None
        SHA-256 of the source reference artifact when available.
    notes : str or None
        Additional audit note.
    """

    substrate_name: str
    catalyst_name: str
    method: str
    basis: str | None
    solvation_model: str | None
    solvent: str | None
    coordinates_column: str
    vibrations_column: str
    negative_frequencies: tuple[float, ...] | None = None
    mode_reviewed: bool | None = None
    source_sha256: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class StateGeometrySpec:
    """Store method-dependent geometry for one constrained state.

    Parameters
    ----------
    state : str
        State name matching a :class:`CoreTopology`.
    geometry_key : GeometryKey
        Method and environment represented by the geometry.
    revision : int
        Positive immutable revision number for this state/profile pair.
    role_coordinates : mapping
        Role names mapped to Cartesian coordinates used for embedding.
    constraint_values : mapping
        Named topology constraints mapped to numerical target values.
    reference : ReferenceRecord
        Provenance for the optimized reference.
    status : {"active", "candidate", "quarantined"}
        Selection status. Only active geometries are used by production
        resolution.
    """

    state: str
    geometry_key: GeometryKey
    revision: int
    role_coordinates: Mapping[str, tuple[float, float, float]]
    constraint_values: Mapping[str, float]
    reference: ReferenceRecord
    status: SpecStatus = "active"

    def __post_init__(self) -> None:
        if int(self.revision) < 1:
            raise ValueError("geometry revisions must be positive")
        coordinates: dict[str, tuple[float, float, float]] = {}
        for role, coordinate in self.role_coordinates.items():
            values = tuple(float(value) for value in coordinate)
            if len(values) != 3 or not all(isfinite(value) for value in values):
                raise ValueError(f"Invalid coordinate for role {role!r}: {coordinate!r}")
            coordinates[str(role)] = values
        constraint_values = {
            str(name): float(value) for name, value in self.constraint_values.items()
        }
        if not all(isfinite(value) for value in constraint_values.values()):
            raise ValueError(f"Non-finite constraint value in {self.state}")
        object.__setattr__(self, "role_coordinates", MappingProxyType(coordinates))
        object.__setattr__(
            self,
            "constraint_values",
            MappingProxyType(constraint_values),
        )

    @property
    def spec_id(self) -> str:
        """Return the immutable row-level specification identifier."""
        method = self.geometry_key.profile_id.replace("/", "::")
        return f"{self.state}::tsguess2-v2::{method}::r{self.revision}"


@dataclass(frozen=True)
class TSGuess2Spec:
    """Resolved TS guess instructions for the SMILES-roundtrip backend.

    Parameters
    ----------
    name : str
        Structure state.
    spec_id : str
        Immutable state/profile/revision identifier stored on generated rows.
    profile_id : str
        Public method/environment profile identifier.
    builder_key : str
        Connected-SMILES builder family.
    core_smarts : str
        SMARTS used to recover role atom indices.
    role_coordinates : mapping
        Role-coordinate anchors used by RDKit embedding.
    constraints : tuple of ConstraintEntry
        Resolved role-based constraint values.
    """

    name: str
    spec_id: str
    profile_id: str
    builder_key: str
    core_smarts: str
    role_coordinates: Mapping[str, tuple[float, float, float]]
    constraints: tuple[ConstraintEntry, ...]

    def constraint_dicts(self) -> list[dict[str, object]]:
        """Return constraints as dataframe-friendly dictionaries."""
        return [entry.as_dict() for entry in self.constraints]


@dataclass(frozen=True)
class SpecSelection:
    """Describe requested and resolved profiles for one state."""

    state: str
    requested_profile: str
    resolved_profile: str
    match: Literal["exact", "same_method_environment_fallback"]
    spec: TSGuess2Spec = field(repr=False)
