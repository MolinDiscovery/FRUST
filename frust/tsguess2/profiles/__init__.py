"""Registry and resolution for built-in ``tsguess2`` geometry profiles."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from frust.tsguess.specs import ConstraintEntry
from frust.tsguess2.models import (
    GeometryKey,
    SpecSelection,
    StateGeometrySpec,
    TSGuess2Spec,
)
from frust.tsguess2.profiles.r2scan3c_smd_chloroform import (
    GEOMETRIES as R2SCAN3C_SMD_GEOMETRIES,
    GEOMETRY_KEY as R2SCAN3C_SMD_KEY,
    QUARANTINED_STATES as R2SCAN3C_SMD_QUARANTINED,
)
from frust.tsguess2.profiles.wb97xd3_631g_gas import (
    GEOMETRIES as WB97_GAS_GEOMETRIES,
    GEOMETRY_KEY as WB97_GAS_KEY,
)
from frust.tsguess2.topologies import CORE_TOPOLOGIES


WB97_SMD_KEY = GeometryKey(
    method="wb97xd3-631g",
    basis="6-31g**",
    solvation_model="smd",
    solvent="chloroform",
)
R2SCAN3C_GAS_KEY = GeometryKey(method="r2scan-3c")

PROFILE_KEYS: dict[str, GeometryKey] = {
    key.profile_id: key
    for key in (WB97_GAS_KEY, WB97_SMD_KEY, R2SCAN3C_GAS_KEY, R2SCAN3C_SMD_KEY)
}

_GEOMETRIES: dict[str, dict[str, StateGeometrySpec]] = {
    WB97_GAS_KEY.profile_id: WB97_GAS_GEOMETRIES,
    R2SCAN3C_SMD_KEY.profile_id: R2SCAN3C_SMD_GEOMETRIES,
}
_QUARANTINED: dict[str, dict[str, str]] = {
    R2SCAN3C_SMD_KEY.profile_id: R2SCAN3C_SMD_QUARANTINED,
}


def resolve_profile_spec(
    state: str,
    profile: str,
    *,
    match: str = "prefer-exact",
) -> SpecSelection:
    """Resolve one state against a method/environment geometry profile.

    Parameters
    ----------
    state : str
        State such as ``"TS1"`` or ``"INT3"``.
    profile : str
        Requested profile identifier.
    match : {"prefer-exact", "exact"}, optional
        ``"prefer-exact"`` permits the other environment of the same method
        when the exact state geometry is unavailable. It never crosses method
        families. ``"exact"`` disables fallback.

    Returns
    -------
    SpecSelection
        Requested/resolved profile metadata and resolved runtime spec.
    """
    state_key = str(state).upper()
    profile_id = normalize_profile_id(profile)
    if match not in {"prefer-exact", "exact"}:
        raise ValueError("match must be 'prefer-exact' or 'exact'")

    geometry = _active_geometry(profile_id, state_key)
    if geometry is not None:
        return SpecSelection(
            state=state_key,
            requested_profile=profile_id,
            resolved_profile=profile_id,
            match="exact",
            spec=_combine(geometry),
        )

    if match == "prefer-exact":
        requested_key = PROFILE_KEYS[profile_id]
        for candidate_id, candidate_key in PROFILE_KEYS.items():
            if candidate_id == profile_id or candidate_key.method != requested_key.method:
                continue
            geometry = _active_geometry(candidate_id, state_key)
            if geometry is not None:
                return SpecSelection(
                    state=state_key,
                    requested_profile=profile_id,
                    resolved_profile=candidate_id,
                    match="same_method_environment_fallback",
                    spec=_combine(geometry),
                )

    detail = _unavailable_detail(profile_id, state_key)
    raise ValueError(
        f"No selectable {state_key} geometry for profile {profile_id!r}. {detail}"
    )


def resolve_profile_specs(
    states: Iterable[str],
    profile: str,
    *,
    match: str = "prefer-exact",
) -> dict[str, SpecSelection]:
    """Resolve all requested states before structure generation starts."""
    return {
        str(state).upper(): resolve_profile_spec(state, profile, match=match)
        for state in states
    }


def normalize_profile_id(profile: str) -> str:
    """Normalize and validate a public profile identifier."""
    value = str(profile).strip().lower().replace("_", "-")
    aliases = {
        "wb97xd3-631g-gas": WB97_GAS_KEY.profile_id,
        "wb97xd3-631g": WB97_GAS_KEY.profile_id,
        "wb97xd3-631g-solv": WB97_SMD_KEY.profile_id,
        "r2scan3c": R2SCAN3C_GAS_KEY.profile_id,
        "r2scan3c-gas": R2SCAN3C_GAS_KEY.profile_id,
        "r2scan3c-solv": R2SCAN3C_SMD_KEY.profile_id,
        "r2scan-3c": R2SCAN3C_GAS_KEY.profile_id,
        "r2scan-3c-solv": R2SCAN3C_SMD_KEY.profile_id,
        "wb97xd3-631g/smd(chloroform)": WB97_SMD_KEY.profile_id,
        "r2scan-3c/smd(chloroform)": R2SCAN3C_SMD_KEY.profile_id,
    }
    value = aliases.get(value, value)
    if value not in PROFILE_KEYS:
        available = ", ".join(sorted(PROFILE_KEYS))
        raise ValueError(f"Unknown tsguess2 profile {profile!r}; expected one of {available}")
    return value


def show_spec_profiles() -> pd.DataFrame:
    """Return state-level availability for all built-in geometry profiles."""
    rows: list[dict[str, object]] = []
    for profile_id, key in PROFILE_KEYS.items():
        geometries = _GEOMETRIES.get(profile_id, {})
        quarantined = _QUARANTINED.get(profile_id, {})
        for state in CORE_TOPOLOGIES:
            geometry = geometries.get(state)
            fallback_profile = _same_method_fallback_profile(profile_id, state)
            if geometry is not None:
                status = geometry.status
                spec_id = geometry.spec_id
                note = geometry.reference.notes
                reference_catalyst = geometry.reference.catalyst_name
                negative_frequencies = geometry.reference.negative_frequencies
                mode_reviewed = geometry.reference.mode_reviewed
                source_sha256 = geometry.reference.source_sha256
            elif state in quarantined:
                status = "quarantined"
                spec_id = None
                note = quarantined[state]
                reference_catalyst = None
                negative_frequencies = None
                mode_reviewed = False
                source_sha256 = None
            else:
                status = "missing"
                spec_id = None
                note = None
                reference_catalyst = None
                negative_frequencies = None
                mode_reviewed = None
                source_sha256 = None
            rows.append(
                {
                    "profile": profile_id,
                    "method": key.method,
                    "environment": key.environment,
                    "state": state,
                    "status": status,
                    "selectable_prefer_exact": status == "active" or fallback_profile is not None,
                    "fallback_profile": fallback_profile,
                    "spec_id": spec_id,
                    "reference_catalyst": reference_catalyst,
                    "negative_frequencies": negative_frequencies,
                    "mode_reviewed": mode_reviewed,
                    "source_sha256": source_sha256,
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def _same_method_fallback_profile(profile_id: str, state: str) -> str | None:
    """Return an active same-method fallback profile for one state."""
    requested_key = PROFILE_KEYS[profile_id]
    for candidate_id, candidate_key in PROFILE_KEYS.items():
        if candidate_id == profile_id or candidate_key.method != requested_key.method:
            continue
        if _active_geometry(candidate_id, state) is not None:
            return candidate_id
    return None


def _active_geometry(profile_id: str, state: str) -> StateGeometrySpec | None:
    geometry = _GEOMETRIES.get(profile_id, {}).get(state)
    if geometry is None or geometry.status != "active":
        return None
    return geometry


def _combine(geometry: StateGeometrySpec) -> TSGuess2Spec:
    topology = CORE_TOPOLOGIES[geometry.state]
    expected = {constraint.name for constraint in topology.constraints}
    actual = set(geometry.constraint_values)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Geometry {geometry.spec_id} does not match {geometry.state} topology; "
            f"missing={missing}, extra={extra}"
        )
    required_roles = {
        role for constraint in topology.constraints for role in constraint.roles
    }
    missing_coordinates = sorted(required_roles - set(geometry.role_coordinates))
    if missing_coordinates:
        raise ValueError(
            f"Geometry {geometry.spec_id} lacks coordinates for roles "
            f"{missing_coordinates}"
        )
    for constraint in topology.constraints:
        value = geometry.constraint_values[constraint.name]
        if constraint.kind == "distance" and value <= 0:
            raise ValueError(f"Distance {constraint.name!r} must be positive")
        if constraint.kind == "angle" and not 0 < value <= 180:
            raise ValueError(f"Angle {constraint.name!r} must be in (0, 180]")
    constraints = tuple(
        ConstraintEntry(
            constraint.kind,
            constraint.roles,
            geometry.constraint_values[constraint.name],
        )
        for constraint in topology.constraints
    )
    return TSGuess2Spec(
        name=geometry.state,
        spec_id=geometry.spec_id,
        profile_id=geometry.geometry_key.profile_id,
        builder_key=topology.builder_key,
        core_smarts=topology.core_smarts,
        role_coordinates=geometry.role_coordinates,
        constraints=constraints,
    )


def _unavailable_detail(profile_id: str, state: str) -> str:
    quarantined = _QUARANTINED.get(profile_id, {})
    if state in quarantined:
        return f"The reference is quarantined: {quarantined[state]}"
    requested_key = PROFILE_KEYS[profile_id]
    for candidate_id, candidate_key in PROFILE_KEYS.items():
        if candidate_key.method != requested_key.method:
            continue
        candidate_quarantine = _QUARANTINED.get(candidate_id, {})
        if state in candidate_quarantine:
            return (
                f"The same-method {candidate_id!r} reference is quarantined: "
                f"{candidate_quarantine[state]}"
            )
    return "The state/profile reference has not been supplied or activated."


__all__ = [
    "PROFILE_KEYS",
    "R2SCAN3C_GAS_KEY",
    "R2SCAN3C_SMD_KEY",
    "WB97_GAS_KEY",
    "WB97_SMD_KEY",
    "normalize_profile_id",
    "resolve_profile_spec",
    "resolve_profile_specs",
    "show_spec_profiles",
]
