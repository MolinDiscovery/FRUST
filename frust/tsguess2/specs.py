"""Public specification facade for the method-aware ``tsguess2`` backend."""

from __future__ import annotations

from frust.tsguess2.models import (
    ConstraintDef,
    CoreTopology,
    GeometryKey,
    ReferenceRecord,
    SpecSelection,
    StateGeometrySpec,
    TSGuess2Spec,
)
from frust.tsguess2.profiles import (
    WB97_GAS_KEY,
    normalize_profile_id,
    resolve_profile_spec,
    resolve_profile_specs,
    show_spec_profiles,
)
from frust.tsguess2.topologies import CORE_TOPOLOGIES


BUILTIN_TS_SPECS_V2: dict[str, TSGuess2Spec] = {
    state: resolve_profile_spec(
        state,
        WB97_GAS_KEY.profile_id,
        match="exact",
    ).spec
    for state in CORE_TOPOLOGIES
}


__all__ = [
    "BUILTIN_TS_SPECS_V2",
    "CORE_TOPOLOGIES",
    "ConstraintDef",
    "CoreTopology",
    "GeometryKey",
    "ReferenceRecord",
    "SpecSelection",
    "StateGeometrySpec",
    "TSGuess2Spec",
    "normalize_profile_id",
    "resolve_profile_spec",
    "resolve_profile_specs",
    "show_spec_profiles",
]
