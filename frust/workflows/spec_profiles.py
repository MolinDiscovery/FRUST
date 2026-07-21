"""Resolve structure-reference profiles from workflow calculator plans."""

from __future__ import annotations

import re

from frust.tsguess2.models import GeometryKey
from frust.workflows.methods import CalculatorSpec, MethodPlan


def profile_for_geometry_stage(method: MethodPlan, stage_id: str) -> str:
    """Return the tsguess2 profile requested by a geometry stage.

    Parameters
    ----------
    method : MethodPlan
        Workflow calculator plan.
    stage_id : str
        Geometry stage, normally ``"dft_ts_opt"`` or ``"dft_opt"``.

    Returns
    -------
    str
        Canonical method/environment profile identifier.
    """
    return geometry_key_for_calculator(method.for_stage(stage_id)).profile_id


def geometry_key_for_calculator(spec: CalculatorSpec) -> GeometryKey:
    """Translate semantic calculator metadata to a tsguess2 geometry key."""
    if spec.engine != "orca" or not spec.method:
        raise ValueError(
            "Automatic tsguess2 profile selection requires an ORCA geometry "
            "stage with semantic method metadata; pass spec_profile explicitly"
        )
    method = _canonical_method(spec.method, spec.basis)
    return GeometryKey(
        method=method,
        basis=spec.basis,
        solvation_model=spec.solvation_model,
        solvent=spec.solvent,
    )


def _canonical_method(method: str, basis: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(method).lower())
    basis_text = re.sub(r"[^a-z0-9]+", "", str(basis or "").lower())
    if text == "r2scan3c":
        return "r2scan-3c"
    if text in {"wb97xd3", "wb97xd3bj"} and basis_text in {"631g", "631gss"}:
        return "wb97xd3-631g"
    raise ValueError(
        f"No automatic tsguess2 profile mapping for method={method!r}, basis={basis!r}; "
        "pass spec_profile explicitly"
    )
