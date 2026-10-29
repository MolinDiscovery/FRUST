"""Shared balanced energy evaluation with per-quantity dependencies."""

from collections import Counter
from collections.abc import Mapping
import pandas as pd


def evaluate_expression(
    selected, coefficients, *, quantity="electronic", correction=0.0, allow_review=False
):
    """Cancel spectators before checking dependencies and evaluating a barrier."""
    from frust.screen.runs import HARTREE_TO_KCAL_MOL, QUALITY_ORDER, _parse_formula

    coefficients = {r: c for r, c in coefficients.items() if abs(c) > 1e-12}
    rows, issues, qualities, protocols, inventory = {}, [], [], set(), Counter()
    for role, coefficient in coefficients.items():
        row = selected.get(role)
        if row is None:
            qualities.append("incomplete")
            issues.append("missing:" + role)
            continue
        rows[role] = row
        qualities.append(str(row.quality_status))
        if row.quality_status != "ready":
            issues.append(str(row.quality_status) + ":" + role)
            issues.append("dependency_" + str(row.quality_status) + ":" + role)
        protocol = row.get("energy_protocol_fingerprint")
        if protocol is None or pd.isna(protocol):
            qualities.append("incomplete")
            issues.append("missing_energy_protocol:" + role)
        else:
            protocols.add(protocol)
        formula = (
            row.formula
            if isinstance(row.formula, Mapping)
            else _parse_formula(str(row.formula))
        )
        for atom, count in formula.items():
            inventory[atom] += coefficient * count
        if pd.isna(row.electronic_energy_hartree) or (
            quantity == "gibbs"
            and pd.isna(row.get("free_energy_hartree", float("nan")))
        ):
            qualities.append("incomplete")
            issues.append("missing_energy:" + role)
    if len(protocols) > 1:
        qualities.append("invalid")
        issues.append("incompatible_energy_protocols")
        issues.append("mixed_electronic_energy_protocols")
    if len(rows) == len(coefficients) and any(
        abs(v) > 1e-8 for v in inventory.values()
    ):
        qualities.append("invalid")
        issues.append("unbalanced_composition")
    quality = max(qualities or ["incomplete"], key=QUALITY_ORDER.get)
    report = quality == "ready" or (
        allow_review
        and quality in {"review", "invalid"}
        and all(
            i.startswith(
                ("review:", "invalid:", "dependency_review:", "dependency_invalid:")
            )
            for i in issues
        )
    )
    electronic_report = report or (
        allow_review
        and len(rows) == len(coefficients)
        and all(not pd.isna(row.electronic_energy_hartree) for row in rows.values())
        and len(protocols) == 1
        and not any(abs(v) > 1e-8 for v in inventory.values())
    )
    e = sum(coefficients[r] * row.electronic_energy_hartree for r, row in rows.items())
    g = sum(
        coefficients[r] * row.get("free_energy_hartree", float("nan"))
        for r, row in rows.items()
    )
    return dict(
        quality_status=quality,
        quality_issues=";".join(dict.fromkeys(issues)),
        relative_e_kcal_mol=(
            e * HARTREE_TO_KCAL_MOL if electronic_report else float("nan")
        ),
        relative_g_kcal_mol=g * HARTREE_TO_KCAL_MOL if report else float("nan"),
        relative_g_corrected_kcal_mol=(
            g * HARTREE_TO_KCAL_MOL + correction if report else float("nan")
        ),
        g_correction_kcal_mol=correction,
        reference_result_ids={
            r: row.result_id for r, row in rows.items() if coefficients[r] < 0
        },
    )


def reaction_correction(reaction, expression, corrections):
    """Apply a declared node correction once to its balanced expression."""
    ts = reaction["ts_state"]
    left = reaction.get("reactant_label")
    right = reaction.get("product_label")
    if expression == "forward_barrier":
        reference = reaction.get("screening_reference_label") or left
        if reference is not None:
            return float(corrections.get(ts, 0.0)) - float(
                corrections.get(reference, 0.0)
            )
    elif expression == "reverse_barrier" and right is not None:
        return float(corrections.get(ts, 0.0)) - float(corrections.get(right, 0.0))
    elif expression == "reaction_energy" and left is not None and right is not None:
        return float(corrections.get(right, 0.0)) - float(corrections.get(left, 0.0))
    return sum(
        float(coefficient) * float(corrections.get(role, 0.0))
        for role, coefficient in reaction["expressions"][expression].items()
    )
