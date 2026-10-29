"""Common selected-reaction and named-branch analysis for portable runs."""

from __future__ import annotations
import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pandas as pd

from frust.workflows.bisarylation import _component_rows
from frust.workflows.expressions import evaluate_expression, reaction_correction
from frust.workflows.reactions import difference
from frust.workflows.connections import connection_status


def _connection_records(root):
    path = Path(root) / "analysis/connections.json"
    return json.loads(path.read_text()).get("records", []) if path.is_file() else []


def _resolved_connection(root, states, case, reaction, records, expression_roles=None):
    matches = [
        r
        for r in records
        if r["system_signature"] == case["system_signature"]
        and r["reaction_id"] == reaction["reaction_id"]
    ]
    record = matches[-1] if matches else None
    if record is None:
        return "unreviewed", None
    if (
        record.get("decision") != "approved"
        or record.get("method_fingerprint") != case["method_fingerprint"]
        or record.get("definition_version") != reaction["schema_version"]
        or record.get("saddle_target_id") != case["bindings"].get(reaction["ts_state"])
        or record.get("definition_fingerprint")
        != hashlib.sha256(json.dumps(reaction, sort_keys=True).encode()).hexdigest()
    ):
        return "stale", record
    status = connection_status(
        states,
        bindings=case["bindings"],
        result_ids=[
            *(
                (role, result_id)
                for role, result_id in record["reference_result_ids"].items()
                if expression_roles is None or role in expression_roles
            ),
            *record["endpoint_result_ids"].items(),
        ],
        saddle_role=reaction["ts_state"],
        saddle_result_id=record["saddle_result_id"],
        evidence=record["evidence"],
        root=root,
        protocol_identity=record.get("energy_protocol_fingerprint"),
    )
    return status, record


def reaction_tables(
    root, states, manifest, dimers, *, calculation_level, candidates=False
):
    """Evaluate each expression independently; retain every raw candidate."""
    from frust.workflows.bisarylation_analysis import _case_dimer, _analysis_case

    quantity = "gibbs" if calculation_level == "full" else "electronic"
    rows, profiles = [], []
    records = _connection_records(root)
    for original in manifest["reaction_plan"]["cases"]:
        case = deepcopy(original)
        aliases = states.attrs.get("frust_identity_aliases", {})
        case["bindings"] = {
            role: aliases.get(identity, identity)
            for role, identity in case["bindings"].items()
        }
        case.setdefault("reviewed_result_ids", {})
        # Legacy pathway decisions are an explicit adapter, not chemistry inference.
        if case["network_id"] == "bisarylation":
            case = _analysis_case(states, case, root)
        local = states
        if "dimer" in case["bindings"] and not dimers.empty:
            local, case = _case_dimer(states, case, dimers)
        selected = _component_rows(
            local, case, quantity=quantity, result_ids=case.get("reviewed_result_ids")
        )
        profile_selected = dict(selected)
        for reaction in case["reactions"]:
            ts = reaction["ts_state"]
            status, record = _resolved_connection(
                root, local, case, reaction, records, expression_roles=set()
            )
            base = dict(selected)
            if record and status == "approved":
                base = _component_rows(
                    local,
                    case,
                    quantity=quantity,
                    result_ids={
                        **record["reference_result_ids"],
                        ts: record["saddle_result_id"],
                    },
                )
            legacy_minima = case.get("reviewed_minimum_result_ids", {}).get(ts, {})
            if legacy_minima:
                base = _component_rows(
                    local,
                    case,
                    quantity=quantity,
                    result_ids={
                        **case.get("reviewed_result_ids", {}),
                        **case.get("reviewed_connection_result_ids", {}),
                        **legacy_minima,
                    },
                )
            current = base.get(ts)
            current_id = current.result_id if current is not None else None
            choices = (
                local.loc[local.structure_id.eq(case["bindings"].get(ts))]
                if candidates
                else [current]
            )
            if candidates:
                choices = [r for _, r in choices.iterrows()]
            for saddle in choices:
                chosen = dict(base)
                saddle_id = saddle.result_id if saddle is not None else None
                if candidates:
                    chosen = dict(selected)
                    chosen[ts] = saddle
                    if saddle_id == current_id:
                        chosen.update(base)
                connection = status
                if record and saddle_id != record["saddle_result_id"]:
                    connection = "unreviewed_candidate"
                legacy_status = case.get("connection_statuses", {}).get(ts)
                if record is None and legacy_status is not None:
                    connection = legacy_status
                if legacy_minima and saddle_id != case.get(
                    "reviewed_connection_result_ids", {}
                ).get(ts, case.get("reviewed_result_ids", {}).get(ts)):
                    connection = "unreviewed_candidate"
                    legacy_status = connection
                gate = (record is not None and connection != "approved") or (
                    legacy_status is not None and legacy_status != "approved"
                )
                if reaction["requires_connection"] and not (
                    connection == "approved"
                    or (
                        record is None
                        and case.get("pathway_review_status") == "approved"
                        and case.get("reviewed_result_ids", {}).get(ts) == saddle_id
                    )
                ):
                    gate = True
                correction = reaction_correction(
                    reaction,
                    "forward_barrier",
                    manifest.get("g_corrections_kcal_mol", {}),
                )
                values = {}
                expression_statuses = {}
                original_chosen = chosen
                for name, terms in reaction["expressions"].items():
                    expression_status = connection
                    if record is not None and connection == "approved":
                        expression_status = _resolved_connection(
                            root,
                            local,
                            case,
                            reaction,
                            records,
                            expression_roles=set(terms),
                        )[0]
                    expression_gate = gate or (
                        record is not None and expression_status != "approved"
                    )
                    expression_chosen = dict(original_chosen)
                    if expression_gate and expression_chosen.get(ts) is not None:
                        expression_chosen[ts] = expression_chosen[ts].copy()
                        if expression_chosen[ts].quality_status == "ready":
                            expression_chosen[ts]["quality_status"] = "review"
                    values[name] = evaluate_expression(
                        expression_chosen,
                        terms,
                        quantity=quantity,
                        correction=reaction_correction(
                            reaction, name, manifest.get("g_corrections_kcal_mol", {})
                        ),
                        allow_review=reaction.get("report_review_energy", False)
                        and not expression_gate,
                    )
                    expression_statuses[name] = expression_status
                    if name == "forward_barrier":
                        chosen = expression_chosen
                        forward_gate = expression_gate
                gate = forward_gate
                connection = expression_statuses["forward_barrier"]
                reaction_energy = values.get("reaction_energy", {})
                forward = values["forward_barrier"]
                reverse = values.get("reverse_barrier", {})
                elevation = (
                    evaluate_expression(
                        chosen,
                        difference(
                            {ts: 1.0}, case.get("profile_terms", {}).get("Dimer", {})
                        ),
                        quantity=quantity,
                        correction=float(
                            manifest.get("g_corrections_kcal_mol", {}).get(ts, 0.0)
                        )
                        - float(
                            manifest.get("g_corrections_kcal_mol", {}).get("Dimer", 0.0)
                        ),
                    )
                    if "Dimer" in case.get("profile_terms", {})
                    else {}
                )
                row = dict(
                    system_name=case["system_name"],
                    substrate_name=case["substrate_name"],
                    catalyst_name=case["catalyst_name"],
                    system_signature=case["system_signature"],
                    rpos=case["rpos"],
                    network_id=case["network_id"],
                    reaction_id=reaction["reaction_id"],
                    ts_type=ts,
                    ts_result_id=saddle_id,
                    ts_cid=saddle.get("cid") if saddle is not None else None,
                    variant=case["variant"],
                    connection_status=connection,
                    comparison_kind=(
                        "verified_connection"
                        if connection == "approved"
                        else "thermodynamic"
                    ),
                    pathway_review_status=case.get(
                        "pathway_review_status", "unreviewed"
                    ),
                    reference_state="+".join(reaction["reactants"]),
                    product_state="+".join(reaction["products"]),
                    reference_result_ids=json.dumps(
                        forward["reference_result_ids"], sort_keys=True
                    ),
                    product_result_ids=json.dumps(
                        reverse.get("reference_result_ids", {}), sort_keys=True
                    ),
                    ts_energy_protocol_fingerprint=(
                        saddle.get("energy_protocol_fingerprint")
                        if saddle is not None
                        else None
                    ),
                    ts_guess_profile=manifest.get(
                        "resolved_ts_spec_profile", manifest.get("ts_spec_profile")
                    ),
                    ts_guess_match=manifest.get("ts_spec_match"),
                    quality_status=forward["quality_status"],
                    quality_issues=forward["quality_issues"]
                    + (";" + connection if gate else ""),
                    delta_e_kcal_mol=forward["relative_e_kcal_mol"],
                    delta_g_kcal_mol=forward["relative_g_kcal_mol"],
                    delta_g_corrected_kcal_mol=forward["relative_g_corrected_kcal_mol"],
                    g_correction_kcal_mol=correction,
                    reverse_delta_e_kcal_mol=reverse.get("relative_e_kcal_mol"),
                    reverse_delta_g_corrected_kcal_mol=reverse.get(
                        "relative_g_corrected_kcal_mol"
                    ),
                    reverse_delta_g_kcal_mol=reverse.get("relative_g_kcal_mol"),
                    reaction_delta_e_kcal_mol=reaction_energy.get(
                        "relative_e_kcal_mol"
                    ),
                    reaction_delta_g_corrected_kcal_mol=reaction_energy.get(
                        "relative_g_corrected_kcal_mol"
                    ),
                    reaction_delta_g_kcal_mol=reaction_energy.get(
                        "relative_g_kcal_mol"
                    ),
                    reaction_energy_quality_status=reaction_energy.get(
                        "quality_status", "not_requested"
                    ),
                    reverse_connection_status=expression_statuses.get(
                        "reverse_barrier", "not_requested"
                    ),
                    reverse_quality_status=reverse.get(
                        "quality_status", "not_requested"
                    ),
                    dimer_relative_e_kcal_mol=elevation.get("relative_e_kcal_mol"),
                    dimer_elevation_quality_status=elevation.get(
                        "quality_status", "not_requested"
                    ),
                    dimer_result_id=case.get("dimer_result_id"),
                    dimer_state_id=case.get("dimer_state_id"),
                    dimer_reference_quality=case.get("dimer_reference_quality"),
                    formula_id=reaction["reaction_id"]
                    + "::v"
                    + str(reaction["schema_version"]),
                )
                if case.get("parent_uma_result_id") is not None:
                    row["parent_uma_result_id"] = case["parent_uma_result_id"]
                if not candidates:
                    profile_selected[ts] = chosen.get(ts)
                if saddle is not None:
                    row.update(
                        {
                            k: saddle.get(k)
                            for k in (
                                "method_family",
                                "energy_method",
                                "energy_model",
                                "solvation_model",
                                "solvent",
                                "energy_protocol_fingerprint",
                            )
                        }
                    )
                if pd.isna(row["delta_e_kcal_mol"]):
                    row["energy_protocol_fingerprint"] = None
                if candidates:
                    row.update(
                        cid=saddle.get("cid"),
                        selected=saddle_id == current_id,
                        ts_review_status=saddle.get("review_status"),
                        ts_quality_issues=saddle.get("quality_issues"),
                        n_imag=saddle.get("n_imag"),
                    )
                rows.append(row)
        if candidates:
            continue
        selected = profile_selected
        for role, applicability in case["applicability"].items():
            if applicability == "discovery" and selected.get(role) is not None:
                selected[role] = selected[role].copy()
                if selected[role].quality_status == "ready":
                    selected[role]["quality_status"] = "review"
        corrections = dict(manifest.get("g_corrections_kcal_mol", {}))
        branches = []
        if case.get("profile_terms"):
            branches.append(
                (
                    case["network_id"],
                    case["profile_terms"],
                    case.get("profile_order", list(case["profile_terms"])),
                    case["reference"],
                )
            )
        elif manifest.get("scope") == "full_cycle":
            for reaction in case["reactions"]:
                ts = reaction["ts_state"]
                expressions = reaction["expressions"]
                if "reverse_barrier" not in expressions:
                    continue
                left = difference({ts: 1.0}, expressions["forward_barrier"])
                right = difference({ts: 1.0}, expressions["reverse_barrier"])
                reference = ts + ":reactants"
                corrections[reference] = corrections.get(
                    reaction.get("screening_reference_label")
                    or reaction.get("reactant_label"),
                    0.0,
                )
                corrections[ts + ":products"] = corrections.get(
                    reaction.get("product_label"), 0.0
                )
                terms = {reference: left, ts: {ts: 1.0}, ts + ":products": right}
                branches.append(
                    (reaction["reaction_id"], terms, list(terms), reference)
                )
        for profile_branch, terms, order_labels, reference in branches:
            for order, label in enumerate(order_labels):
                node = terms[label]
                value = evaluate_expression(
                    selected,
                    difference(node, terms[reference]),
                    quantity=quantity,
                    allow_review=all(
                        r.get("report_review_energy", False) for r in case["reactions"]
                    ),
                    correction=corrections.get(label, 0.0)
                    - corrections.get(reference, 0.0),
                )
                if label == reference:
                    from frust.screen.runs import QUALITY_ORDER

                    qualities = [
                        (
                            selected[r].quality_status
                            if selected.get(r) is not None
                            else "incomplete"
                        )
                        for r in node
                    ]
                    quality = max(qualities or ["incomplete"], key=QUALITY_ORDER.get)
                    value.update(
                        quality_status=quality,
                        relative_e_kcal_mol=(
                            0.0 if quality in {"ready", "review"} else float("nan")
                        ),
                        relative_g_kcal_mol=(
                            0.0 if quality in {"ready", "review"} else float("nan")
                        ),
                        relative_g_corrected_kcal_mol=(
                            0.0 if quality in {"ready", "review"} else float("nan")
                        ),
                    )
                value["reference_result_ids"] = json.dumps(
                    value["reference_result_ids"], sort_keys=True
                )
                value["result_ids"] = json.dumps(
                    {
                        role: selected[role].result_id
                        for role in node
                        if selected.get(role) is not None
                    },
                    sort_keys=True,
                )
                profiles.append(
                    dict(
                        system_name=case["system_name"],
                        rpos=case["rpos"],
                        network_id=case["network_id"],
                        variant=case["variant"],
                        profile_state=label,
                        reference=reference,
                        profile_branch=profile_branch,
                        profile_order=order,
                        applicability=case["applicability"].get(label, "required"),
                        mechanism_id=case.get("profile_mechanism_id")
                        or case["network_id"] + "::v1",
                        **value,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(profiles)
