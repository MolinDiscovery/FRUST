"""Lightweight selected-reaction dependency expansion for every chemistry."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

import pandas as pd

from frust.structures.roles import effective_bound_role
from frust.structures.planner import normalize_systems, plan_targets
from frust.structures.specs import DIMER_STATES
from frust.tsguess.matching import parse_rpos_value
from frust.workflows.reactions import NETWORKS, difference


def plan_reactions(
    systems,
    *,
    ts_types,
    scope,
    method_fingerprint,
    dimer_reference="lowest",
    pathway_reviews=None,
    dimer_pairing="input",
):
    """Expand only requested reactions and the terms of requested quantities."""
    selected = tuple(dict.fromkeys(str(s).upper() for s in ts_types))
    known = {r.ts_state for n in NETWORKS.values() for r in n.reactions}
    if not selected or set(selected) - known:
        raise ValueError(
            f"Expected nonempty registered TS selection; unknown: {sorted(set(selected)-known)}"
        )
    if scope not in {"barriers", "full_cycle"}:
        raise ValueError("Unknown coverage scope")
    if dimer_reference not in {"lowest", *DIMER_STATES}:
        raise ValueError("Unknown dimer reference")
    dimers = DIMER_STATES if dimer_reference == "lowest" else (dimer_reference,)
    targets, cases, reasons, groups = {}, [], {}, {}
    reviews = pathway_reviews or {}
    for _, row in normalize_systems(systems).iterrows():
        system = row.to_dict()
        for site in parse_rpos_value(system.get("rpos"), system["substrate_smiles"]):
            for network in NETWORKS.values():
                reactions = [r for r in network.reactions if r.ts_state in selected]
                if not reactions:
                    continue
                if dimer_pairing != "input" and not network.supports_racemic:
                    raise ValueError(
                        f"{network.network_id} does not support racemic dimer pairing"
                    )
                roles = {
                    role: value for r in reactions for role, value in r.roles.items()
                }
                bound, bound_site = (
                    effective_bound_role(system, site)
                    if any(v[1] == "bound" for v in roles.values())
                    else (system["substrate_smiles"], site)
                )
                payload = dict(
                    catalyst_smiles=system["catalyst_smiles"],
                    incoming_substrate_smiles=system["substrate_smiles"],
                    incoming_rpos=int(site),
                    bound_substrate_smiles=bound,
                    bound_rpos=bound_site,
                )
                signature = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                review = reviews.get(signature)
                variant = (
                    review.get("variant", network.default_variant)
                    if review
                    else network.default_variant
                )
                if review and variant != "unresolved":
                    if (
                        review.get("system_signature") != signature
                        or review.get("method_fingerprint") != method_fingerprint
                    ):
                        raise ValueError(
                            "Pathway hypothesis does not match chemical system/method"
                        )
                    hashes = review.get("evidence_sha256", [])
                    if not hashes or any(
                        len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                        for h in hashes
                    ):
                        raise ValueError("Resolved hypothesis requires evidence hashes")
                controlled = (
                    set().union(*(set(v) for v in network.alternatives.values()))
                    if network.alternatives
                    else set()
                )
                allowed = (set(roles) - controlled) | set(
                    network.alternatives.get(variant, roles)
                )
                if network.alternatives and variant not in network.alternatives:
                    raise ValueError("Unknown pathway alternative")
                records = []
                for reaction in reactions:
                    if reaction.ts_state not in allowed:
                        continue
                    definition = asdict(reaction)
                    # Alternatives change chemistry via recipe data; no scheduler branch.
                    expressions = reaction.expressions(scope)
                    overrides = network.variant_expressions.get(variant, {}).get(
                        reaction.ts_state, {}
                    )
                    expressions = {
                        name: overrides.get(name, terms)
                        for name, terms in expressions.items()
                    }
                    if overrides:
                        definition["reactant_label"] = "INT5"
                        definition["product_label"] = "Product"
                        definition["reactants"] = difference(
                            {reaction.ts_state: 1.0}, overrides["forward_barrier"]
                        )
                        definition["products"] = difference(
                            {reaction.ts_state: 1.0}, overrides["reverse_barrier"]
                        )
                    definition["expressions"] = expressions
                    records.append(definition)
                required = (
                    set().union(
                        *(set(e) for r in records for e in r["expressions"].values())
                    )
                    if records
                    else set()
                )
                # A complete selected branch can additionally provide its named ledger.
                complete = scope == "full_cycle" and {
                    r.ts_state for r in reactions
                } == {r.ts_state for r in network.reactions}
                profiles = (
                    {
                        label: terms
                        for label, terms in network.profile_terms.items()
                        if label not in roles or label in allowed
                    }
                    if complete
                    else {}
                )
                required.update(role for terms in profiles.values() for role in terms)
                required.update(network.barrier_roles)
                if scope == "full_cycle":
                    required.update(network.full_cycle_roles)
                bindings = {}
                applicability = {
                    s: (
                        "not_requested"
                        if s not in selected
                        and any(r.ts_state == s for r in network.reactions)
                        else (
                            "not_applicable"
                            if s not in allowed
                            else (
                                "discovery"
                                if variant == "unresolved"
                                and s in network.discovery_states
                                else "required"
                            )
                        )
                    )
                    for s in roles
                }
                for role in sorted(required):
                    state, binding = roles[role]
                    context = {**system, "rpos": int(site)}
                    if binding == "bound":
                        context.update(substrate_smiles=bound, rpos=bound_site)
                        if bound != system["substrate_smiles"]:
                            label = hashlib.sha256(bound.encode()).hexdigest()[:12]
                            context.update(
                                system_name=system["system_name"] + "__bound_" + label,
                                substrate_name="bound_" + label,
                            )
                    state_options = dimers if state == "dimer" else (state,)
                    for state_option in state_options:
                        planned = plan_targets(
                            pd.DataFrame([context]),
                            states=[state_option],
                            builder_options={"dimer_pairing": dimer_pairing},
                        )
                        for index, target in enumerate(planned):
                            targets[target.target_id] = target
                            if target.state_id in network.execution_groups:
                                groups[target.target_id] = network.execution_groups[
                                    target.state_id
                                ]
                            reasons.setdefault(target.target_id, []).extend(
                                r["reaction_id"]
                                for r in records
                                if any(role in e for e in r["expressions"].values())
                            )
                            if (
                                role in network.barrier_roles
                                or role in network.full_cycle_roles
                            ):
                                reasons[target.target_id].append(
                                    network.network_id + ":reference_pool"
                                )
                            if profiles:
                                reasons[target.target_id].append(
                                    network.network_id + ":profile"
                                )
                            if index == 0:
                                bindings[
                                    (
                                        role
                                        if state_option == state_options[0]
                                        else state_option
                                    )
                                ] = target.target_id
                cases.append(
                    dict(
                        network_id=network.network_id,
                        definition_version=1,
                        system_name=system["system_name"],
                        substrate_name=system["substrate_name"],
                        catalyst_name=system["catalyst_name"],
                        rpos=int(site),
                        system_signature=signature,
                        method_fingerprint=method_fingerprint,
                        variant=variant,
                        applicability=applicability,
                        bindings=bindings,
                        reactions=records,
                        profile_terms=profiles,
                        profile_order=list(profiles),
                        profile_mechanism_id=network.profile_mechanism_id,
                        reference=network.reference,
                    )
                )
    return dict(
        schema_version=1,
        cases=cases,
        targets=list(targets.values()),
        execution_groups=groups,
        reasons={k: sorted(set(v)) for k, v in reasons.items()},
        method_fingerprint=method_fingerprint,
    )
