"""Exact result and portable evidence binding for reviewed reaction connections."""

from pathlib import Path
import hashlib


def connection_status(
    states,
    *,
    bindings,
    result_ids,
    saddle_role,
    saddle_result_id,
    evidence,
    root,
    protocol_identity=None,
):
    """Check one connection independently of other reactions in a network."""
    if not saddle_result_id:
        return "missing_saddle_binding"
    expected = [
        *(result_ids.items() if hasattr(result_ids, "items") else result_ids),
        (saddle_role, saddle_result_id),
    ]
    protocols = set()
    for role, result_id in expected:
        match = states.loc[
            states.result_id.eq(result_id) & states.structure_id.eq(bindings.get(role))
        ]
        if len(match) != 1 or match.iloc[0].quality_status != "ready":
            return "stale"
        protocol = match.iloc[0].get("energy_protocol_fingerprint")
        if not isinstance(protocol, str) or not protocol:
            return "missing_protocol"
        protocols.add(protocol)
    if protocol_identity is not None and protocols != {protocol_identity}:
        return "stale_protocol"
    if len(protocols) != 1:
        return "incompatible_protocols"
    if not evidence:
        return "missing_evidence"
    allowed = (Path(root) / "analysis/pathway_evidence").resolve()
    for item in evidence:
        path = (Path(root) / item["path"]).resolve()
        if not path.is_relative_to(allowed) or not path.is_file():
            return "missing_evidence"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            return "changed_evidence"
    return "approved"


def save_connection(
    run,
    *,
    system_name,
    rpos,
    reaction_id,
    saddle_result_id,
    reference_result_ids,
    endpoint_result_ids,
    direction_sides,
    evidence_files,
    decision="approved",
    note="",
):
    """Persist an explicit connection without deriving neighbors from ranking."""
    import json
    from datetime import datetime, timezone

    if run.manifest.get("calculation_level") != "full":
        raise ValueError("Connection review requires full characterization")
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    cases = [
        c
        for c in run.manifest.get("reaction_plan", {}).get("cases", [])
        if c["system_name"] == system_name
        and c["rpos"] == rpos
        and any(r["reaction_id"] == reaction_id for r in c["reactions"])
    ]
    if len(cases) != 1:
        raise ValueError(
            "Connection needs one exact planned reaction and chemical system"
        )
    case = cases[0]
    reaction = next(r for r in case["reactions"] if r["reaction_id"] == reaction_id)
    if (
        set(direction_sides.values()) != {"reactants", "products"}
        or len(direction_sides) != 2
    ):
        raise ValueError("Map both saved directions to reactants and products")
    if not endpoint_result_ids or not reference_result_ids:
        raise ValueError("Explicit endpoints and energy-reference results are required")
    from frust.structures.specs import get_state_spec

    endpoint_roles = set(endpoint_result_ids)
    for role in endpoint_roles:
        if (
            role not in reaction["roles"]
            or get_state_spec(reaction["roles"][role][0]).state_kind
            == "transition_state"
        ):
            raise ValueError("Endpoints require registered minimum roles")
    for side in ("reactants", "products"):
        side_roles = {
            r
            for r in reaction[side]
            if get_state_spec(reaction["roles"][r][0]).scope != "global"
        }
        if not endpoint_roles & side_roles:
            raise ValueError(
                "Endpoint results must explicitly cover both chemical sides"
            )
    required = set(reaction["expressions"]["forward_barrier"]) - {reaction["ts_state"]}
    if not required.issubset(reference_result_ids):
        raise ValueError("Connection is missing its forward energy references")
    sources = [Path(p).expanduser().resolve() for p in evidence_files]
    if not sources or any(not p.is_file() for p in sources):
        raise ValueError("Connection requires existing source evidence files")
    evidence = []
    directory = run.analysis_dir / "pathway_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = directory / (digest + source.suffix)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Existing evidence changed")
        destination.write_bytes(data)
        evidence.append(
            dict(path=str(destination.relative_to(run.path)), sha256=digest)
        )
    status = connection_status(
        run.states(),
        bindings=case["bindings"],
        result_ids=[*reference_result_ids.items(), *endpoint_result_ids.items()],
        saddle_role=reaction["ts_state"],
        saddle_result_id=saddle_result_id,
        evidence=evidence,
        root=run.path,
    )
    if status != "approved":
        raise ValueError("Connection results are not qualified: " + status)
    saddle_row = run.states().loc[lambda df: df.result_id.eq(saddle_result_id)].iloc[0]
    record = dict(
        schema_version=1,
        energy_protocol_fingerprint=saddle_row.energy_protocol_fingerprint,
        definition_fingerprint=hashlib.sha256(
            json.dumps(reaction, sort_keys=True).encode()
        ).hexdigest(),
        definition_version=reaction["schema_version"],
        reaction_id=reaction_id,
        system_signature=case["system_signature"],
        method_fingerprint=case["method_fingerprint"],
        saddle_target_id=case["bindings"][reaction["ts_state"]],
        saddle_result_id=saddle_result_id,
        endpoint_result_ids=dict(endpoint_result_ids),
        reference_result_ids=dict(reference_result_ids),
        direction_sides=dict(direction_sides),
        evidence=evidence,
        decision=decision,
        note=note,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    path = run.analysis_dir / "connections.json"
    previous = json.loads(path.read_text()).get("records", []) if path.exists() else []
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(schema_version=1, records=[*previous, record]), indent=2) + "\n"
    )
    temporary.replace(path)
    run.refresh_analysis()
