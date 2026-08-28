"""Canonical state registry for modern FRUST structure workflows."""

from __future__ import annotations

from dataclasses import dataclass

from frust.structures.models import StateKind, TargetScope


STANDARD_MOLECULE_STATES = (
    "dimer",
    "HH",
    "ligand",
    "catalyst",
    "int1",
    "int2",
    "HBpin-ligand",
    "HBpin-mol",
)
DIMER_STATES = (
    "dimer",
    "dimer_bh_bridged",
    "dimer_eight_membered",
)


@dataclass(frozen=True)
class StateSpec:
    """Construction metadata for one canonical chemical state.

    Parameters
    ----------
    state_id : str
        Canonical state name.
    state_kind : {"minimum", "transition_state", "constrained_minimum"}
        Chemical interpretation used by result schemas.
    builder_spec : str
        Versioned shared-builder identifier.
    scope : str
        Component dependency and target-deduplication scope.
    """

    state_id: str
    state_kind: StateKind
    builder_spec: str
    scope: TargetScope


STATE_SPECS: dict[str, StateSpec] = {
    "dimer": StateSpec("dimer", "minimum", "cycle::dimer::v2", "catalyst"),
    "dimer_bh_bridged": StateSpec(
        "dimer_bh_bridged",
        "minimum",
        "cycle::dimer_bh_bridged::v1",
        "catalyst",
    ),
    "dimer_eight_membered": StateSpec(
        "dimer_eight_membered",
        "minimum",
        "cycle::dimer_eight_membered::v1",
        "catalyst",
    ),
    "HH": StateSpec("HH", "minimum", "cycle::HH::v2", "global"),
    "ligand": StateSpec("ligand", "minimum", "cycle::ligand::v2", "substrate"),
    "catalyst": StateSpec("catalyst", "minimum", "cycle::catalyst::v2", "catalyst"),
    "int1": StateSpec("int1", "minimum", "cycle::int1::v3", "system_rpos"),
    "int2": StateSpec("int2", "minimum", "cycle::int2::v3", "system_rpos"),
    "HBpin-ligand": StateSpec(
        "HBpin-ligand", "minimum", "cycle::HBpin-ligand::v2", "substrate_rpos"
    ),
    "HBpin-mol": StateSpec("HBpin-mol", "minimum", "cycle::HBpin-mol::v2", "global"),
    **{
        f"TS{index}": StateSpec(
            f"TS{index}",
            "transition_state",
            f"connected_graph::TS{index}::v2",
            "system_rpos",
        )
        for index in range(1, 5)
    },
    "INT3": StateSpec(
        "INT3", "constrained_minimum", "connected_graph::INT3::v2", "system_rpos"
    ),
}


def get_state_spec(state_id: str) -> StateSpec:
    """Return a canonical state specification.

    Parameters
    ----------
    state_id : str
        State name. TS and INT names are case-insensitive.

    Returns
    -------
    StateSpec
        Registered construction specification.
    """
    text = str(state_id)
    upper = text.upper()
    key = upper if upper.startswith("TS") or upper == "INT3" else text
    try:
        return STATE_SPECS[key]
    except KeyError as exc:
        known = ", ".join(STATE_SPECS)
        raise ValueError(
            f"Unknown structure state {state_id!r}; expected one of {known}"
        ) from exc
