"""Typed, serializable structure targets shared by FRUST workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StateKind = Literal["minimum", "transition_state", "constrained_minimum"]
TargetScope = Literal[
    "global", "substrate", "catalyst", "system", "substrate_rpos", "system_rpos"
]


@dataclass(frozen=True)
class ChemicalSystem:
    """Chemical inputs needed to construct one or more structure states.

    Parameters
    ----------
    system_name : str
        Stable label for the substrate/catalyst combination.
    substrate_name, catalyst_name : str
        Human-readable component labels.
    substrate_smiles, catalyst_smiles : str
        Component SMILES used by structure builders.
    metadata : dict, optional
        Extra JSON-compatible component metadata.
    """

    system_name: str
    substrate_name: str
    catalyst_name: str
    substrate_smiles: str
    catalyst_smiles: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a dataframe-friendly system mapping."""
        return {
            "system_name": self.system_name,
            "substrate_name": self.substrate_name,
            "catalyst_name": self.catalyst_name,
            "substrate_smiles": self.substrate_smiles,
            "catalyst_smiles": self.catalyst_smiles,
            **self.metadata,
        }


@dataclass(frozen=True)
class StructureTarget:
    """Lightweight plan for constructing one chemical state.

    Parameters
    ----------
    target_id : str
        Stable scientific identity independent of scheduler/file naming.
    tag : str
        Scheduler-safe target label.
    system : ChemicalSystem
        Chemical components used by the selected builder.
    state_id : str
        Canonical state identifier, for example ``"int2"`` or ``"INT3"``.
    state_kind : {"minimum", "transition_state", "constrained_minimum"}
        Chemical interpretation of the generated structure.
    builder_spec : str
        Versioned builder identifier.
    scope : str
        Deduplication scope used during target planning.
    rpos : int or None, optional
        Reactive atom position for position-dependent states.
    builder_options : dict, optional
        Small serializable builder options.

    Notes
    -----
    Calculation stages deliberately do not live on a target. ``mols``,
    ``screen_ts``, and ``int3`` remain separate workflows and each owns one
    readable, chemically homogeneous stage graph.
    """

    target_id: str
    tag: str
    system: ChemicalSystem
    state_id: str
    state_kind: StateKind
    builder_spec: str
    scope: TargetScope
    rpos: int | None = None
    builder_options: dict[str, Any] = field(default_factory=dict)

    @property
    def payload(self) -> dict[str, Any]:
        """Return a JSON-compatible payload for local or cluster execution."""
        return self.as_dict()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return compact target metadata used by workflow inspection."""
        metadata = {
            "target_id": self.target_id,
            "state_id": self.state_id,
            "state_kind": self.state_kind,
            "builder_spec": self.builder_spec,
            "system_name": self.system.system_name,
            "rpos": self.rpos,
        }
        if self.state_kind == "transition_state":
            metadata["ts_type"] = self.state_id
        return metadata

    def as_dict(self) -> dict[str, Any]:
        """Return the complete serializable target description."""
        return {
            "target_id": self.target_id,
            "tag": self.tag,
            "system": self.system.as_dict(),
            "state_id": self.state_id,
            "state_kind": self.state_kind,
            "builder_spec": self.builder_spec,
            "scope": self.scope,
            "rpos": self.rpos,
            "builder_options": dict(self.builder_options),
        }
