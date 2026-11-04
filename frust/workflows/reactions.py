"""Versioned chemistry definitions, independent of builders and calculators."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite


def difference(left, right):
    """Subtract balanced terms and cancel spectators before dependency expansion."""
    return {
        r: c
        for r in left.keys() | right.keys()
        if abs(c := left.get(r, 0.0) - right.get(r, 0.0)) > 1e-12
    }


@dataclass(frozen=True)
class ReactionDefinition:
    """Internal reaction sides and requested coverage in chemical roles."""

    reaction_id: str
    network: str
    ts_state: str
    reactants: dict[str, float]
    products: dict[str, float]
    roles: dict[str, tuple[str, str]]
    screening_reference: dict[str, float] | None = None
    requires_connection: bool = False
    schema_version: int = 1
    report_review_energy: bool = False
    reactant_label: str | None = None
    product_label: str | None = None
    screening_reference_label: str | None = None

    def expressions(self, scope):
        if scope not in {"barriers", "full_cycle"}:
            raise ValueError("scope must be barriers or full_cycle")
        ts = {self.ts_state: 1.0}
        result = {
            "forward_barrier": difference(
                ts, self.screening_reference or self.reactants
            )
        }
        if scope == "full_cycle":
            result.update(
                reverse_barrier=difference(ts, self.products),
                reaction_energy=difference(self.products, self.reactants),
            )
        return result

    def validate(self):
        from frust.structures.specs import get_state_spec

        for role, (state, binding) in self.roles.items():
            get_state_spec(state)
            if binding not in {"incoming", "bound"}:
                raise ValueError(f"Unknown component binding {binding}")
        if self.ts_state not in self.roles:
            raise ValueError("Saddle role must be registered")
        for label, terms in [
            ("reactants", self.reactants),
            ("products", self.products),
            ("screening", self.screening_reference or {}),
        ]:
            if set(terms) - self.roles.keys():
                raise ValueError("Reaction references unregistered roles")
            if any(
                not isfinite(v) or v == 0 or (label != "screening" and v < 0)
                for v in terms.values()
            ):
                raise ValueError("Reaction sides require finite positive coefficients")


@dataclass(frozen=True)
class NetworkDefinition:
    """Named balanced ledger and per-system pathway alternatives."""

    network_id: str
    reactions: tuple[ReactionDefinition, ...]
    profile_terms: dict[str, dict[str, float]]
    reference: str
    alternatives: dict[str, tuple[str, ...]] = field(default_factory=dict)
    discovery_states: tuple[str, ...] = ()
    default_variant: str = "defined"
    supports_racemic: bool = True
    execution_groups: dict[str, str] = field(default_factory=dict)
    variant_expressions: dict = field(default_factory=dict)
    full_cycle_roles: tuple[str, ...] = ()
    barrier_roles: tuple[str, ...] = ()
    profile_mechanism_id: str | None = None

    def validate(self):
        roles = {}
        for reaction in self.reactions:
            reaction.validate()
            for role, binding in reaction.roles.items():
                if role in roles and roles[role] != binding:
                    raise ValueError("Inconsistent role binding across reactions")
                roles[role] = binding
        if (set(self.full_cycle_roles) | set(self.barrier_roles)) - roles.keys():
            raise ValueError("Coverage references unregistered roles")
        if self.reference not in self.profile_terms:
            raise ValueError("Unknown profile reference")
        if any(set(t) - roles.keys() for t in self.profile_terms.values()):
            raise ValueError("Profile references unregistered roles")


NETWORKS: dict[str, NetworkDefinition] = {}


def register_network(definition):
    """Register internal/test chemistry data without changing orchestration."""
    definition.validate()
    if definition.network_id in NETWORKS:
        raise ValueError("Network already registered")
    existing = {r.ts_state for n in NETWORKS.values() for r in n.reactions}
    if existing & {r.ts_state for r in definition.reactions}:
        raise ValueError("TS family already registered")
    NETWORKS[definition.network_id] = definition


def _builtins():
    from frust.workflows.bisarylation import BISARYL_PROFILE_TERMS, PATH_STATES

    # Role bindings describe inputs, never atom indices or calculation options.
    prod_roles = {
        s: (s, "incoming")
        for s in (
            "TS1",
            "TS2",
            "TS3",
            "TS4",
            "int1",
            "int2",
            "INT3",
            "ligand",
            "catalyst",
            "dimer",
            "HBpin-mol",
            "HBpin-ligand",
            "HH",
        )
    }
    side_roles = {
        s: (s, "incoming")
        for s in (
            "TS5",
            "TS6",
            "TS7",
            "TS8",
            "INT4",
            "INT5",
            "INT6",
            "int2",
            "catalyst",
            "dimer",
            "HBpin-mol",
            "HH",
        )
    }
    side_roles.update(
        incoming_ligand=("ligand", "incoming"),
        bound_ligand=("ligand", "bound"),
        bound_int2=("int2", "bound"),
        bound_product=("HBpin-ligand", "bound"),
    )
    prod_terms = {
        "Dimer": {"dimer": 0.5, "ligand": 1.0, "HBpin-mol": 1.0},
        "Cat": {"catalyst": 1.0, "ligand": 1.0, "HBpin-mol": 1.0},
        "TS1": {"TS1": 1.0, "HBpin-mol": 1.0},
        "int1": {"int1": 1.0, "HBpin-mol": 1.0},
        "TS2": {"TS2": 1.0, "HBpin-mol": 1.0},
        "int2": {"int2": 1.0, "HH": 1.0, "HBpin-mol": 1.0},
        "TS3": {"TS3": 1.0, "HH": 1.0},
        "INT3": {"INT3": 1.0, "HH": 1.0},
        "TS4": {"TS4": 1.0, "HH": 1.0},
        "Product": {"dimer": 0.5, "HBpin-ligand": 1.0, "HH": 1.0},
    }
    prod_pairs = [
        ("TS1", "Cat", "int1"),
        ("TS2", "int1", "int2"),
        ("TS3", "int2", "INT3"),
        ("TS4", "INT3", "Product"),
    ]
    productive = []
    for ts, left, right in prod_pairs:
        # Remove spectators also present in the saddle's ledger node.
        spectators = {r: c for r, c in prod_terms[ts].items() if r != ts}
        sides = [difference(prod_terms[label], spectators) for label in (left, right)]
        productive.append(
            ReactionDefinition(
                f"productive.{ts.lower()}",
                "productive",
                ts,
                *sides,
                prod_roles,
                screening_reference=difference(prod_terms["Dimer"], spectators),
                report_review_energy=True,
                reactant_label=left,
                product_label=right,
                screening_reference_label="Dimer",
            )
        )
    register_network(
        NetworkDefinition(
            "productive",
            tuple(productive),
            prod_terms,
            "Dimer",
            execution_groups={
                "int1": "cycle_molecules",
                "int2": "cycle_molecules",
                "HBpin-ligand": "cycle_molecules",
                "INT3": "int3",
            },
            barrier_roles=("HH", "HBpin-mol"),
            full_cycle_roles=("catalyst", "int1", "int2", "INT3", "HBpin-ligand"),
            profile_mechanism_id="frust_balanced_cycle::v3",
        )
    )
    pairs = [
        ("TS5", {"bound_int2": 1.0, "incoming_ligand": 1.0}, {"INT4": 1.0}),
        ("TS6", {"INT4": 1.0}, {"INT5": 1.0, "HH": 1.0}),
        ("TS7", {"INT5": 1.0, "HBpin-mol": 1.0}, {"INT6": 1.0}),
        ("TS8", {"INT6": 1.0}, {"int2": 1.0, "bound_product": 1.0}),
    ]
    side = tuple(
        ReactionDefinition(
            f"bisarylation.{ts.lower()}",
            "bisarylation",
            ts,
            left,
            right,
            side_roles,
            requires_connection=ts in {"TS7", "TS8"},
            reactant_label={"TS5": "M", "TS6": "INT4", "TS7": "INT5", "TS8": "INT6"}[
                ts
            ],
            product_label={
                "TS5": "INT4",
                "TS6": "INT5",
                "TS7": "INT6",
                "TS8": "Product",
            }[ts],
        )
        for ts, left, right in pairs
    )
    register_network(
        NetworkDefinition(
            "bisarylation",
            side,
            BISARYL_PROFILE_TERMS,
            "M",
            alternatives={k: tuple(sorted(v)) for k, v in PATH_STATES.items()},
            discovery_states=("TS7", "INT6", "TS8"),
            default_variant="unresolved",
            supports_racemic=False,
            variant_expressions={
                variant: {
                    ts: {
                        "forward_barrier": difference(
                            BISARYL_PROFILE_TERMS[ts], BISARYL_PROFILE_TERMS["INT5"]
                        ),
                        "reverse_barrier": difference(
                            BISARYL_PROFILE_TERMS[ts], BISARYL_PROFILE_TERMS["Product"]
                        ),
                        "reaction_energy": difference(
                            BISARYL_PROFILE_TERMS["Product"],
                            BISARYL_PROFILE_TERMS["INT5"],
                        ),
                    }
                }
                for variant, ts in [("ts7_only", "TS7"), ("ts8_only", "TS8")]
            },
        )
    )


_builtins()
