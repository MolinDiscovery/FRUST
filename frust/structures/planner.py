"""Lightweight target planning for molecule, TS, and INT3 workflows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from frust.cluster.naming import sanitize_tag
from frust.screen import expand as expand_screen
from frust.screen import read as read_screen
from frust.structures.models import ChemicalSystem, StructureTarget
from frust.structures.specs import DIMER_STATES, STANDARD_MOLECULE_STATES, get_state_spec
from frust.tsguess.matching import parse_rpos_value

DEFAULT_CATALYST_SMILES = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"
DEFAULT_CATALYST_NAME = "frust_catalyst"


def normalize_systems(input_data: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Normalize component, expanded-system, or substrate-only inputs.

    Parameters
    ----------
    input_data : str, pathlib.Path, or pandas.DataFrame
        CSV path or input table. Component tables use ``role``/``smiles``;
        expanded tables use the screen-system columns; substrate-only tables
        use ``smiles`` and the built-in FRUST catalyst.

    Returns
    -------
    pandas.DataFrame
        One row per explicit substrate/catalyst system.
    """
    df = (
        input_data.copy()
        if isinstance(input_data, pd.DataFrame)
        else pd.read_csv(input_data)
    )
    expanded_columns = {
        "system_name",
        "substrate_name",
        "catalyst_name",
        "substrate_smiles",
        "catalyst_smiles",
    }
    if expanded_columns.issubset(df.columns):
        out = df.copy()
        if "rpos" not in out:
            out["rpos"] = pd.NA
        return out
    if {"role", "smiles"}.issubset(df.columns):
        return expand_screen(read_screen(df))
    if "smiles" not in df:
        raise ValueError("structure workflow input must contain a 'smiles' column")
    if df["smiles"].isna().any():
        raise ValueError("structure workflow input contains missing SMILES values")

    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        name = (
            _first_text(row, "substrate_name", "compound_name", "name")
            or f"substrate_{position:03d}"
        )
        catalyst_name = _first_text(row, "catalyst_name") or DEFAULT_CATALYST_NAME
        catalyst_smiles = (
            _first_text(row, "catalyst_smiles") or DEFAULT_CATALYST_SMILES
        )
        system_name = _first_text(row, "system_name") or f"{name}__{catalyst_name}"
        values = {
            "system_name": system_name,
            "substrate_name": name,
            "catalyst_name": catalyst_name,
            "substrate_smiles": str(row["smiles"]),
            "catalyst_smiles": catalyst_smiles,
            "smiles": str(row["smiles"]),
            "rpos": row.get("rpos", pd.NA),
        }
        for column, value in row.items():
            values.setdefault(column, value)
        rows.append(values)
    return pd.DataFrame(rows)


def molecule_states(select_mols: str | Iterable[str]) -> tuple[str, ...]:
    """Resolve molecule selections to canonical state ids.

    Parameters
    ----------
    select_mols : str or iterable of str
        ``"all"`` selects the standard catalytic-cycle states. ``"dimers"``
        selects ``"dimer"``, ``"dimer_bh_bridged"``, and
        ``"dimer_eight_membered"``. ``"uniques"`` selects ``"ligand"``,
        ``"int1"``, ``"int2"``, and ``"HBpin-ligand"``. ``"generics"``
        selects ``"dimer"``, ``"HH"``, ``"catalyst"``, and ``"HBpin-mol"``.
        A single state or an explicit state collection may instead contain
        ``"dimer"``, ``"dimer_bh_bridged"``, ``"dimer_eight_membered"``,
        ``"HH"``, ``"ligand"``, ``"catalyst"``, ``"int1"``, ``"int2"``,
        ``"HBpin-ligand"``, or ``"HBpin-mol"``.

    Returns
    -------
    tuple of str
        Canonical molecule state ids in construction order.
    """
    all_states = STANDARD_MOLECULE_STATES
    accepted_states = (*all_states, *DIMER_STATES[1:])
    if select_mols == "all":
        return all_states
    if select_mols == "dimers":
        return DIMER_STATES
    if select_mols == "uniques":
        return ("ligand", "int1", "int2", "HBpin-ligand")
    if select_mols == "generics":
        return ("dimer", "HH", "catalyst", "HBpin-mol")
    requested = (select_mols,) if isinstance(select_mols, str) else tuple(select_mols)
    unknown = sorted(set(requested) - set(accepted_states))
    if unknown:
        migration = (
            " 'mol2' was renamed to 'int2', and the former 'int2' was renamed "
            "to 'int1'."
            if "mol2" in unknown
            else ""
        )
        raise ValueError(
            f"Unknown molecule states: {unknown}.{migration} "
            f"Expected one of {list(accepted_states)}, 'all', 'dimers', "
            "'uniques', or 'generics'."
        )
    return requested


def plan_targets(
    systems: pd.DataFrame,
    *,
    states: Iterable[str],
    builder_options: dict[str, Any] | None = None,
) -> list[StructureTarget]:
    """Plan deduplicated structure targets without constructing geometries.

    Parameters
    ----------
    systems : pandas.DataFrame
        Explicit systems from :func:`normalize_systems`.
    states : iterable of str
        Canonical states to plan for every applicable system.
    builder_options : dict or None, optional
        Small serializable options copied onto each target.

    Returns
    -------
    list of StructureTarget
        Lightweight targets deduplicated according to each state's scope.
    """
    targets: list[StructureTarget] = []
    seen: set[tuple[Any, ...]] = set()
    specs = [get_state_spec(state) for state in states]
    needs_rpos = any(spec.scope.endswith("_rpos") for spec in specs)
    for _, row in systems.iterrows():
        system = _chemical_system(row)
        rpos_values = (
            parse_rpos_value(row.get("rpos"), system.substrate_smiles)
            if needs_rpos
            else ()
        )
        for spec in specs:
            positions = rpos_values if spec.scope.endswith("_rpos") else (None,)
            for rpos in positions:
                scope_key = _scope_key(spec.scope, system, rpos)
                dedup_key = (spec.state_id, *scope_key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                target_id = _target_id(spec.state_id, system, spec.scope, rpos)
                targets.append(
                    StructureTarget(
                        target_id=target_id,
                        tag=sanitize_tag(target_id.replace(":", "__")),
                        system=system,
                        state_id=spec.state_id,
                        state_kind=spec.state_kind,
                        builder_spec=spec.builder_spec,
                        scope=spec.scope,
                        rpos=rpos,
                        builder_options=dict(builder_options or {}),
                    )
                )
    return targets


def _chemical_system(row: pd.Series) -> ChemicalSystem:
    standard = {
        "system_name",
        "substrate_name",
        "catalyst_name",
        "substrate_smiles",
        "catalyst_smiles",
        "smiles",
        "rpos",
    }
    metadata = {
        column: value for column, value in row.items() if column not in standard
    }
    return ChemicalSystem(
        system_name=str(row["system_name"]),
        substrate_name=str(row["substrate_name"]),
        catalyst_name=str(row["catalyst_name"]),
        substrate_smiles=str(row["substrate_smiles"]),
        catalyst_smiles=str(row["catalyst_smiles"]),
        metadata=metadata,
    )


def _scope_key(scope: str, system: ChemicalSystem, rpos: int | None) -> tuple[Any, ...]:
    if scope == "global":
        return ()
    if scope == "substrate":
        return (system.substrate_smiles,)
    if scope == "catalyst":
        return (system.catalyst_smiles,)
    if scope == "system":
        return (system.substrate_smiles, system.catalyst_smiles)
    if scope == "substrate_rpos":
        return (system.substrate_smiles, int(rpos))
    return (system.substrate_smiles, system.catalyst_smiles, int(rpos))


def _target_id(
    state_id: str, system: ChemicalSystem, scope: str, rpos: int | None
) -> str:
    if scope == "global":
        base = state_id
    elif scope == "catalyst":
        base = f"{state_id}:{system.catalyst_name}"
    elif scope.startswith("substrate"):
        base = f"{state_id}:{system.substrate_name}"
    else:
        base = f"{state_id}:{system.system_name}"
    return f"{base}:r{rpos}" if rpos is not None else base


def _first_text(row: pd.Series, *columns: str) -> str | None:
    for column in columns:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return None
