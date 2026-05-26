"""Row-level constraint rendering for FRUST calculator inputs."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def dataframe_has_row_constraints(df: pd.DataFrame) -> bool:
    """Return whether a dataframe carries row-level FRUST constraints.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to inspect.

    Returns
    -------
    bool
        ``True`` when both ``constraint_roles`` and ``constraint_spec`` columns
        are present.
    """
    return {"constraint_roles", "constraint_spec"}.issubset(df.columns)


def row_has_constraint_spec(row: Mapping[str, Any]) -> bool:
    """Return whether one row contains a usable row-level constraint spec.

    Parameters
    ----------
    row : mapping
        Row-like object containing FRUST metadata.

    Returns
    -------
    bool
        ``True`` when ``constraint_roles`` and ``constraint_spec`` are non-empty.
    """
    return not _is_missing(row.get("constraint_roles")) and not _is_missing(
        row.get("constraint_spec")
    )


def render_xtb_constraints(row: Mapping[str, Any], *, force_constant: float = 50) -> str | None:
    """Render row-level constraints for xTB/g-xTB input.

    Parameters
    ----------
    row : mapping
        Row containing ``constraint_roles`` and ``constraint_spec``.
    force_constant : float, optional
        xTB force constant used for the generated ``$constrain`` block.

    Returns
    -------
    str or None
        xTB ``$constrain`` block, or ``None`` when the row has no row-level
        constraint spec.
    """
    if not row_has_constraint_spec(row):
        return None

    roles = _role_mapping(row)
    lines = ["$constrain", f"force constant={force_constant:g}"]
    for entry in _constraint_entries(row):
        kind = str(entry.get("kind", "")).lower()
        atom_indices = _entry_atom_indices(entry, roles, offset=1)
        value = _entry_value(entry)

        if kind == "distance":
            lines.append(f"distance: {atom_indices[0]}, {atom_indices[1]}, {value:g}")
        elif kind == "angle":
            lines.append(
                f"angle: {atom_indices[0]}, {atom_indices[1]}, {atom_indices[2]}, {value:g}"
            )
        else:
            raise ValueError(f"Unsupported constraint kind: {kind!r}")

    lines.append("$end")
    return "\n".join(lines)


def render_orca_constraints(row: Mapping[str, Any]) -> str | None:
    """Render row-level constraints for ORCA input.

    Parameters
    ----------
    row : mapping
        Row containing ``constraint_roles`` and ``constraint_spec``.

    Returns
    -------
    str or None
        ORCA ``%geom Constraints`` block, or ``None`` when the row has no
        row-level constraint spec.
    """
    if not row_has_constraint_spec(row):
        return None

    roles = _role_mapping(row)
    lines = ["%geom Constraints"]
    for entry in _constraint_entries(row):
        kind = str(entry.get("kind", "")).lower()
        atom_indices = _entry_atom_indices(entry, roles, offset=0)
        value = _entry_value(entry)
        freeze = str(entry.get("freeze", "C"))

        if kind == "distance":
            lines.append(f"  {{B {atom_indices[0]} {atom_indices[1]} {value:g} {freeze}}}")
        elif kind == "angle":
            lines.append(
                f"  {{A {atom_indices[0]} {atom_indices[1]} {atom_indices[2]} {value:g} {freeze}}}"
            )
        else:
            raise ValueError(f"Unsupported constraint kind: {kind!r}")

    lines.extend(["end", "end"])
    return textwrap.dedent("\n".join(lines)).strip()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (Mapping, list, tuple)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _role_mapping(row: Mapping[str, Any]) -> dict[str, int]:
    value = row.get("constraint_roles")
    if not isinstance(value, Mapping):
        raise ValueError("'constraint_roles' must be a mapping from role names to atom indices")
    roles: dict[str, int] = {}
    for key, atom_idx in value.items():
        roles[str(key)] = int(atom_idx)
    return roles


def _constraint_entries(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("constraint_spec")
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = list(value)
    else:
        raise ValueError("'constraint_spec' must be a sequence of constraint mappings")
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise ValueError("'constraint_spec' entries must be mappings")
    return entries


def _entry_roles(entry: Mapping[str, Any]) -> list[str]:
    roles = entry.get("roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
        raise ValueError("constraint entries must contain a sequence-valued 'roles' field")
    return [str(role) for role in roles]


def _entry_atom_indices(
    entry: Mapping[str, Any],
    roles: Mapping[str, int],
    *,
    offset: int,
) -> list[int]:
    atom_indices: list[int] = []
    for role in _entry_roles(entry):
        if role not in roles:
            raise ValueError(f"constraint role {role!r} is missing from 'constraint_roles'")
        atom_indices.append(int(roles[role]) + offset)
    return atom_indices


def _entry_value(entry: Mapping[str, Any]) -> float:
    if "value" not in entry:
        raise ValueError("constraint entries must contain a numeric 'value'")
    return float(entry["value"])
