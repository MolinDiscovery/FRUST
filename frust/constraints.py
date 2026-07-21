"""Row-level constraint rendering for FRUST calculator inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
import re
import textwrap
from typing import Any

import pandas as pd


_LEGACY_ROLE_SLOTS: dict[str, dict[str, int]] = {
    "TS1": {"cat_B": 0, "cat_N": 1, "transfer_H": 4, "substrate_C": 5},
    "TS2": {
        "cat_B": 0,
        "cat_N": 1,
        "B_transfer_H": 3,
        "N_transfer_H": 4,
        "substrate_C": 5,
    },
    "TS3": {"cat_B": 0, "pin_B": 3, "transfer_H": 4, "substrate_C": 5},
    "TS4": {"cat_B": 0, "transfer_H": 3, "pin_B": 4, "substrate_C": 5},
    "INT3": {"cat_B": 0, "pin_B": 3, "transfer_H": 4, "substrate_C": 5},
}


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


def validate_dataframe_constraints(df: pd.DataFrame) -> None:
    """Validate role-based constraints for every dataframe row.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe passed to a constrained calculator step.

    Raises
    ------
    ValueError
        If the dataframe lacks the role/spec columns, any row has missing
        constraint data, or a constraint references an invalid role or kind.

    Examples
    --------
    A constrained row is self-describing; Stepper does not infer chemistry
    from ``step_type`` or positional atom lists::

        constraint_roles = {"cat_B": 4, "transfer_H": 17}
        constraint_spec = [
            {"kind": "distance", "roles": ("cat_B", "transfer_H"), "value": 1.28}
        ]
    """
    required = {"constraint_roles", "constraint_spec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "`constraint=True` requires role-based row constraints; missing "
            f"column(s): {', '.join(missing)}. Generate structures with a modern "
            "TS backend or call ft.upgrade_legacy_constraints(...) explicitly."
        )

    for index, row in df.iterrows():
        if not row_has_constraint_spec(row):
            raise ValueError(
                f"Constrained dataframe row {index!r} has missing or empty "
                "'constraint_roles'/'constraint_spec' data"
            )
        try:
            # Rendering exercises the complete schema, including role lookup,
            # arity, numeric values, and supported constraint kinds.
            render_xtb_constraints(row)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValueError(
                f"Invalid role-based constraints in dataframe row {index!r}: {exc}"
            ) from exc


def upgrade_legacy_constraints(
    df: pd.DataFrame,
    *,
    spec_profile: str,
    state: str | None = None,
    spec_match: str = "prefer-exact",
    drop_legacy: bool = True,
) -> pd.DataFrame:
    """Convert a historical positional constraint dataframe explicitly.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical dataframe containing ``constraint_atoms``.
    spec_profile : str
        Modern method/environment geometry profile whose constraint values
        should be attached, for example ``"wb97xd3-631g/gas"``.
    state : str or None, optional
        State to apply to every row. When omitted, each row is inferred from
        ``state_id``, ``structure_type``, then ``custom_name``.
    spec_match : {"prefer-exact", "exact"}, optional
        Geometry-profile resolution policy.
    drop_legacy : bool, optional
        Remove ``constraint_atoms`` after conversion. Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        Copy containing ``constraint_roles``, ``constraint_spec``, and
        ``ts_spec_id``.

    Notes
    -----
    This is an explicit migration boundary, not a compatibility path in
    :class:`frust.stepper.Stepper`. The positional layouts are recognized only
    for historical FRUST TS1--TS4 and INT3 dataframes.

    Examples
    --------
    >>> import frust as ft
    >>> modern = ft.upgrade_legacy_constraints(
    ...     old_df,
    ...     spec_profile="wb97xd3-631g/gas",
    ... )
    >>> step = ft.Stepper(step_type="auto")
    >>> optimized = step.xtb(modern, constraint=True)
    """
    if "constraint_atoms" not in df.columns:
        raise ValueError(
            "Legacy constraint upgrade requires a 'constraint_atoms' column"
        )

    from frust.tsguess2.profiles import resolve_profile_spec

    out = df.copy(deep=True)
    roles_column: list[dict[str, int]] = []
    specs_column: list[list[dict[str, object]]] = []
    spec_ids: list[str] = []
    selections: dict[str, dict[str, str]] = {}

    explicit_state = None if state is None else _normalize_constraint_state(state)
    for index, row in out.iterrows():
        row_state = explicit_state or _infer_constraint_state(row, index=index)
        slots = _LEGACY_ROLE_SLOTS[row_state]
        atom_indices = _legacy_atom_indices(row.get("constraint_atoms"), index=index)
        largest_slot = max(slots.values())
        if len(atom_indices) <= largest_slot:
            raise ValueError(
                f"Legacy constraint row {index!r} for {row_state} requires at "
                f"least {largest_slot + 1} positional atoms, got {len(atom_indices)}"
            )

        selection = resolve_profile_spec(
            row_state,
            spec_profile,
            match=spec_match,
        )
        roles_column.append(
            {role: int(atom_indices[position]) for role, position in slots.items()}
        )
        specs_column.append(selection.spec.constraint_dicts())
        spec_ids.append(selection.spec.spec_id)
        selections[row_state] = {
            "requested_profile": selection.requested_profile,
            "resolved_profile": selection.resolved_profile,
            "match": selection.match,
            "spec_id": selection.spec.spec_id,
        }

    out["constraint_roles"] = roles_column
    out["constraint_spec"] = specs_column
    out["ts_spec_id"] = spec_ids
    if drop_legacy:
        out = out.drop(columns=["constraint_atoms"])
    out.attrs["frust_constraint_upgrade"] = {
        "schema_version": 1,
        "source": "constraint_atoms",
        "drop_legacy": bool(drop_legacy),
        "selections": selections,
    }
    return out


def _normalize_constraint_state(value: Any) -> str:
    state = str(value).strip().upper()
    if state not in _LEGACY_ROLE_SLOTS:
        available = ", ".join(_LEGACY_ROLE_SLOTS)
        raise ValueError(f"Unsupported legacy constraint state {value!r}; expected {available}")
    return state


def _infer_constraint_state(row: Mapping[str, Any], *, index: Any) -> str:
    for column in ("state_id", "structure_type"):
        value = row.get(column)
        if not _is_missing(value):
            try:
                return _normalize_constraint_state(value)
            except ValueError:
                pass

    custom_name = row.get("custom_name")
    if not _is_missing(custom_name):
        match = re.match(r"\s*(TS[1-4]|INT3)\b", str(custom_name), flags=re.IGNORECASE)
        if match:
            return _normalize_constraint_state(match.group(1))
    raise ValueError(
        f"Could not infer a supported constraint state for legacy row {index!r}; "
        "pass state= explicitly or provide state_id/structure_type/custom_name"
    )


def _legacy_atom_indices(value: Any, *, index: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"Legacy constraint row {index!r} must contain a sequence-valued "
            "'constraint_atoms'"
        )
    try:
        return [int(atom_index) for atom_index in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Legacy constraint row {index!r} contains a non-integer atom index"
        ) from exc


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
        kind = _entry_kind(entry)
        atom_indices = _entry_atom_indices(entry, roles, offset=1)
        _validate_entry_arity(kind, atom_indices)
        value = _entry_value(entry, kind=kind)

        if kind == "distance":
            lines.append(f"distance: {atom_indices[0]}, {atom_indices[1]}, {value:g}")
        else:
            lines.append(
                f"angle: {atom_indices[0]}, {atom_indices[1]}, {atom_indices[2]}, {value:g}"
            )

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
        kind = _entry_kind(entry)
        atom_indices = _entry_atom_indices(entry, roles, offset=0)
        _validate_entry_arity(kind, atom_indices)
        value = _entry_value(entry, kind=kind)
        freeze = str(entry.get("freeze", "C"))

        if kind == "distance":
            lines.append(f"  {{B {atom_indices[0]} {atom_indices[1]} {value:g} {freeze}}}")
        else:
            lines.append(
                f"  {{A {atom_indices[0]} {atom_indices[1]} {atom_indices[2]} {value:g} {freeze}}}"
            )

    lines.extend(["end", "end"])
    return textwrap.dedent("\n".join(lines)).strip()


def render_orca_geometry_controls(
    row: Mapping[str, Any],
    *,
    calc_hess: bool = False,
    read_hessian: bool = False,
    ts_mode: Sequence[str] | None = None,
    ts_active_atoms: Sequence[str] | None = None,
    ts_active_atoms_factor: float | None = None,
    recalc_hess: int | None = None,
    trust_radius: float | None = None,
) -> str | None:
    """Render one ORCA ``%geom`` block from row-level chemical roles.

    Parameters
    ----------
    row : mapping
        Row containing ``constraint_roles`` when role-based controls are used.
    calc_hess : bool, optional
        Add ``Calc_Hess true``.
    read_hessian : bool, optional
        Read ``private_input.hess`` as the initial Hessian.
    ts_mode : sequence of str or None, optional
        Chemical roles defining the internal coordinate ORCA should follow.
        Two roles produce a bond coordinate, three an angle, and four a
        dihedral. For TS3, use ``("pin_B", "substrate_C")``.
    ts_active_atoms : sequence of str or None, optional
        Chemical roles whose atoms should be included in ORCA's active-atom
        treatment. For TS3, the complete reactive core is ``("cat_B",
        "transfer_H", "pin_B", "substrate_C")``.
    ts_active_atoms_factor : float or None, optional
        Positive ORCA ``TS_Active_Atoms_Factor``. Requires
        ``ts_active_atoms``.
    recalc_hess : int or None, optional
        Positive number of optimization cycles between exact Hessian
        recalculations.
    trust_radius : float or None, optional
        Positive adaptive ORCA trust radius.

    Returns
    -------
    str or None
        A single ORCA ``%geom`` block, or ``None`` when no controls were
        requested.

    Examples
    --------
    >>> row = {
    ...     "atoms": ["B", "H", "B", "C"],
    ...     "constraint_roles": {
    ...         "cat_B": 0, "transfer_H": 1, "pin_B": 2, "substrate_C": 3,
    ...     },
    ... }
    >>> block = render_orca_geometry_controls(
    ...     row,
    ...     read_hessian=True,
    ...     ts_mode=("pin_B", "substrate_C"),
    ...     ts_active_atoms=("cat_B", "transfer_H", "pin_B", "substrate_C"),
    ...     ts_active_atoms_factor=1.5,
    ...     recalc_hess=3,
    ...     trust_radius=0.15,
    ... )
    >>> "TS_Mode {B 2 3}" in block
    True
    """
    requested = any(
        (
            calc_hess,
            read_hessian,
            ts_mode is not None,
            ts_active_atoms is not None,
            ts_active_atoms_factor is not None,
            recalc_hess is not None,
            trust_radius is not None,
        )
    )
    if not requested:
        return None

    lines = ["%geom"]
    if calc_hess:
        lines.append("  Calc_Hess true")
    if read_hessian:
        lines.extend(
            [
                "  inhess Read",
                '  InHessName "private_input.hess"',
            ]
        )

    if ts_mode is not None:
        mode_roles = _role_sequence(ts_mode, name="ts_mode")
        coordinate_types = {2: "B", 3: "A", 4: "D"}
        try:
            coordinate_type = coordinate_types[len(mode_roles)]
        except KeyError as exc:
            raise ValueError("ts_mode requires two, three, or four chemical roles") from exc
        mode_atoms = _role_atom_indices(row, mode_roles)
        lines.append(
            f"  TS_Mode {{{coordinate_type} {' '.join(str(index) for index in mode_atoms)}}}"
        )

    if ts_active_atoms is not None:
        active_roles = _role_sequence(ts_active_atoms, name="ts_active_atoms")
        active_atoms = _role_atom_indices(row, active_roles)
        lines.append(
            f"  TS_Active_Atoms {{ {' '.join(str(index) for index in active_atoms)} }}"
        )
    if ts_active_atoms_factor is not None:
        if ts_active_atoms is None:
            raise ValueError("ts_active_atoms_factor requires ts_active_atoms")
        factor = _positive_finite_float(
            ts_active_atoms_factor,
            name="ts_active_atoms_factor",
        )
        lines.append(f"  TS_Active_Atoms_Factor {factor:g}")

    if recalc_hess is not None:
        interval = _positive_integer(recalc_hess, name="recalc_hess")
        lines.append(f"  Recalc_Hess {interval}")
    if trust_radius is not None:
        radius = _positive_finite_float(trust_radius, name="trust_radius")
        lines.append(f"  Trust {radius:g}")

    lines.append("end")
    return "\n".join(lines)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (Mapping, list, tuple)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _role_sequence(value: Sequence[str], *, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of chemical role names")
    roles = [str(role).strip() for role in value]
    if not roles or any(not role for role in roles):
        raise ValueError(f"{name} must contain non-empty chemical role names")
    if len(set(roles)) != len(roles):
        raise ValueError(f"{name} must not contain duplicate chemical roles")
    return roles


def _role_atom_indices(row: Mapping[str, Any], role_names: Sequence[str]) -> list[int]:
    roles = _role_mapping(row)
    missing = [role for role in role_names if role not in roles]
    if missing:
        available = ", ".join(sorted(roles)) or "none"
        raise ValueError(
            "ORCA geometry controls reference missing chemical role(s): "
            f"{', '.join(missing)}. Available roles: {available}"
        )
    return [roles[role] for role in role_names]


def _positive_finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number <= 0 or float(value) != number:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _role_mapping(row: Mapping[str, Any]) -> dict[str, int]:
    value = row.get("constraint_roles")
    if not isinstance(value, Mapping):
        raise ValueError("'constraint_roles' must be a mapping from role names to atom indices")
    roles: dict[str, int] = {}
    for key, atom_idx in value.items():
        index = int(atom_idx)
        if index < 0:
            raise ValueError(f"constraint role {key!r} has a negative atom index")
        roles[str(key)] = index
    atoms = row.get("atoms")
    if isinstance(atoms, Sequence) and not isinstance(atoms, (str, bytes)):
        invalid = {role: index for role, index in roles.items() if index >= len(atoms)}
        if invalid:
            raise ValueError(f"constraint roles reference atoms outside the row: {invalid}")
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


def _entry_kind(entry: Mapping[str, Any]) -> str:
    kind = str(entry.get("kind", "")).lower()
    if kind not in {"distance", "angle"}:
        raise ValueError(f"Unsupported constraint kind: {kind!r}")
    return kind


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


def _validate_entry_arity(kind: str, atom_indices: Sequence[int]) -> None:
    expected = 2 if kind == "distance" else 3
    if len(atom_indices) != expected:
        raise ValueError(
            f"{kind} constraints require exactly {expected} roles, "
            f"got {len(atom_indices)}"
        )


def _entry_value(entry: Mapping[str, Any], *, kind: str) -> float:
    if "value" not in entry:
        raise ValueError("constraint entries must contain a numeric 'value'")
    value = float(entry["value"])
    if not isfinite(value):
        raise ValueError("constraint entries must contain a finite numeric 'value'")
    if kind == "distance" and value <= 0:
        raise ValueError("distance constraint values must be positive")
    if kind == "angle" and not 0 < value <= 180:
        raise ValueError("angle constraint values must be in (0, 180]")
    return value
