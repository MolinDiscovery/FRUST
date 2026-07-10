"""Compact log-message formatting for dataframe stage metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def conformer_generation_summary(
    df: pd.DataFrame,
    *,
    label: str = "initial_conformers",
) -> str | None:
    """Return a compact conformer-generation log message.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe carrying optional ``df.attrs["frust_conformers"]`` metadata.
    label : str, optional
        Name shown in square brackets at the beginning of the log message.

    Returns
    -------
    str or None
        Formatted message, or ``None`` when no useful summary can be built.
    """
    conformers = df.attrs.get("frust_conformers")
    if not isinstance(conformers, Mapping):
        return f"[{label}] prepared {len(df)} row(s)"

    records = [
        record
        for record in conformers.get("structures", []) or []
        if isinstance(record, Mapping)
    ]
    generated = _int_or_none(conformers.get("total_generated_confs"))
    if generated is None and records:
        generated = sum(int(record.get("generated_n_confs") or 0) for record in records)
    if generated is None:
        generated = int(len(df))

    n_structures = _int_or_none(conformers.get("n_structures"))
    if n_structures is None and records:
        n_structures = len(records)

    parts = [f"[{label}] generated {generated} conformer row(s)"]
    if n_structures is not None:
        parts[-1] += f" from {n_structures} structure(s)"

    requested = _format_conformer_record_values(
        records,
        "requested_n_confs",
        fallback=conformers.get("requested_n_confs"),
        none_label="auto",
    )
    if requested is not None:
        parts.append(f"requested={requested}")

    resolved = _format_conformer_record_values(records, "resolved_n_confs")
    if resolved is not None:
        parts.append(f"resolved={resolved}")

    missing = _conformer_missing_count(records, generated)
    if missing is not None:
        parts.append(f"missing={missing}")

    backend = conformers.get("backend")
    if backend:
        parts.append(f"backend={backend}")

    return "; ".join(parts)


def pruning_summary(
    df: pd.DataFrame,
    *,
    name: str = "initial_prune",
    input_rows: int | None = None,
) -> str | None:
    """Return a compact pruning log message from ``frust_steps`` metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe carrying ``df.attrs["frust_steps"][name]`` metadata.
    name : str, optional
        Pruning step name.
    input_rows : int or None, optional
        Fallback input-row count when provenance does not include one.

    Returns
    -------
    str or None
        Formatted message, or ``None`` when the pruning step metadata is absent.
    """
    step_meta = _step_metadata(df, name)
    if not step_meta:
        return None

    row_counts = step_meta.get("row_counts") if isinstance(step_meta, Mapping) else {}
    options = step_meta.get("options") if isinstance(step_meta, Mapping) else {}

    in_rows = _int_or_none(row_counts.get("input_rows") if isinstance(row_counts, Mapping) else None)
    if in_rows is None:
        in_rows = input_rows
    out_rows = _int_or_none(row_counts.get("output_rows") if isinstance(row_counts, Mapping) else None)
    if out_rows is None:
        out_rows = int(len(df))
    dropped = _int_or_none(row_counts.get("dropped_rows") if isinstance(row_counts, Mapping) else None)
    if dropped is None and in_rows is not None:
        dropped = int(in_rows) - int(out_rows)

    parts = [f"[{name}] kept {out_rows}/{in_rows if in_rows is not None else '?'} row(s)"]
    if dropped is not None:
        parts.append(f"dropped={dropped}")

    if isinstance(options, Mapping):
        modes = options.get("modes")
        if modes:
            parts.append(f"modes={_format_mode_list(modes)}")
            active_modes = {modes} if isinstance(modes, str) else set(modes)
            if "moi" in active_modes:
                moi_max_deviation = _format_optional_value(
                    options.get("moi_max_deviation")
                )
                if moi_max_deviation is not None:
                    parts.append(f"moi_max_deviation={moi_max_deviation}")
            if active_modes.intersection({"rmsd", "rot_corr_rmsd"}):
                rmsd_max_rmsd = _format_optional_value(options.get("rmsd_max_rmsd"))
                if rmsd_max_rmsd is not None:
                    parts.append(f"rmsd_max_rmsd={rmsd_max_rmsd}")
                rmsd_max_dev = _format_optional_value(options.get("rmsd_max_dev"))
                if rmsd_max_dev is not None:
                    parts.append(f"rmsd_max_dev={rmsd_max_dev}")

    return "; ".join(parts)


def filter_summary(
    *,
    name: str = "filter",
    input_rows: int | None,
    output_rows: int,
) -> str:
    """Return a compact row-filtering log message.

    Parameters
    ----------
    name : str, optional
        Filter step name.
    input_rows : int or None
        Number of rows before filtering, when known.
    output_rows : int
        Number of rows after filtering.

    Returns
    -------
    str
        Formatted filter summary.
    """
    dropped = None if input_rows is None else int(input_rows) - int(output_rows)
    parts = [f"[{name}] kept {output_rows}/{input_rows if input_rows is not None else '?'} row(s)"]
    if dropped is not None:
        parts.append(f"dropped={dropped}")
    return "; ".join(parts)


def _step_metadata(df: pd.DataFrame, name: str) -> Mapping[str, Any]:
    """Return recorded ``frust_steps`` metadata for one step."""
    steps = df.attrs.get("frust_steps", {})
    if not isinstance(steps, Mapping):
        return {}
    meta = steps.get(name)
    return meta if isinstance(meta, Mapping) else {}


def _format_conformer_record_values(
    records: list[Mapping[str, Any]],
    key: str,
    *,
    fallback: Any = None,
    none_label: str | None = None,
) -> str | None:
    """Format per-structure conformer metadata values for logging."""
    values = [record.get(key, fallback) for record in records] or [fallback]
    normalized = [_format_optional_value(value, none_label=none_label) for value in values]
    normalized = [value for value in normalized if value is not None]
    if not normalized:
        return None
    unique = list(dict.fromkeys(normalized))
    if len(unique) == 1:
        return unique[0]
    return ",".join(unique)


def _format_optional_value(value: Any, *, none_label: str | None = None) -> str | None:
    """Format optional scalar metadata for log messages."""
    if value is None:
        return none_label
    try:
        if pd.isna(value):
            return none_label
    except (TypeError, ValueError):
        pass
    return str(value)


def _conformer_missing_count(records: list[Mapping[str, Any]], generated: int) -> int | None:
    """Return total requested-but-missing conformers when available."""
    if not records:
        return None
    resolved_values = [record.get("resolved_n_confs") for record in records]
    if any(value is None for value in resolved_values):
        return None
    try:
        resolved_total = sum(int(value) for value in resolved_values)
    except (TypeError, ValueError):
        return None
    return int(resolved_total) - int(generated)


def _int_or_none(value: Any) -> int | None:
    """Return ``value`` as an integer when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_mode_list(value: Any) -> str:
    """Format a pruning mode value for logs."""
    if isinstance(value, str):
        return value
    try:
        return ",".join(str(item) for item in value)
    except TypeError:
        return str(value)
