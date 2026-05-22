"""Dataframe inspection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from frust.schema import energy_columns, infer_group_columns, normalize_dataframe


def show_steps(df: pd.DataFrame, *, detail: str = "summary") -> pd.DataFrame:
    """Summarize FRUST calculation step metadata as a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST results dataframe. The helper reads ``df.attrs["frust_steps"]``
        when present.
    detail : {"summary", "full"}, optional
        Level of detail to return. ``"summary"`` is intended for quick
        inspection and omits verbose executable and environment provenance.
        ``"full"`` includes every available metadata column.

    Returns
    -------
    pandas.DataFrame
        One row per recorded calculation step, indexed by step name. Columns
        that are entirely empty are removed.
    """
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be 'summary' or 'full'")

    rows = []

    for step_name, step in _steps_mapping(df).items():
        if not isinstance(step, Mapping):
            continue

        calculator = step.get("calculator", {})
        if not isinstance(calculator, Mapping):
            calculator = {}

        resources = calculator.get("resources", {})
        if not isinstance(resources, Mapping):
            resources = {}

        input_data = step.get("input", {})
        if not isinstance(input_data, Mapping):
            input_data = {}

        columns = step.get("columns")

        rows.append(
            {
                "step": step_name,
                "engine": step.get("engine"),
                "calc_name": calculator.get("name"),
                "mode": calculator.get("mode"),
                "backend": calculator.get("backend"),
                "options": _format_keys(step.get("options")),
                "columns": _format_list(columns),
                "n_columns": len(columns) if isinstance(columns, list) else 0,
                "n_cores": resources.get("n_cores"),
                "memory_gb": resources.get("memory_gb"),
                "xtra_inp_str": input_data.get("xtra_inp_str"),
                "detailed_inp_str": input_data.get("detailed_inp_str"),
                "input": _format_mapping(input_data),
                "executables": _format_paths(calculator.get("executables")),
                "environment": _format_paths(calculator.get("environment")),
                "gxtb": step.get("gxtb"),
                "gxtb_exe": step.get("gxtb_exe"),
                "gxtb_exe_source": step.get("gxtb_exe_source"),
            }
        )

    if not rows:
        out = pd.DataFrame()
        out.index.name = "step"
        return out

    out = pd.DataFrame(rows).set_index("step")
    if detail == "summary":
        out = out[
            [
                "engine",
                "calc_name",
                "mode",
                "options",
                "columns",
                "n_columns",
                "n_cores",
                "memory_gb",
                "xtra_inp_str",
            ]
        ]
    return out.dropna(axis=1, how="all")


def lowest_energy_rows(
    df: pd.DataFrame,
    n: int = 1,
    *,
    energy_col: str | None = None,
    group_cols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the lowest-energy rows per FRUST structure group.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST results dataframe.
    n : int, optional
        Number of rows to keep per group. Defaults to 1.
    energy_col : str or None, optional
        Energy column to sort by. If ``None``, the latest FRUST energy column is
        used, matching ``Stepper(lowest=...)`` behavior.
    group_cols : list of str, tuple of str, or None, optional
        Columns that define one structure group. If ``None``, FRUST infers the
        same structure identity columns used by ``Stepper(lowest=...)``.

    Returns
    -------
    pandas.DataFrame
        Normalized dataframe containing up to ``n`` low-energy rows per group,
        sorted by group columns and energy. Original row indices are preserved.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    out = normalize_dataframe(df)
    out.attrs.update(getattr(df, "attrs", {}))

    if energy_col is None:
        e_cols = energy_columns(out)
        if not e_cols:
            raise ValueError("cannot select lowest rows: no energy column found")
        energy_col = e_cols[-1]
    else:
        energy_col = _normalized_column_name(energy_col)

    if energy_col not in out.columns:
        raise ValueError(f"cannot select lowest rows: {energy_col!r} is not a column")

    if group_cols is None:
        group_cols = infer_group_columns(out)
        if not group_cols:
            raise ValueError("cannot select lowest rows: no structure identity columns found")
    else:
        group_cols = [_normalized_column_name(col) for col in group_cols]
        missing = [col for col in group_cols if col not in out.columns]
        if missing:
            raise ValueError(
                "cannot select lowest rows: missing group columns "
                + ", ".join(map(repr, missing))
            )

    sort_keys = list(group_cols) + [energy_col]
    return (
        out.sort_values(sort_keys, na_position="last")
        .groupby(list(group_cols), dropna=False)
        .head(n)
    )


def _normalized_column_name(column: str) -> str:
    """Return the canonical FRUST name for one possibly legacy column name."""
    normalized = normalize_dataframe(pd.DataFrame(columns=[column]))
    return str(normalized.columns[0])


def _steps_mapping(df: pd.DataFrame) -> Mapping[str, Any]:
    """Return the dataframe's FRUST step metadata mapping."""
    steps = df.attrs.get("frust_steps", {})
    if isinstance(steps, Mapping):
        return steps
    return {}


def _format_keys(value: Any) -> str | None:
    """Format mapping keys as a compact option list."""
    if not isinstance(value, Mapping) or not value:
        return None
    return " ".join(map(str, value))


def _format_list(value: Any) -> str | None:
    """Format list-like metadata as a comma-separated string."""
    if not isinstance(value, list) or not value:
        return None
    return ", ".join(map(str, value))


def _format_mapping(value: Any) -> str | None:
    """Format call/input metadata as one compact multiline string."""
    if not isinstance(value, Mapping) or not value:
        return None

    parts = []
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            formatted = _format_keys(item)
        elif isinstance(item, list):
            formatted = _format_list(item)
        else:
            formatted = str(item)

        if formatted is not None:
            parts.append(f"{key}: {formatted}")

    return "\n".join(parts) if parts else None


def _format_paths(items: Any) -> str | None:
    """Format executable or environment path provenance."""
    if not isinstance(items, Mapping) or not items:
        return None

    parts = []

    for name, info in items.items():
        if not isinstance(info, Mapping):
            parts.append(f"{name}={info}")
            continue

        path = info.get("path")
        source = info.get("source")
        resolved = info.get("resolved")
        configured = info.get("configured")

        details = []

        if source is not None:
            details.append(str(source))

        if resolved is not None:
            details.append(f"resolved={resolved}")

        if configured is not None and configured != path:
            details.append(f"configured={configured}")

        suffix = f" [{', '.join(details)}]" if details else ""
        parts.append(f"{name}: {path}{suffix}")

    return "\n".join(parts)
