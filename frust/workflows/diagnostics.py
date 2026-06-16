"""Diagnostics for finished FRUST workflow outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from frust.schema import normal_termination_columns, normalize_dataframe

FailureDetail = Literal["summary", "full"]

SUMMARY_COLUMNS = [
    "target",
    "file",
    "row_index",
    "ts_type",
    "substrate_name",
    "compound_name",
    "catalyst_name",
    "rpos",
    "cid",
    "failed_stage",
    "failed_nt_cols",
    "error",
    "backend_hint",
    "last_checkpoint",
    "target_dir",
    "problem",
]

FULL_EXTRA_COLUMNS = [
    "failed_nt_values",
    "backend_column",
    "source",
]

IDENTITY_COLUMNS = [
    "ts_type",
    "substrate_name",
    "compound_name",
    "catalyst_name",
    "rpos",
    "cid",
]

FALSE_STRINGS = {"", "0", "false", "f", "no", "n", "none", "nan", "na", "<na>"}
BACKEND_SUFFIXES = (".out", ".log", ".txt")
BACKEND_PATTERNS = (
    "error termination",
    "did not terminate normally",
    "does not terminate normally",
    "traceback",
    "runtimeerror",
    "error:",
    " error ",
    "failed",
)


def inspect_failures(
    source: str | Path | pd.DataFrame | Mapping[str, Any],
    *,
    detail: FailureDetail = "summary",
) -> pd.DataFrame:
    """Inspect failed rows and skipped targets from workflow outputs.

    Parameters
    ----------
    source : str, pathlib.Path, pandas.DataFrame, or mapping
        Workflow diagnostics source. Accepted values are:

        - a workflow run directory containing ``collection_report.json``;
        - a ``collection_report.json`` path;
        - a target parquet path;
        - a loaded FRUST results dataframe;
        - a loaded collection-report mapping.
    detail : {"summary", "full"}, optional
        Output detail. ``"summary"`` returns compact notebook-facing columns.
        ``"full"`` also includes status values and backend-output column names.

    Returns
    -------
    pandas.DataFrame
        One row per failed workflow row or target-level collection problem.
        The ``problem`` column uses these values:

        ``"failed_stage"``
            A row exists, but at least one ``*-NT`` column is false or missing.
        ``"missing_output"``
            The collector expected a final parquet that does not exist.
        ``"read_error"``
            A parquet or report file could not be read.
        ``"unknown_skipped"``
            A skipped file exists, but no row-level failure could be extracted.

    Examples
    --------
    Show the failed target that prevented a workflow result from being merged:

    >>> failures = inspect_failures("runs/screen_ts/collection_report.json")
    >>> failures[["target", "failed_stage", "error"]]
    """
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be 'summary' or 'full'")

    if isinstance(source, pd.DataFrame):
        records = _summarize_dataframe_failures(source)
    elif isinstance(source, Mapping):
        records = _summarize_collection_failures(source)
    else:
        records = _inspect_path(Path(source))

    return _records_frame(records, detail=detail)


def _summarize_dataframe_failures(
    df: pd.DataFrame,
    *,
    file: str | Path | None = None,
    target: str | None = None,
    problem: str = "failed_stage",
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Build failure records from one FRUST dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST results dataframe with stage-prefixed normal-termination columns.
    file : str or pathlib.Path, optional
        Parquet file that produced ``df``.
    target : str, optional
        Workflow target tag. When omitted, the value is read from
        ``df.attrs["frust_workflow"]["target"]`` or inferred from ``file``.
    problem : str, optional
        Problem label to store in each returned record. Defaults to
        ``"failed_stage"``.
    source : str, optional
        Human-readable source label, such as ``"skipped_files"``.

    Returns
    -------
    list of dict
        JSON-serializable failure records.
    """
    frame = normalize_dataframe(df)
    file_path = None if file is None else Path(file)
    target_name = _target_name(target, frame, file_path)
    nt_cols = normal_termination_columns(frame)
    records: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        failed_cols = [col for col in nt_cols if _is_failed_value(row.get(col))]
        if not failed_cols:
            continue

        failed_stage = _stage_from_nt_col(failed_cols[0])
        error = _row_error(row, failed_cols)
        backend_hint, backend_column = _row_backend_hint(row, failed_stage)
        failed_values = {
            col: _json_scalar(row.get(col))
            for col in failed_cols
        }
        record = {
            "target": target_name,
            "file": _path_text(file_path),
            "row_index": _json_scalar(row_index),
            "failed_stage": failed_stage,
            "failed_nt_cols": list(failed_cols),
            "failed_nt_values": failed_values,
            "error": error,
            "backend_hint": backend_hint,
            "backend_column": backend_column,
            "last_checkpoint": _path_text(file_path),
            "target_dir": _path_text(file_path.parent if file_path is not None else None),
            "problem": problem,
            "source": source,
        }
        for column in IDENTITY_COLUMNS:
            record[column] = _json_scalar(row.get(column)) if column in frame.columns else None
        records.append(_json_record(record))
    return records


def _summarize_collection_failures(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build failure records from a workflow collection report.

    Parameters
    ----------
    report : mapping
        Loaded ``collection_report.json`` payload.

    Returns
    -------
    list of dict
        JSON-serializable failure records.
    """
    existing = report.get("failure_summary")
    if isinstance(existing, list):
        return [_json_record(record) for record in existing if isinstance(record, Mapping)]

    records: list[dict[str, Any]] = []
    for file_name in _string_items(report.get("skipped_files")):
        path = Path(file_name)
        records.extend(_failure_records_from_parquet(path, source="skipped_files"))

    for file_name in _string_items(report.get("missing_files")):
        records.append(_missing_output_record(file_name))

    errors = report.get("errors")
    if isinstance(errors, list):
        for item in errors:
            if not isinstance(item, Mapping):
                continue
            records.append(
                _read_error_record(
                    item.get("file"),
                    error=item.get("error"),
                    source="errors",
                )
            )

    for file_name in _string_items(report.get("errored_files")):
        if any(record.get("file") == file_name for record in records):
            continue
        records.append(_read_error_record(file_name, source="errored_files"))

    return [_json_record(record) for record in records]


def _collection_failure_summary(
    *,
    skipped_files: Sequence[str | Path] | None = None,
    missing_files: Sequence[str | Path] | None = None,
    errors: Sequence[Mapping[str, Any]] | None = None,
    errored_files: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Build collection-report failure summaries from collector file lists.

    Parameters
    ----------
    skipped_files : sequence of str or pathlib.Path, optional
        Parquet files skipped because normal-termination columns failed.
    missing_files : sequence of str or pathlib.Path, optional
        Expected target parquet files that were not present.
    errors : sequence of mapping, optional
        Collector read errors with ``file`` and ``error`` fields.
    errored_files : sequence of str or pathlib.Path, optional
        Files that produced read errors.

    Returns
    -------
    list of dict
        JSON-serializable records suitable for ``collection_report.json``.
    """
    payload = {
        "skipped_files": [str(path) for path in skipped_files or []],
        "missing_files": [str(path) for path in missing_files or []],
        "errors": [dict(error) for error in errors or []],
        "errored_files": [str(path) for path in errored_files or []],
    }
    return _summarize_collection_failures(payload)


def _inspect_path(path: Path) -> list[dict[str, Any]]:
    """Inspect a path source and return failure records."""
    if path.is_dir():
        report_path = path / "collection_report.json"
        if report_path.exists():
            return _inspect_report_path(report_path)
        parquet_files = sorted(path.glob("*.parquet"))
        records: list[dict[str, Any]] = []
        for parquet in parquet_files:
            records.extend(_failure_records_from_parquet(parquet, source="parquet"))
        return records

    if path.suffix.lower() == ".json":
        return _inspect_report_path(path)

    return _failure_records_from_parquet(path, source="parquet")


def _inspect_report_path(path: Path) -> list[dict[str, Any]]:
    """Read a collection report and return failure records."""
    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        return [_read_error_record(path, error=str(exc), source="collection_report")]
    records = _summarize_collection_failures(report)
    for record in records:
        if record.get("file") is None:
            record["file"] = str(path)
        record.setdefault("source", "collection_report")
    return records


def _failure_records_from_parquet(path: Path, *, source: str | None = None) -> list[dict[str, Any]]:
    """Read one parquet and return row-level failure records."""
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return [_read_error_record(path, error=str(exc), source=source)]

    records = _summarize_dataframe_failures(df, file=path, source=source)
    if records:
        return records

    return [_unknown_skipped_record(path, source=source)] if source == "skipped_files" else []


def _records_frame(records: list[dict[str, Any]], *, detail: FailureDetail) -> pd.DataFrame:
    """Return a stable-column dataframe from failure records."""
    columns = SUMMARY_COLUMNS if detail == "summary" else [*SUMMARY_COLUMNS, *FULL_EXTRA_COLUMNS]
    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(records)
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def _missing_output_record(file_name: Any) -> dict[str, Any]:
    """Build a missing-output collection record."""
    path = Path(str(file_name))
    return _json_record(
        {
            "target": _target_from_path(path),
            "file": str(path),
            "row_index": None,
            "failed_stage": None,
            "failed_nt_cols": [],
            "failed_nt_values": {},
            "error": "Expected workflow output was not found.",
            "backend_hint": None,
            "backend_column": None,
            "last_checkpoint": None,
            "target_dir": str(path.parent),
            "problem": "missing_output",
            "source": "missing_files",
        }
    )


def _read_error_record(file_name: Any, *, error: Any = None, source: str | None = None) -> dict[str, Any]:
    """Build a read-error collection record."""
    path = None if file_name is None else Path(str(file_name))
    return _json_record(
        {
            "target": _target_from_path(path),
            "file": _path_text(path),
            "row_index": None,
            "failed_stage": None,
            "failed_nt_cols": [],
            "failed_nt_values": {},
            "error": None if error is None else str(error),
            "backend_hint": None,
            "backend_column": None,
            "last_checkpoint": None,
            "target_dir": _path_text(path.parent if path is not None else None),
            "problem": "read_error",
            "source": source,
        }
    )


def _unknown_skipped_record(path: Path, *, source: str | None = None) -> dict[str, Any]:
    """Build a skipped-output record when no row-level failure is visible."""
    return _json_record(
        {
            "target": _target_from_path(path),
            "file": str(path),
            "row_index": None,
            "failed_stage": None,
            "failed_nt_cols": [],
            "failed_nt_values": {},
            "error": "Skipped output had no row-level failed normal-termination columns.",
            "backend_hint": None,
            "backend_column": None,
            "last_checkpoint": str(path),
            "target_dir": str(path.parent),
            "problem": "unknown_skipped",
            "source": source,
        }
    )


def _target_name(target: str | None, df: pd.DataFrame, file_path: Path | None) -> str | None:
    """Return target from explicit value, dataframe attrs, or file path."""
    if target:
        return str(target)
    workflow = df.attrs.get("frust_workflow", {})
    if isinstance(workflow, Mapping) and workflow.get("target") is not None:
        return str(workflow["target"])
    return _target_from_path(file_path)


def _target_from_path(path: Path | None) -> str | None:
    """Infer target tag from a target parquet path."""
    if path is None:
        return None
    return path.parent.name or None


def _stage_from_nt_col(column: str) -> str:
    """Return the stage prefix for a normal-termination column."""
    text = str(column)
    for suffix in ("-NT", "-normal_termination"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _is_failed_value(value: Any) -> bool:
    """Return whether a normal-termination value represents failure."""
    if _is_missing(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in FALSE_STRINGS
    try:
        return not bool(value)
    except Exception:
        return True


def _is_missing(value: Any) -> bool:
    """Return whether ``value`` is missing without treating arrays as scalars."""
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return missing
    try:
        return bool(missing)
    except Exception:
        return False


def _row_error(row: pd.Series, failed_cols: Sequence[str]) -> str | None:
    """Return the most relevant row-level error text."""
    for nt_col in failed_cols:
        stage = _stage_from_nt_col(nt_col)
        error_col = f"{stage}-error"
        if error_col in row.index:
            value = row.get(error_col)
            if not _is_missing(value) and str(value).strip():
                return _clean_text(value, limit=500)
    return None


def _row_backend_hint(row: pd.Series, failed_stage: str) -> tuple[str | None, str | None]:
    """Return a compact hint from saved backend-output text."""
    columns = [
        col
        for col in row.index
        if str(col).startswith(f"{failed_stage}-") and str(col).endswith(BACKEND_SUFFIXES)
    ]
    if not columns:
        columns = [col for col in row.index if str(col).endswith(BACKEND_SUFFIXES)]

    for column in columns:
        value = row.get(column)
        if _is_missing(value) or not isinstance(value, str):
            continue
        hint = _backend_hint_from_text(value)
        if hint:
            return hint, str(column)
    return None, None


def _backend_hint_from_text(text: str) -> str | None:
    """Extract one high-signal backend failure line."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        normalized = f" {line.lower()} "
        if any(pattern in normalized for pattern in BACKEND_PATTERNS):
            return _clean_text(line, limit=240)
    return None


def _clean_text(value: Any, *, limit: int) -> str:
    """Return one-line text clipped to ``limit`` characters."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _path_text(path: Path | None) -> str | None:
    """Return ``path`` as text or ``None``."""
    return None if path is None else str(path)


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a record mapping."""
    return {str(key): _json_value(value) for key, value in record.items()}


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable value."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar."""
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _string_items(value: Any) -> list[str]:
    """Return string items from a report list field."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
