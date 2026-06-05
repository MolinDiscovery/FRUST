"""Timing helpers for FRUST workflow and calculator provenance."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


def utc_timestamp() -> str:
    """Return the current UTC timestamp in a compact ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def monotonic_seconds() -> float:
    """Return a monotonic timestamp suitable for elapsed-time measurement."""
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    """Return seconds elapsed since a monotonic start timestamp."""
    return round(max(0.0, time.perf_counter() - float(start)), 6)


def format_duration(seconds: Any) -> str | None:
    """Format elapsed seconds for compact human-facing tables and logs."""
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(total):
        return None
    total = max(0.0, total)
    if total < 60:
        return f"{total:.1f}s"

    whole = int(round(total))
    minutes, sec = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def row_timing_stats(values: Sequence[float]) -> dict[str, float] | None:
    """Return compact descriptive statistics for row elapsed times."""
    numeric = sorted(float(value) for value in values if _is_finite_number(value))
    if not numeric:
        return None
    return {
        "min_s": round(numeric[0], 6),
        "mean_s": round(mean(numeric), 6),
        "median_s": round(median(numeric), 6),
        "p95_s": round(_percentile(numeric, 0.95), 6),
        "max_s": round(numeric[-1], 6),
    }


def build_step_timing(
    *,
    started_at: str,
    finished_at: str,
    elapsed_s: float,
    input_rows: int,
    output_rows: int,
    processed_rows: int,
    skipped_rows: int,
    row_records: Sequence[dict[str, Any]],
    slowest: int = 5,
) -> dict[str, Any]:
    """Build calculator-stage timing metadata for ``frust_steps``."""
    elapsed_values = [
        float(record["elapsed_s"])
        for record in row_records
        if not record.get("skipped") and _is_finite_number(record.get("elapsed_s"))
    ]
    slowest_rows = sorted(
        [
            _jsonable_mapping(record)
            for record in row_records
            if not record.get("skipped")
        ],
        key=lambda record: float(record.get("elapsed_s") or 0.0),
        reverse=True,
    )[: max(0, int(slowest))]
    return {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": round(float(elapsed_s), 6),
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "processed_rows": int(processed_rows),
        "skipped_rows": int(skipped_rows),
        "row_elapsed_s": row_timing_stats(elapsed_values),
        "slowest_rows": slowest_rows,
    }


def build_workflow_timing_record(
    *,
    workflow: str,
    target: str,
    group: str | None,
    stage: str | None,
    kind: str,
    started_at: str,
    finished_at: str,
    elapsed_s: float,
    input_rows: int | None = None,
    output_rows: int | None = None,
    job_id: str | int | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-friendly workflow timing record."""
    return _jsonable_mapping(
        {
            "schema_version": 1,
            "workflow": workflow,
            "target": target,
            "group": group,
            "stage": stage,
            "kind": kind,
            "job_id": job_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_s": round(float(elapsed_s), 6),
            "input_rows": input_rows,
            "output_rows": output_rows,
            "resources": resources,
        }
    )


def append_workflow_timing(
    attrs: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Append one workflow timing record to dataframe attrs in place."""
    timing = attrs.setdefault(
        "frust_workflow_timing",
        {"schema_version": 1, "records": []},
    )
    if not isinstance(timing, dict):
        attrs["frust_workflow_timing"] = timing = {
            "schema_version": 1,
            "records": [],
        }
    records = timing.setdefault("records", [])
    if isinstance(records, list):
        records.append(_jsonable_mapping(record))


def write_timing_sidecar(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one workflow timing sidecar JSON file."""
    sidecar = Path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(_jsonable_mapping(payload), indent=2, sort_keys=True),
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * float(q)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _jsonable_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _jsonable_value(value)
        for key, value in mapping.items()
        if value is not None
    }


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
