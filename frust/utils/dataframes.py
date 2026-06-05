"""Dataframe inspection helpers."""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from frust.schema import energy_columns, infer_group_columns, normalize_dataframe
from frust.utils.timing import format_duration

_PUBCHEM_CACHE_COLUMNS = [
    "canonical_smiles",
    "input_smiles",
    "pubchem_iupac",
    "pubchem_cid",
    "lookup_status",
    "lookup_error",
]
_STEP_VARIANT_RE = re.compile(r"^(?P<base>.+)__variant_\d{3}$")
_SUMMARY_COLUMNS = [
    "engine",
    "calc_name",
    "mode",
    "elapsed",
    "mean_row",
    "max_row",
    "core_hours",
    "options",
    "columns",
    "n_columns",
    "n_variants",
    "n_sources",
    "lowest",
    "filter_energy_col",
    "input_rows",
    "output_rows",
    "dropped_rows",
    "n_cores",
    "memory_gb",
    "xtra_inp_str",
]
_SUMMARY_STABLE_COLUMNS = [
    "engine",
    "calc_name",
    "mode",
    "options",
    "columns",
    "n_columns",
    "lowest",
    "filter_energy_col",
    "n_cores",
    "memory_gb",
]
_SUMMARY_COUNT_COLUMNS = ["input_rows", "output_rows", "dropped_rows"]


def merge_dataframe_attrs(
    dfs: Sequence[pd.DataFrame],
    *,
    source_files: Sequence[str | Path] | None = None,
    skipped_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Merge FRUST dataframe attrs across concatenated result tables.

    Parameters
    ----------
    dfs : sequence of pandas.DataFrame
        Dataframes whose attrs should be merged.
    source_files : sequence of str or pathlib.Path, optional
        File labels matching ``dfs``. These are recorded in merged step
        metadata and in the merge provenance block.
    skipped_files : sequence of str or pathlib.Path, optional
        Files skipped during merge filtering.

    Returns
    -------
    dict
        Merged attrs suitable for assigning to a concatenated dataframe.
    """
    frames = list(dfs)
    sources = _source_labels(source_files, len(frames))
    skipped = [str(path) for path in (skipped_files or [])]
    attr_items = [
        (source, getattr(frame, "attrs", {}) or {})
        for source, frame in zip(sources, frames)
    ]

    merge_info: dict[str, Any] = {
        "input_files": sources,
        "skipped_files": skipped,
        "step_variants": {},
        "attr_conflicts": {},
    }
    merged: dict[str, Any] = {}

    steps = _merge_frust_steps(attr_items, merge_info)
    if steps:
        merged["frust_steps"] = steps

    conformers = _merge_frust_conformers(attr_items)
    if conformers:
        merged["frust_conformers"] = conformers

    workflow_timing = _merge_workflow_timing(attr_items)
    if workflow_timing:
        merged["frust_workflow_timing"] = workflow_timing

    attr_keys = sorted(
        {
            key
            for _, attrs in attr_items
            for key in attrs
            if key
            not in {
                "frust_steps",
                "frust_conformers",
                "frust_merge",
                "frust_workflow_timing",
            }
        },
        key=str,
    )
    for key in attr_keys:
        groups: list[dict[str, Any]] = []
        for source, attrs in attr_items:
            if key not in attrs:
                continue
            value = attrs[key]
            fingerprint = _attr_fingerprint(value)
            for group in groups:
                if group["fingerprint"] == fingerprint:
                    group["source_files"].append(source)
                    break
            else:
                groups.append(
                    {
                        "fingerprint": fingerprint,
                        "value": copy.deepcopy(value),
                        "source_files": [source],
                    }
                )

        if len(groups) == 1:
            merged[key] = groups[0]["value"]
        elif groups:
            merge_info["attr_conflicts"][str(key)] = [
                {
                    "source_files": group["source_files"],
                    "value_repr": repr(group["value"]),
                }
                for group in groups
            ]

    merge_info["n_input_files"] = len(sources)
    merge_info["n_skipped_files"] = len(skipped)
    merge_info["n_merged_files"] = len(frames)
    merged["frust_merge"] = merge_info
    return merged


def show_steps(df: pd.DataFrame, *, detail: str = "summary") -> pd.DataFrame:
    """Summarize FRUST calculation step metadata as a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST results dataframe. The helper reads ``df.attrs["frust_steps"]``
        when present.
    detail : {"summary", "full"}, optional
        Level of detail to return. ``"summary"`` is intended for quick
        inspection, collapses merged step variants such as
        ``"xtb_opt__variant_001"``, and omits verbose executable and environment
        provenance. ``"full"`` includes every available metadata column and
        keeps stored variants as separate rows.

    Returns
    -------
    pandas.DataFrame
        Summary rows or full provenance rows indexed by step name. Columns that
        are entirely empty are removed.
    """
    if detail not in {"summary", "full"}:
        raise ValueError("detail must be 'summary' or 'full'")

    rows = []
    conformer_row = _conformer_generation_step_row(df)
    if conformer_row is not None:
        rows.append(conformer_row)

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

        filtering = step.get("filtering", {})
        if not isinstance(filtering, Mapping):
            filtering = {}

        row_counts = step.get("row_counts", {})
        if not isinstance(row_counts, Mapping):
            row_counts = {}

        timing = step.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        row_elapsed = timing.get("row_elapsed_s", {})
        if not isinstance(row_elapsed, Mapping):
            row_elapsed = {}

        columns = step.get("columns")
        elapsed_s = timing.get("elapsed_s")
        mean_row_s = row_elapsed.get("mean_s")
        max_row_s = row_elapsed.get("max_s")
        n_cores = resources.get("n_cores")

        rows.append(
            {
                "step": step_name,
                "engine": step.get("engine"),
                "calc_name": calculator.get("name"),
                "mode": calculator.get("mode"),
                "backend": calculator.get("backend"),
                "elapsed_s": elapsed_s,
                "elapsed": format_duration(elapsed_s),
                "mean_row_s": mean_row_s,
                "mean_row": format_duration(mean_row_s),
                "max_row_s": max_row_s,
                "max_row": format_duration(max_row_s),
                "core_hours": _core_hours(elapsed_s, n_cores),
                "processed_rows": timing.get("processed_rows"),
                "skipped_rows": timing.get("skipped_rows"),
                "options": _format_keys(step.get("options")),
                "columns": _format_list(columns),
                "n_columns": len(columns) if isinstance(columns, list) else 0,
                "lowest": filtering.get("lowest"),
                "filter_energy_col": filtering.get("energy_col"),
                "input_rows": row_counts.get("input_rows", filtering.get("input_rows")),
                "output_rows": row_counts.get("output_rows", filtering.get("output_rows")),
                "dropped_rows": row_counts.get("dropped_rows", filtering.get("dropped_rows")),
                "n_filter_groups": filtering.get("n_groups"),
                "n_cores": n_cores,
                "memory_gb": resources.get("memory_gb"),
                "xtra_inp_str": input_data.get("xtra_inp_str"),
                "detailed_inp_str": input_data.get("detailed_inp_str"),
                "input": _format_mapping(input_data),
                "executables": _format_paths(calculator.get("executables")),
                "environment": _format_paths(calculator.get("environment")),
                "gxtb": step.get("gxtb"),
                "gxtb_exe": step.get("gxtb_exe"),
                "gxtb_exe_source": step.get("gxtb_exe_source"),
                "_source_files": _step_source_files(step),
            }
        )

    if not rows:
        out = pd.DataFrame()
        out.index.name = "step"
        return out

    out = pd.DataFrame(rows).set_index("step")
    if detail == "summary":
        out = _summarize_step_rows(out)
        out = out[[column for column in _SUMMARY_COLUMNS if column in out.columns]]
    else:
        out = _format_full_step_rows(out)
    return out.dropna(axis=1, how="all")


def show_timing(df: pd.DataFrame, *, detail: str = "summary") -> pd.DataFrame:
    """Summarize FRUST timing metadata as a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST results dataframe. The helper reads timing metadata from
        ``df.attrs["frust_steps"]`` and ``df.attrs["frust_workflow_timing"]``.
    detail : {"summary", "rows", "workflow"}, optional
        Timing view to return. ``"summary"`` shows one row per calculator
        stage, ``"rows"`` shows stored slowest-row diagnostics, and
        ``"workflow"`` shows workflow target, stage, and stage-group records.

    Returns
    -------
    pandas.DataFrame
        Timing summary table. Empty timing metadata returns an empty dataframe.
    """
    if detail not in {"summary", "rows", "workflow"}:
        raise ValueError("detail must be 'summary', 'rows', or 'workflow'")

    if detail == "workflow":
        return _workflow_timing_frame(df)
    if detail == "rows":
        return _row_timing_frame(df)
    return _step_timing_frame(df)


def _step_timing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return one timing summary row per calculator step."""
    rows: list[dict[str, Any]] = []
    for step_name, step in _steps_mapping(df).items():
        if not isinstance(step, Mapping):
            continue
        timing = step.get("timing", {})
        if not isinstance(timing, Mapping):
            continue
        calculator = step.get("calculator", {})
        if not isinstance(calculator, Mapping):
            calculator = {}
        resources = calculator.get("resources", {})
        if not isinstance(resources, Mapping):
            resources = {}
        row_elapsed = timing.get("row_elapsed_s", {})
        if not isinstance(row_elapsed, Mapping):
            row_elapsed = {}

        elapsed_s = timing.get("elapsed_s")
        mean_row_s = row_elapsed.get("mean_s")
        max_row_s = row_elapsed.get("max_s")
        n_cores = resources.get("n_cores")
        rows.append(
            {
                "step": step_name,
                "engine": step.get("engine"),
                "elapsed": format_duration(elapsed_s),
                "elapsed_s": elapsed_s,
                "input_rows": timing.get("input_rows"),
                "output_rows": timing.get("output_rows"),
                "processed_rows": timing.get("processed_rows"),
                "skipped_rows": timing.get("skipped_rows"),
                "mean_row": format_duration(mean_row_s),
                "mean_row_s": mean_row_s,
                "max_row": format_duration(max_row_s),
                "max_row_s": max_row_s,
                "n_cores": n_cores,
                "core_hours": _core_hours(elapsed_s, n_cores),
                "started_at": timing.get("started_at"),
                "finished_at": timing.get("finished_at"),
            }
        )

    if not rows:
        out = pd.DataFrame()
        out.index.name = "step"
        return out
    return pd.DataFrame(rows).set_index("step").dropna(axis=1, how="all")


def _row_timing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return stored slowest-row timing diagnostics."""
    rows: list[dict[str, Any]] = []
    for step_name, step in _steps_mapping(df).items():
        if not isinstance(step, Mapping):
            continue
        timing = step.get("timing", {})
        if not isinstance(timing, Mapping):
            continue
        for record in timing.get("slowest_rows", []) or []:
            if not isinstance(record, Mapping):
                continue
            elapsed_s = record.get("elapsed_s")
            rows.append(
                {
                    "step": step_name,
                    "row_number": record.get("row_number"),
                    "row_index": record.get("row_index"),
                    "label": record.get("label"),
                    "cid": record.get("cid"),
                    "elapsed": format_duration(elapsed_s),
                    "elapsed_s": elapsed_s,
                    "normal_termination": record.get("normal_termination"),
                }
            )
    if not rows:
        out = pd.DataFrame()
        out.index.name = "step"
        return out
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def _workflow_timing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return workflow timing records stored in dataframe attrs."""
    timing = df.attrs.get("frust_workflow_timing", {})
    if not isinstance(timing, Mapping):
        return pd.DataFrame()
    records = timing.get("records", [])
    if not isinstance(records, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        resources = record.get("resources")
        if not isinstance(resources, Mapping):
            resources = {}
        elapsed_s = record.get("elapsed_s")
        n_cores = resources.get("n_cores")
        rows.append(
            {
                "workflow": record.get("workflow"),
                "target": record.get("target"),
                "group": record.get("group"),
                "stage": record.get("stage"),
                "kind": record.get("kind"),
                "elapsed": format_duration(elapsed_s),
                "elapsed_s": elapsed_s,
                "input_rows": record.get("input_rows"),
                "output_rows": record.get("output_rows"),
                "n_cores": n_cores,
                "memory_gb": resources.get("memory_gb"),
                "core_hours": _core_hours(elapsed_s, n_cores),
                "job_id": record.get("job_id"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "source_files": _format_list(record.get("source_files")),
            }
        )
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def _summarize_step_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse stored step variants into one compact row per logical step."""
    grouped: dict[str, list[pd.Series]] = {}
    for step_name, row in rows.iterrows():
        grouped.setdefault(_base_step_name(str(step_name)), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for step_name, group_rows in grouped.items():
        source_files = _unique_strings(
            [
                source
                for row in group_rows
                for source in _coerce_source_files(row.get("_source_files"))
            ]
        )
        weights = [
            max(1, len(_coerce_source_files(row.get("_source_files"))))
            for row in group_rows
        ]

        summary: dict[str, Any] = {
            "step": step_name,
            "n_variants": len(group_rows),
            "n_sources": len(source_files) if source_files else None,
        }
        for column in _SUMMARY_STABLE_COLUMNS:
            summary[column] = _summarize_mixed_values(
                [row.get(column) for row in group_rows]
            )
        for column in _SUMMARY_COUNT_COLUMNS:
            summary[column] = _weighted_sum(
                [row.get(column) for row in group_rows],
                weights,
            )
        elapsed_s = _numeric_sum([row.get("elapsed_s") for row in group_rows])
        mean_row_s = _weighted_mean(
            [row.get("mean_row_s") for row in group_rows],
            [row.get("processed_rows") for row in group_rows],
        )
        max_row_s = _numeric_max([row.get("max_row_s") for row in group_rows])
        summary["elapsed"] = format_duration(elapsed_s)
        summary["mean_row"] = format_duration(mean_row_s)
        summary["max_row"] = format_duration(max_row_s)
        summary["core_hours"] = _numeric_sum(
            [row.get("core_hours") for row in group_rows]
        )
        summary["xtra_inp_str"] = _summarize_mixed_values(
            [
                _compact_multiline_text(row.get("xtra_inp_str"))
                for row in group_rows
            ]
        )
        summary_rows.append(summary)

    if not summary_rows:
        out = pd.DataFrame()
        out.index.name = "step"
        return out
    return pd.DataFrame(summary_rows).set_index("step")


def _format_full_step_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Format internal full-detail helper columns for display."""
    out = rows.copy()
    if "_source_files" in out.columns:
        source_values = out["_source_files"].map(
            lambda value: _format_list(_coerce_source_files(value))
        )
        if source_values.notna().any():
            out["source_files"] = source_values
        out = out.drop(columns=["_source_files"])
    return out


def _base_step_name(step_name: str) -> str:
    """Return the logical step name for a stored variant row."""
    match = _STEP_VARIANT_RE.match(step_name)
    if match is None:
        return step_name
    return match.group("base")


def _step_source_files(step: Mapping[str, Any]) -> list[str]:
    """Return source files recorded on one step metadata block."""
    return _coerce_source_files(step.get("source_files"))


def _coerce_source_files(value: Any) -> list[str]:
    """Normalize source-file metadata to a unique string list."""
    if isinstance(value, (list, tuple, set)):
        return _unique_strings(value)
    if _none_if_missing(value) is None:
        return []
    return [str(value)]


def _weighted_sum(values: Sequence[Any], weights: Sequence[int]) -> int | float | None:
    """Return a weighted numeric sum while ignoring missing values."""
    total = 0.0
    any_value = False
    for value, weight in zip(values, weights):
        value = _none_if_missing(value)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        total += numeric * int(weight)
        any_value = True
    if not any_value:
        return None
    if total.is_integer():
        return int(total)
    return total


def _numeric_sum(values: Sequence[Any]) -> float | None:
    """Return a numeric sum while ignoring missing values."""
    total = 0.0
    any_value = False
    for value in values:
        value = _none_if_missing(value)
        if value is None:
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
        any_value = True
    return total if any_value else None


def _numeric_max(values: Sequence[Any]) -> float | None:
    """Return the maximum numeric value while ignoring missing values."""
    numeric = []
    for value in values:
        value = _none_if_missing(value)
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(numeric) if numeric else None


def _numeric_min(values: Sequence[Any]) -> float | None:
    """Return the minimum numeric value while ignoring missing values."""
    numeric = []
    for value in values:
        value = _none_if_missing(value)
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return min(numeric) if numeric else None


def _weighted_mean(values: Sequence[Any], weights: Sequence[Any]) -> float | None:
    """Return a weighted mean while ignoring missing values."""
    total = 0.0
    total_weight = 0.0
    for value, weight in zip(values, weights):
        value = _none_if_missing(value)
        weight = _none_if_missing(weight)
        if value is None or weight is None:
            continue
        try:
            numeric = float(value)
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if numeric_weight <= 0:
            continue
        total += numeric * numeric_weight
        total_weight += numeric_weight
    if total_weight <= 0:
        return None
    return total / total_weight


def _core_hours(elapsed_s: Any, n_cores: Any) -> float | None:
    """Return core-hours from elapsed seconds and core count."""
    elapsed_s = _none_if_missing(elapsed_s)
    n_cores = _none_if_missing(n_cores)
    if elapsed_s is None or n_cores is None:
        return None
    try:
        return round(float(elapsed_s) * float(n_cores) / 3600.0, 6)
    except (TypeError, ValueError):
        return None


def _min_text(values: Sequence[Any]) -> str | None:
    """Return the lexicographically earliest non-empty text value."""
    texts = [str(value) for value in values if _none_if_missing(value) is not None]
    return min(texts) if texts else None


def _max_text(values: Sequence[Any]) -> str | None:
    """Return the lexicographically latest non-empty text value."""
    texts = [str(value) for value in values if _none_if_missing(value) is not None]
    return max(texts) if texts else None


def _summarize_mixed_values(values: Sequence[Any]) -> Any:
    """Return one value, a compact mixed label, or a mixed count."""
    unique: list[Any] = []
    fingerprints = set()
    for value in values:
        value = _none_if_missing(value)
        if value is None:
            continue
        fingerprint = _attr_fingerprint(value)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(value)

    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]

    labels = [str(value) for value in unique]
    if (
        len(labels) <= 3
        and all("\n" not in label for label in labels)
        and all(len(label) <= 80 for label in labels)
    ):
        return "mixed: " + "; ".join(labels)
    return f"mixed ({len(labels)} values)"


def _compact_multiline_text(value: Any) -> str | None:
    """Return text with multiline blocks collapsed to one display line."""
    value = _none_if_missing(value)
    if value is None:
        return None
    text = str(value)
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    first = lines[0].strip()
    if not first:
        first = next((line.strip() for line in lines if line.strip()), "")
    return f"{first} ... ({len(lines)} lines)"


def map_substrate_names(
    df: pd.DataFrame,
    *,
    smiles_col: str = "substrate_smiles",
    name_col: str = "substrate_name",
    mapped_col: str = "substrate_pubchem_iupac",
    cid_col: str = "substrate_pubchem_cid",
    original_col: str | None = None,
    replace: bool = True,
    add_metadata: bool = False,
    cache_path: str | Path | None = None,
    force: bool = False,
    sanitize_names: bool = True,
    strict: bool = False,
    inplace: bool = False,
    return_mapping: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Map substrate SMILES to PubChem names and annotate a FRUST dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST dataframe containing a substrate SMILES column.
    smiles_col : str, optional
        Column containing substrate SMILES strings.
    name_col : str, optional
        Column to replace when ``replace=True``.
    mapped_col : str, optional
        Column that stores the raw PubChem IUPAC name.
    cid_col : str, optional
        Column that stores the PubChem CID.
    original_col : str or None, optional
        Optional column used to preserve original names before replacement.
    replace : bool, optional
        Replace ``name_col`` with mapped PubChem names for successful lookups.
    add_metadata : bool, optional
        Add canonical SMILES, PubChem IUPAC, PubChem CID, and lookup-status
        columns to the returned dataframe. Metadata columns are also added when
        ``replace=False`` so the call has a visible dataframe effect.
    cache_path : str, pathlib.Path, or None, optional
        CSV cache path. ``None`` uses ``~/.cache/frust/pubchem_names.csv``.
    force : bool, optional
        Re-query PubChem even when a cache entry exists.
    sanitize_names : bool, optional
        Sanitize replacement names for FRUST/file-name compatibility.
    strict : bool, optional
        Raise an error if any unique substrate SMILES cannot be mapped.
    inplace : bool, optional
        Mutate ``df`` directly instead of returning a copy.
    return_mapping : bool, optional
        Return ``(dataframe, mapping_dataframe)``.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
        Annotated dataframe, optionally with the unique SMILES mapping table.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"cannot map substrate names: {smiles_col!r} is not a column")

    from frust.utils.mols import (
        canonicalize_smiles,
        lookup_pubchem_name,
        sanitize_molecule_name,
    )

    out = df if inplace else df.copy()
    if not inplace:
        out.attrs.update(getattr(df, "attrs", {}))

    cache_file = _resolve_pubchem_cache_path(cache_path)
    cache_records = _read_pubchem_cache(cache_file)

    canonical_series = out[smiles_col].map(
        lambda value: _canonical_smiles_or_none(value, canonicalize_smiles)
    )
    unique_inputs: dict[str, str] = {}
    for raw_smiles, canonical in zip(out[smiles_col], canonical_series):
        if canonical is not None and canonical not in unique_inputs:
            unique_inputs[canonical] = str(raw_smiles).strip()

    mapping_records: list[dict[str, Any]] = []
    records_by_key: dict[str, dict[str, Any]] = {}
    cache_changed = False

    for canonical, raw_smiles in unique_inputs.items():
        if not force and canonical in cache_records:
            cache_record = cache_records[canonical]
            record = _mapping_record_from_cache(
                cache_record,
                raw_smiles=raw_smiles,
                smiles_col=smiles_col,
                mapped_col=mapped_col,
                cid_col=cid_col,
                cache_hit=True,
            )
        else:
            lookup = lookup_pubchem_name(raw_smiles)
            cache_record = _cache_record_from_lookup(lookup, raw_smiles, canonical)
            cache_records[canonical] = cache_record
            cache_changed = True
            record = _mapping_record_from_cache(
                cache_record,
                raw_smiles=raw_smiles,
                smiles_col=smiles_col,
                mapped_col=mapped_col,
                cid_col=cid_col,
                cache_hit=False,
            )

        mapping_records.append(record)
        records_by_key[canonical] = record

    if cache_changed:
        _write_pubchem_cache(cache_file, cache_records)

    failures = [
        record
        for record in mapping_records
        if record["lookup_status"] != "success"
    ]
    if strict and failures:
        failed = ", ".join(
            f"{record[smiles_col]!r} ({record['lookup_status']})"
            for record in failures
        )
        raise ValueError(f"PubChem lookup failed for substrate SMILES: {failed}")

    mapped_names = canonical_series.map(
        lambda key: _record_value(records_by_key.get(key), mapped_col)
    )
    mapped_cids = canonical_series.map(
        lambda key: _record_value(records_by_key.get(key), cid_col)
    )
    mapped_status = canonical_series.map(
        lambda key: _record_value(records_by_key.get(key), "lookup_status") or "missing_smiles"
    )

    if add_metadata or not replace:
        out["substrate_canonical_smiles"] = canonical_series
        out[mapped_col] = mapped_names
        out[cid_col] = mapped_cids
        out["substrate_pubchem_status"] = mapped_status

    if replace:
        if name_col not in out.columns:
            out[name_col] = pd.NA
        if original_col is not None and original_col not in out.columns:
            out[original_col] = out[name_col]

        success = mapped_status.eq("success") & mapped_names.notna()
        replacement_names = mapped_names.loc[success].map(
            lambda value: sanitize_molecule_name(str(value))
            if sanitize_names
            else str(value)
        )
        out.loc[success, name_col] = replacement_names

    mapping_df = pd.DataFrame(
        mapping_records,
        columns=[
            smiles_col,
            "canonical_smiles",
            mapped_col,
            cid_col,
            "lookup_status",
            "lookup_error",
            "cache_hit",
        ],
    )
    mapping_df.attrs.update(getattr(df, "attrs", {}))

    if return_mapping:
        return out, mapping_df
    return out


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


def _resolve_pubchem_cache_path(cache_path: str | Path | None) -> Path:
    """Resolve the PubChem name-cache path."""
    if cache_path is not None:
        return Path(cache_path).expanduser()
    return Path.home() / ".cache" / "frust" / "pubchem_names.csv"


def _read_pubchem_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Read PubChem cache records keyed by canonical SMILES."""
    if not cache_path.exists():
        return {}

    try:
        cache_df = pd.read_csv(cache_path, dtype=object)
    except Exception as exc:
        warnings.warn(
            f"Ignoring unreadable PubChem cache {cache_path}: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return {}

    missing = set(_PUBCHEM_CACHE_COLUMNS) - set(cache_df.columns)
    if missing:
        warnings.warn(
            f"Ignoring PubChem cache {cache_path}: missing columns {sorted(missing)}",
            UserWarning,
            stacklevel=2,
        )
        return {}

    cache_df = cache_df.dropna(subset=["canonical_smiles"])
    cache_df = cache_df.drop_duplicates("canonical_smiles", keep="last")
    records: dict[str, dict[str, Any]] = {}
    for record in cache_df[_PUBCHEM_CACHE_COLUMNS].to_dict("records"):
        normalized = {key: _none_if_missing(value) for key, value in record.items()}
        canonical = normalized.get("canonical_smiles")
        if canonical is not None:
            records[str(canonical)] = normalized
    return records


def _write_pubchem_cache(cache_path: Path, records: dict[str, dict[str, Any]]) -> None:
    """Write PubChem cache records."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame(records.values(), columns=_PUBCHEM_CACHE_COLUMNS)
    cache_df = cache_df.sort_values("canonical_smiles", na_position="last")
    cache_df.to_csv(cache_path, index=False)


def _cache_record_from_lookup(
    lookup: Mapping[str, Any],
    raw_smiles: str,
    canonical_smiles: str,
) -> dict[str, Any]:
    """Normalize one lookup result into the on-disk cache schema."""
    return {
        "canonical_smiles": canonical_smiles,
        "input_smiles": raw_smiles,
        "pubchem_iupac": _none_if_missing(lookup.get("pubchem_iupac")),
        "pubchem_cid": _none_if_missing(lookup.get("pubchem_cid")),
        "lookup_status": _none_if_missing(lookup.get("lookup_status")) or "error",
        "lookup_error": _none_if_missing(lookup.get("lookup_error")),
    }


def _mapping_record_from_cache(
    cache_record: Mapping[str, Any],
    *,
    raw_smiles: str,
    smiles_col: str,
    mapped_col: str,
    cid_col: str,
    cache_hit: bool,
) -> dict[str, Any]:
    """Convert one cache record to a user-facing mapping record."""
    return {
        smiles_col: raw_smiles,
        "canonical_smiles": _record_value(cache_record, "canonical_smiles"),
        mapped_col: _record_value(cache_record, "pubchem_iupac"),
        cid_col: _coerce_pubchem_cid(_record_value(cache_record, "pubchem_cid")),
        "lookup_status": _record_value(cache_record, "lookup_status") or "error",
        "lookup_error": _record_value(cache_record, "lookup_error"),
        "cache_hit": cache_hit,
    }


def _canonical_smiles_or_none(value: Any, canonicalize_smiles) -> str | None:
    """Canonicalize a possibly missing SMILES value."""
    if _is_missing_scalar(value):
        return None
    return canonicalize_smiles(str(value))


def _record_value(record: Mapping[str, Any] | None, key: str) -> Any:
    """Return a scalar record value with pandas missing values normalized."""
    if record is None:
        return None
    return _none_if_missing(record.get(key))


def _none_if_missing(value: Any) -> Any:
    """Convert pandas missing scalars and empty strings to ``None``."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _coerce_pubchem_cid(value: Any) -> Any:
    """Convert cache-loaded integer-like PubChem CIDs back to integers."""
    value = _none_if_missing(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a value should be treated as a missing scalar."""
    return _none_if_missing(value) is None


def _conformer_generation_step_row(df: pd.DataFrame) -> dict[str, Any] | None:
    """Build a compact ``show_steps`` row from conformer-generation attrs."""
    conformers = df.attrs.get("frust_conformers", {})
    if not isinstance(conformers, Mapping):
        return None

    records = [
        record
        for record in conformers.get("structures", []) or []
        if isinstance(record, Mapping)
    ]
    if not records:
        return None

    generated_total = conformers.get("total_generated_confs")
    if generated_total is None:
        generated_total = sum(
            int(record.get("generated_n_confs") or 0)
            for record in records
        )

    resolved_values = [
        _none_if_missing(record.get("resolved_n_confs"))
        for record in records
    ]
    resolved_numbers = [
        int(value)
        for value in resolved_values
        if value is not None
    ]
    resolved_total = (
        sum(resolved_numbers)
        if len(resolved_numbers) == len(records)
        else None
    )

    missing_total = None
    if resolved_total is not None:
        missing_total = int(resolved_total) - int(generated_total)

    requested_values = [
        _none_if_missing(record.get("requested_n_confs", conformers.get("requested_n_confs")))
        for record in records
    ]
    requested = _format_conformer_values(requested_values, none_label="auto")
    resolved = _format_conformer_values(resolved_values)
    options = _format_conformer_options(
        n_structures=len(records),
        requested=requested,
        resolved=resolved,
        generated=int(generated_total),
        missing=missing_total,
    )

    return {
        "step": "initial_conformers",
        "engine": "rdkit",
        "calc_name": conformers.get("source"),
        "mode": "embedding",
        "backend": None,
        "options": options,
        "columns": "coords_embedded",
        "n_columns": None,
        "lowest": None,
        "filter_energy_col": None,
        "input_rows": None,
        "output_rows": int(generated_total),
        "dropped_rows": None,
        "n_filter_groups": None,
        "n_cores": conformers.get("n_cores"),
        "memory_gb": None,
        "xtra_inp_str": None,
        "detailed_inp_str": None,
        "input": None,
        "executables": None,
        "environment": None,
        "gxtb": None,
        "gxtb_exe": None,
        "gxtb_exe_source": None,
    }


def _format_conformer_values(values: Sequence[Any], *, none_label: str | None = None) -> Any:
    """Format repeated conformer metadata values for summary display."""
    normalized = [
        none_label if _none_if_missing(value) is None else _none_if_missing(value)
        for value in values
    ]
    normalized = [value for value in normalized if value is not None]
    if not normalized:
        return None

    unique: list[Any] = []
    for value in normalized:
        if value in unique:
            continue
        unique.append(value)

    if len(unique) == 1:
        return unique[0]
    return "mixed: " + ", ".join(map(str, unique))


def _format_conformer_options(
    *,
    n_structures: int,
    requested: Any,
    resolved: Any,
    generated: int,
    missing: int | None,
) -> str:
    """Format conformer-generation counts without adding sparse columns."""
    parts = [f"structures={n_structures}"]
    if requested is not None:
        parts.append(f"requested={requested}")
    if resolved is not None:
        parts.append(f"resolved={resolved}")
    parts.append(f"generated={generated}")
    if missing is not None:
        parts.append(f"missing={missing}")
    return " ".join(parts)


def _source_labels(
    source_files: Sequence[str | Path] | None,
    n_frames: int,
) -> list[str]:
    """Return source labels matching a dataframe sequence."""
    if source_files is None:
        return [f"dataframe_{idx:03d}" for idx in range(n_frames)]
    labels = [str(path) for path in source_files]
    if len(labels) != n_frames:
        raise ValueError("source_files must match the number of dataframes")
    return labels


def _merge_frust_steps(
    attr_items: Sequence[tuple[str, Mapping[str, Any]]],
    merge_info: dict[str, Any],
) -> dict[str, Any]:
    """Merge ``frust_steps`` attrs with namespaced conflict variants."""
    variants_by_step: dict[str, list[dict[str, Any]]] = {}

    for source, attrs in attr_items:
        steps = attrs.get("frust_steps", {})
        if not isinstance(steps, Mapping):
            continue
        for step_name, step_meta in steps.items():
            if not isinstance(step_meta, Mapping):
                continue
            comparison_meta = copy.deepcopy(dict(step_meta))
            step_timing = comparison_meta.pop("timing", None)
            step_sources = _existing_step_sources(comparison_meta, source)
            variants = variants_by_step.setdefault(str(step_name), [])
            fingerprint = _attr_fingerprint(comparison_meta)
            for variant in variants:
                if variant["fingerprint"] == fingerprint:
                    variant["source_files"].extend(step_sources)
                    if isinstance(step_timing, Mapping):
                        variant["metadata"]["timing"] = _merge_step_timing(
                            variant["metadata"].get("timing"),
                            step_timing,
                        )
                    break
            else:
                metadata = comparison_meta
                if isinstance(step_timing, Mapping):
                    metadata["timing"] = copy.deepcopy(dict(step_timing))
                variants.append(
                    {
                        "fingerprint": fingerprint,
                        "metadata": metadata,
                        "source_files": step_sources,
                    }
                )

    merged_steps: dict[str, Any] = {}
    for step_name, variants in variants_by_step.items():
        variant_records = []
        for idx, variant in enumerate(variants):
            output_name = step_name if idx == 0 else f"{step_name}__variant_{idx:03d}"
            metadata = copy.deepcopy(variant["metadata"])
            source_files = _unique_strings(variant["source_files"])
            metadata["source_files"] = source_files
            merged_steps[output_name] = metadata
            variant_records.append(
                {
                    "step": output_name,
                    "source_files": source_files,
                }
            )
        if len(variant_records) > 1:
            merge_info["step_variants"][step_name] = variant_records

    return merged_steps


def _merge_step_timing(
    existing: Any,
    new: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge compatible calculator timing metadata across dataframe attrs."""
    if not isinstance(existing, Mapping):
        return copy.deepcopy(dict(new))

    merged = copy.deepcopy(dict(existing))
    new_dict = dict(new)
    merged["schema_version"] = 1
    merged["started_at"] = _min_text(
        [merged.get("started_at"), new_dict.get("started_at")]
    )
    merged["finished_at"] = _max_text(
        [merged.get("finished_at"), new_dict.get("finished_at")]
    )
    for key in (
        "elapsed_s",
        "input_rows",
        "output_rows",
        "processed_rows",
        "skipped_rows",
    ):
        total = _numeric_sum([merged.get(key), new_dict.get(key)])
        if total is not None:
            merged[key] = round(total, 6) if key == "elapsed_s" else int(total)

    existing_rows = _row_elapsed_mapping(merged.get("row_elapsed_s"))
    new_rows = _row_elapsed_mapping(new_dict.get("row_elapsed_s"))
    row_elapsed = {}
    min_s = _numeric_min([existing_rows.get("min_s"), new_rows.get("min_s")])
    max_s = _numeric_max([existing_rows.get("max_s"), new_rows.get("max_s")])
    mean_s = _weighted_mean(
        [existing_rows.get("mean_s"), new_rows.get("mean_s")],
        [existing.get("processed_rows"), new_dict.get("processed_rows")],
    )
    if min_s is not None:
        row_elapsed["min_s"] = round(min_s, 6)
    if mean_s is not None:
        row_elapsed["mean_s"] = round(mean_s, 6)
    if max_s is not None:
        row_elapsed["max_s"] = round(max_s, 6)
    if row_elapsed:
        merged["row_elapsed_s"] = row_elapsed

    slowest_rows = []
    for source in (existing, new_dict):
        rows = source.get("slowest_rows", [])
        if isinstance(rows, list):
            slowest_rows.extend(
                copy.deepcopy(dict(row))
                for row in rows
                if isinstance(row, Mapping)
            )
    if slowest_rows:
        merged["slowest_rows"] = sorted(
            slowest_rows,
            key=lambda row: float(row.get("elapsed_s") or 0.0),
            reverse=True,
        )[:5]
    return merged


def _merge_workflow_timing(
    attr_items: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Merge workflow timing records across result dataframes."""
    records: list[dict[str, Any]] = []
    for source, attrs in attr_items:
        timing = attrs.get("frust_workflow_timing", {})
        if not isinstance(timing, Mapping):
            continue
        for record in timing.get("records", []) or []:
            if not isinstance(record, Mapping):
                continue
            item = copy.deepcopy(dict(record))
            item["source_files"] = _unique_strings(
                [*(_coerce_source_files(item.get("source_files"))), source]
            )
            records.append(item)

    if not records:
        return {}
    return {
        "schema_version": 1,
        "records": records,
    }


def _row_elapsed_mapping(value: Any) -> Mapping[str, Any]:
    """Return row elapsed stats as a mapping."""
    return value if isinstance(value, Mapping) else {}


def _merge_frust_conformers(
    attr_items: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Merge compact conformer-generation attrs across result tables."""
    records: list[dict[str, Any]] = []
    top_level_values: dict[str, list[Any]] = {}

    for source, attrs in attr_items:
        conformers = attrs.get("frust_conformers", {})
        if not isinstance(conformers, Mapping):
            continue
        for key in ("source", "requested_n_confs", "n_cores"):
            if key in conformers:
                top_level_values.setdefault(key, []).append(conformers[key])
        for record in conformers.get("structures", []) or []:
            if not isinstance(record, Mapping):
                continue
            item = copy.deepcopy(dict(record))
            sources = item.get("source_files")
            if isinstance(sources, list) and sources:
                item["source_files"] = _unique_strings([*sources, source])
            else:
                item["source_files"] = [source]
            records.append(item)

    if not records:
        return {}

    merged: dict[str, Any] = {
        "schema_version": 1,
        "n_structures": len(records),
        "total_generated_confs": int(
            sum(
                int(record.get("generated_n_confs") or 0)
                for record in records
            )
        ),
        "structures": records,
    }
    for key, values in top_level_values.items():
        fingerprints = {_attr_fingerprint(value) for value in values}
        if len(fingerprints) == 1 and values:
            merged[key] = copy.deepcopy(values[0])
    return merged


def _existing_step_sources(step_meta: dict[str, Any], fallback: str) -> list[str]:
    """Pop existing step source files or return the current source label."""
    existing = step_meta.pop("source_files", None)
    if isinstance(existing, list) and existing:
        return [str(item) for item in existing]
    return [fallback]


def _unique_strings(values: Sequence[Any]) -> list[str]:
    """Return unique strings while preserving first-seen order."""
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _attr_fingerprint(value: Any) -> Any:
    """Return a stable comparable fingerprint for attrs metadata."""
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (repr(key), _attr_fingerprint(item))
                for key, item in sorted(value.items(), key=lambda entry: repr(entry[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_attr_fingerprint(item) for item in value))
    if isinstance(value, set):
        items = [_attr_fingerprint(item) for item in value]
        return ("set", tuple(sorted(items, key=repr)))
    if isinstance(value, Path):
        return ("path", str(value))
    if hasattr(value, "tolist"):
        try:
            return ("array", _attr_fingerprint(value.tolist()))
        except Exception:
            pass
    try:
        if bool(pd.isna(value)):
            return ("missing", None)
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool, type(None))):
        return ("scalar", value)
    return ("repr", type(value).__name__, repr(value))


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
