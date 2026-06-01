"""Dataframe inspection helpers."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from frust.schema import energy_columns, infer_group_columns, normalize_dataframe

_PUBCHEM_CACHE_COLUMNS = [
    "canonical_smiles",
    "input_smiles",
    "pubchem_iupac",
    "pubchem_cid",
    "lookup_status",
    "lookup_error",
]


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

    attr_keys = sorted(
        {
            key
            for _, attrs in attr_items
            for key in attrs
            if key not in {"frust_steps", "frust_conformers", "frust_merge"}
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
                "lowest": filtering.get("lowest"),
                "filter_energy_col": filtering.get("energy_col"),
                "input_rows": row_counts.get("input_rows", filtering.get("input_rows")),
                "output_rows": row_counts.get("output_rows", filtering.get("output_rows")),
                "dropped_rows": row_counts.get("dropped_rows", filtering.get("dropped_rows")),
                "n_filter_groups": filtering.get("n_groups"),
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
                "lowest",
                "filter_energy_col",
                "input_rows",
                "output_rows",
                "dropped_rows",
                "n_cores",
                "memory_gb",
                "xtra_inp_str",
            ]
        ]
    return out.dropna(axis=1, how="all")


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
            step_sources = _existing_step_sources(comparison_meta, source)
            variants = variants_by_step.setdefault(str(step_name), [])
            fingerprint = _attr_fingerprint(comparison_meta)
            for variant in variants:
                if variant["fingerprint"] == fingerprint:
                    variant["source_files"].extend(step_sources)
                    break
            else:
                variants.append(
                    {
                        "fingerprint": fingerprint,
                        "metadata": comparison_meta,
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
