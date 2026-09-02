"""Safe migrations for portable catalyst-screen result bundles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from frust.screen.references import (
    ReferenceLibrary,
    ReferenceRecord,
    _validate_reference_binding_target,
)
from frust.screen.runs import build_analysis
from frust.structures import StructureTarget, plan_targets


_BINDING_COLUMNS = (
    "structure_id",
    "custom_name",
    "system_name",
    "substrate_name",
    "catalyst_name",
    "state_id",
    "state_kind",
    "rpos",
    "molecule_role",
)


def repair_reference_bindings(
    run_dir: str | Path,
    *,
    apply: bool = False,
    backup_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Rebind reused reference rows in an existing portable screen run.

    The migration matches each imported reference to a current target by its
    checksum-protected scientific identity. It never edits the shared library
    or the immutable entries copied into the run. By default it performs a
    dry run and returns the proposed label changes.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Portable run directory containing ``manifest.json`` and
        ``calculations/references``.
    apply : bool, optional
        Write repaired aggregate parquets and rebuild analysis when ``True``.
        The default ``False`` only validates and reports proposed changes.
    backup_dir : str, pathlib.Path, or None, optional
        Backup destination used with ``apply=True``. When omitted, create a
        timestamped directory under ``run_dir/repair_backups``.

    Returns
    -------
    pandas.DataFrame
        One row per reused reference and aggregate file, showing old and new
        target labels. ``attrs`` records whether changes were applied and the
        backup location.

    Raises
    ------
    FileNotFoundError
        If the portable run or local reference index is incomplete.
    ValueError
        If a reused row cannot be matched unambiguously to a compatible target.

    Examples
    --------
    Inspect a proposed repair first, then apply it with an automatic backup::

        proposed = ft.screen.repair_reference_bindings("results")
        print(proposed[["reference_id", "old_substrate_name", "substrate_name"]])

        repaired = ft.screen.repair_reference_bindings("results", apply=True)
    """
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Catalyst-screen manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_type") != "catalyst_screen":
        raise ValueError(f"Unsupported run type {manifest.get('run_type')!r}")

    reference_dir = root / "calculations" / "references"
    aggregate_frames = {
        path: pd.read_parquet(path)
        for path in _reference_aggregate_paths(reference_dir)
    }
    reused_ids = {
        str(reference_id)
        for frame in aggregate_frames.values()
        if "reference_source" in frame
        for reference_id in frame.loc[
            frame["reference_source"].eq("shared_library"), "reference_id"
        ]
    }
    records = _local_reference_records(reference_dir, reference_ids=reused_ids)
    targets = _reference_targets(manifest)
    target_by_reference: dict[str, StructureTarget] = {}
    repaired_frames: dict[Path, pd.DataFrame] = {}
    report_rows: list[dict[str, Any]] = []

    for path, frame in aggregate_frames.items():
        if "reference_source" not in frame:
            raise ValueError(
                f"Reference aggregate has no reference_source column: {path}"
            )
        reused = frame["reference_source"].eq("shared_library")
        if not reused.any():
            continue
        updated = frame.copy()
        updated.attrs.clear()
        updated.attrs.update(frame.attrs)
        bindings: list[dict[str, Any]] = []
        file_changed = False
        for row_index in frame.index[reused]:
            reference_id = str(frame.at[row_index, "reference_id"])
            try:
                record = records[reference_id]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Run-local entry for reused reference {reference_id!r} is missing"
                ) from exc
            target = target_by_reference.get(reference_id)
            if target is None:
                target = _compatible_target(record, targets)
                target_by_reference[reference_id] = target
            materialized = record.materialize(target)
            binding = materialized.attrs["frust_reference_bindings"]["bindings"][0]
            bindings.append(binding)
            source = frame.loc[row_index]
            rebound = materialized.iloc[0]
            row_changed = any(
                column in frame
                and column in materialized
                and not _same_value(source.get(column), rebound.get(column))
                for column in _BINDING_COLUMNS
            )
            if row_changed:
                file_changed = True
                report_rows.append(
                    {
                        "file": str(path.relative_to(root)),
                        "reference_id": reference_id,
                        "state_id": target.state_id,
                        "target_id": target.target_id,
                        "old_system_name": source.get("system_name"),
                        "system_name": target.system.system_name,
                        "old_substrate_name": source.get("substrate_name"),
                        "substrate_name": target.system.substrate_name,
                        "old_catalyst_name": source.get("catalyst_name"),
                        "catalyst_name": target.system.catalyst_name,
                    }
                )
            for column in _BINDING_COLUMNS:
                if column in updated and column in materialized:
                    updated.at[row_index, column] = rebound[column]
        updated.attrs["frust_reference_bindings"] = {
            "schema_version": 1,
            "bindings": _unique_bindings(bindings),
        }
        if file_changed:
            repaired_frames[path] = updated

    report = pd.DataFrame(report_rows)
    report.attrs["applied"] = bool(apply)
    report.attrs["run_dir"] = str(root)
    if not apply:
        report.attrs["backup_dir"] = None
        return report
    if not repaired_frames:
        report.attrs["backup_dir"] = None
        return report

    destination = (
        Path(backup_dir).expanduser().resolve()
        if backup_dir is not None
        else root / "repair_backups" / _timestamp()
    )
    _backup_run_files(root, repaired_frames, destination)
    for path, frame in repaired_frames.items():
        _atomic_write_parquet(frame, path)

    analysis_report = build_analysis(root)
    run_report_path = root / "run_report.json"
    if run_report_path.exists():
        run_report = json.loads(run_report_path.read_text())
        run_report["analysis"] = analysis_report
        run_report["reference_binding_repair"] = {
            "applied_at": _utc_now(),
            "backup_dir": str(destination),
            "n_bound_rows": len(report),
        }
        _atomic_write_json(run_report, run_report_path)
    repair_report = {
        "schema_version": 1,
        "applied_at": _utc_now(),
        "backup_dir": str(destination),
        "n_bound_rows": len(report),
        "changes": report.to_dict("records"),
        "analysis": analysis_report,
    }
    _atomic_write_json(repair_report, root / "reference_binding_repair.json")
    report.attrs["backup_dir"] = str(destination)
    report.attrs["analysis"] = analysis_report
    return report


def _local_reference_records(
    reference_dir: Path,
    *,
    reference_ids: set[str],
) -> dict[str, ReferenceRecord]:
    """Load checksum-valid immutable records without initializing the library."""
    index_path = reference_dir / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Run-local reference index not found: {index_path}")
    library = ReferenceLibrary(reference_dir)
    index = pd.read_parquet(index_path)
    records: dict[str, ReferenceRecord] = {}
    for _, row in index.iterrows():
        reference_id = str(row["reference_id"])
        if reference_id not in reference_ids:
            continue
        path = reference_dir / str(row["entry_path"])
        ReferenceLibrary._validate_checksums(path)
        records[reference_id] = ReferenceRecord(library, reference_id, path)
    return records


def _reference_targets(manifest: dict[str, Any]) -> list[StructureTarget]:
    """Reconstruct lightweight reference targets from the portable manifest."""
    systems = pd.DataFrame(manifest.get("systems", []))
    if systems.empty:
        raise ValueError("Catalyst-screen manifest contains no expanded systems")
    state_ids = list(
        dict.fromkeys(
            str(entry["state_id"])
            for entry in manifest.get("reference_plan", [])
        )
    )
    if not state_ids:
        raise ValueError("Catalyst-screen manifest contains no reference plan")
    return plan_targets(systems, states=state_ids)


def _compatible_target(
    record: ReferenceRecord,
    targets: list[StructureTarget],
) -> StructureTarget:
    """Return the one target compatible with a record's scientific identity."""
    matches: list[StructureTarget] = []
    metadata = record.metadata
    for target in targets:
        if target.state_id != metadata.get("state_id"):
            continue
        try:
            _validate_reference_binding_target(metadata, target)
        except ValueError:
            continue
        matches.append(target)
    if len(matches) != 1:
        raise ValueError(
            f"Reference {record.reference_id!r} matched {len(matches)} current targets; "
            "expected exactly one"
        )
    return matches[0]


def _reference_aggregate_paths(reference_dir: Path) -> list[Path]:
    """Return terminal and nested aggregate files that may contain reuse rows."""
    paths = [reference_dir / name for name in ("reused.parquet", "merged.parquet")]
    tiers_dir = reference_dir / "tiers"
    if tiers_dir.exists():
        for tier_dir in sorted(path for path in tiers_dir.iterdir() if path.is_dir()):
            paths.extend(tier_dir / name for name in ("reused.parquet", "merged.parquet"))
    return [path for path in paths if path.exists()]


def _unique_bindings(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate JSON-compatible bindings while preserving order."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding in bindings:
        fingerprint = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(binding)
    return unique


def _same_value(left: Any, right: Any) -> bool:
    """Compare scalar dataframe labels while treating missing values as equal."""
    left_missing = bool(pd.isna(left))
    right_missing = bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def _backup_run_files(
    root: Path,
    repaired_frames: dict[Path, pd.DataFrame],
    destination: Path,
) -> None:
    """Back up raw aggregates and derived artifacts before applying a repair."""
    if destination.exists():
        raise FileExistsError(f"Repair backup already exists: {destination}")
    paths = [*repaired_frames]
    paths.extend(path for path in (root / "analysis").glob("*") if path.is_file())
    paths.extend(
        path
        for path in (root / "run_report.json", root / "manifest.json")
        if path.exists()
    )
    for path in paths:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically replace one parquet file in its existing directory."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace one readable JSON sidecar."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
