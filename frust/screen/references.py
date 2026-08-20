"""Inspectable, immutable reference calculations for catalyst screens."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from tooltoad.chemutils import ac2xyz

from frust.results import free_energy_components, result_column
from frust.schema import normal_termination_columns
from frust.structures import StructureTarget
from frust.workflows.methods import MethodPlan


ReviewDecision = Literal["approved", "rejected", "unreviewed"]
ReusePolicy = Literal["approved", "auto_valid"]
INDEX_COLUMNS = [
    "reference_id",
    "cache_key",
    "state_id",
    "compound_name",
    "formula",
    "charge",
    "multiplicity",
    "method",
    "method_fingerprint",
    "thermochemistry_mode",
    "free_energy_hartree",
    "created_at",
    "source_run",
    "entry_path",
]


@dataclass(frozen=True)
class ReferenceRecord:
    """One immutable scientific reference calculation.

    Parameters
    ----------
    library : ReferenceLibrary
        Library that owns the entry and its review sidecar.
    reference_id : str
        Stable content-derived reference identifier.
    path : pathlib.Path
        Entry directory containing metadata, dataframe, XYZ, and calculator
        files.
    """

    library: "ReferenceLibrary"
    reference_id: str
    path: Path

    @property
    def metadata(self) -> dict[str, Any]:
        """Return immutable entry metadata plus current review state."""
        payload = json.loads((self.path / "metadata.json").read_text())
        payload["review"] = self.library.review_status(self.reference_id)
        return payload

    def summary(self) -> pd.Series:
        """Return a compact human-readable entry summary."""
        metadata = self.metadata
        fields = [
            "reference_id",
            "state_id",
            "compound_name",
            "formula",
            "method",
            "thermochemistry_mode",
            "free_energy_hartree",
            "auto_validation",
            "review",
            "source_run",
        ]
        return pd.Series({field: metadata.get(field) for field in fields})

    def dataframe(self) -> pd.DataFrame:
        """Load the complete canonical FRUST result row."""
        return pd.read_parquet(self.path / "result.parquet")

    def xyz_path(self) -> Path:
        """Return the cached optimized XYZ path."""
        return self.path / "optimized.xyz"

    def write_xyz(self, path: str | Path, *, overwrite: bool = True) -> Path:
        """Copy the optimized XYZ structure to a user-selected path."""
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"XYZ file already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.xyz_path(), destination)
        return destination

    def files(self) -> list[Path]:
        """Return calculator input/output files retained by this entry."""
        root = self.path / "calculator_files"
        return sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []

    def view(self, **kwargs: Any) -> Any:
        """Display the optimized structure with FRUST's molecule viewer."""
        from frust.vis import plot_row

        return plot_row(self.dataframe(), 0, **kwargs)

    def plot_vibrations(self, *, mode: int = 0, **kwargs: Any) -> Any:
        """Display one cached normal mode with FRUST's vibration viewer."""
        from frust.vis import plot_vibs

        return plot_vibs(self.dataframe(), row_index=0, vId=mode, **kwargs)

    def approve(self, *, note: str = "", reviewer: str = "") -> None:
        """Approve this immutable entry for production reuse."""
        self.library.set_review(
            self.reference_id,
            "approved",
            note=note,
            reviewer=reviewer,
        )

    def reject(self, *, note: str = "", reviewer: str = "") -> None:
        """Reject this entry without deleting its audit record."""
        self.library.set_review(
            self.reference_id,
            "rejected",
            note=note,
            reviewer=reviewer,
        )


class ReferenceLibrary:
    """Searchable scientific library of immutable reference calculations.

    Parameters
    ----------
    root : str or pathlib.Path
        Library directory. Entries remain readable as ordinary JSON, parquet,
        XYZ, and calculator input/output files.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries_dir = self.root / "entries"
        self.index_path = self.root / "index.parquet"
        self.reviews_path = self.root / "reviews.csv"
        self.lock_path = self.root / ".library.lock"

    def initialize(self) -> "ReferenceLibrary":
        """Create the library directories and empty inspection files."""
        self.root.mkdir(parents=True, exist_ok=True)
        with _FileLock(self.lock_path):
            self.entries_dir.mkdir(parents=True, exist_ok=True)
            if not self.index_path.exists():
                _atomic_write_parquet(
                    pd.DataFrame(columns=INDEX_COLUMNS),
                    self.index_path,
                )
            if not self.reviews_path.exists():
                _atomic_write_csv(
                    pd.DataFrame(
                        columns=[
                            "reference_id",
                            "decision",
                            "note",
                            "reviewer",
                            "reviewed_at",
                        ]
                    ),
                    self.reviews_path,
                )
        return self

    def index(self) -> pd.DataFrame:
        """Return the searchable library index with current review states."""
        self.initialize()
        index = pd.read_parquet(self.index_path)
        reviews = self._reviews()
        if reviews.empty:
            index["review"] = "unreviewed"
            return index
        latest = reviews.drop_duplicates("reference_id", keep="last")
        index = index.merge(
            latest[["reference_id", "decision"]],
            on="reference_id",
            how="left",
        )
        index["review"] = index.pop("decision").fillna("unreviewed")
        return index

    def summary(self) -> pd.DataFrame:
        """Return counts grouped by method, state, and review decision."""
        index = self.index()
        if index.empty:
            return pd.DataFrame(columns=["method", "state_id", "review", "count"])
        return (
            index.groupby(["method", "state_id", "review"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )

    def search(
        self,
        *,
        state_id: str | None = None,
        compound_name: str | None = None,
        method: str | None = None,
        formula: str | None = None,
        review: ReviewDecision | None = None,
    ) -> pd.DataFrame:
        """Search entries by common scientific labels."""
        result = self.index()
        filters = {
            "state_id": state_id,
            "compound_name": compound_name,
            "method": method,
            "formula": formula,
            "review": review,
        }
        for column, value in filters.items():
            if value is not None:
                result = result[result[column].astype(str).str.casefold() == str(value).casefold()]
        return result.reset_index(drop=True)

    def get(self, reference_id: str) -> ReferenceRecord:
        """Return one entry by reference ID."""
        matches = self.index()
        matches = matches[matches["reference_id"].eq(str(reference_id))]
        if matches.empty:
            raise KeyError(f"Unknown reference ID {reference_id!r}")
        path = self.root / str(matches.iloc[-1]["entry_path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Reference entry directory is missing: {path}")
        self._validate_checksums(path)
        return ReferenceRecord(self, str(reference_id), path)

    def find(
        self,
        target: StructureTarget,
        method: MethodPlan,
        *,
        protocol: Mapping[str, Any] | None = None,
        reuse_policy: ReusePolicy = "approved",
    ) -> ReferenceRecord | None:
        """Return the newest compatible reusable entry, if one exists."""
        if reuse_policy not in {"approved", "auto_valid"}:
            raise ValueError("reuse_policy must be 'approved' or 'auto_valid'")
        cache_key, _ = reference_identity(target, method, protocol=protocol)
        candidates = self.index()
        candidates = candidates[candidates["cache_key"].eq(cache_key)].iloc[::-1]
        for _, candidate in candidates.iterrows():
            reference_id = str(candidate["reference_id"])
            decision = str(candidate["review"])
            if decision == "rejected":
                continue
            if reuse_policy == "approved" and decision != "approved":
                continue
            try:
                return self.get(reference_id)
            except (KeyError, FileNotFoundError, ValueError):
                continue
        return None

    def publish(
        self,
        df: pd.DataFrame,
        target: StructureTarget,
        method: MethodPlan,
        *,
        protocol: Mapping[str, Any] | None = None,
        source_run: str | Path | None = None,
        source_target_dir: str | Path | None = None,
    ) -> ReferenceRecord:
        """Publish one complete minimum result as an immutable entry."""
        self.initialize()
        if len(df) != 1:
            raise ValueError("reference publication requires exactly one selected result row")
        validation = _validate_reference_result(df, method)
        cache_key, identity = reference_identity(target, method, protocol=protocol)
        result_content = _reference_result_content(df, method)
        reference_id = "ref_" + _content_hash(
            {"identity": identity, "result": result_content}
        )[:16]
        compound_name = _compound_name(target)
        method_slug = _slug(method.name)
        state_slug = _slug(target.state_id)
        compound_slug = _slug(compound_name)
        entry_rel = Path("entries") / method_slug / state_slug / compound_slug / reference_id
        entry_path = self.root / entry_rel
        if entry_path.exists():
            self._validate_checksums(entry_path)
            metadata = json.loads((entry_path / "metadata.json").read_text())
            self._append_index(metadata, entry_rel)
            return ReferenceRecord(self, reference_id, entry_path)

        parent = entry_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mkdtemp(prefix=f".{reference_id}-", dir=parent))
        try:
            result_path = temp_path / "result.parquet"
            df.to_parquet(result_path, index=False)
            coords_col = result_column(df, "coords", purpose="optimized")
            row = df.iloc[0]
            (temp_path / "optimized.xyz").write_text(ac2xyz(row["atoms"], row[coords_col]))
            _copy_scientific_calculator_files(source_target_dir, temp_path / "calculator_files")
            energy = free_energy_components(
                df,
                thermochemistry=method.thermochemistry,
            ).iloc[0]
            metadata = {
                "schema_version": 1,
                "reference_id": reference_id,
                "cache_key": cache_key,
                "result_fingerprint": _content_hash(result_content),
                "state_id": target.state_id,
                "state_kind": target.state_kind,
                "compound_name": compound_name,
                "formula": _formula(row["atoms"]),
                "charge": int(row.get("charge", 0) or 0),
                "multiplicity": int(row.get("multiplicity", 1) or 1),
                "method": method.name,
                "method_fingerprint": method.fingerprint(),
                "method_plan": method.to_dict(),
                "thermochemistry_mode": str(energy["thermochemistry_mode"]),
                "free_energy_hartree": float(energy["free_energy_hartree"]),
                "auto_validation": validation,
                "identity": identity,
                "source_run": None if source_run is None else str(source_run),
                "created_at": _utc_now(),
            }
            metadata["checksums"] = _entry_checksums(temp_path)
            metadata_path = temp_path / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
            )
            (temp_path / "metadata.sha256").write_text(
                _file_sha256(metadata_path) + "  metadata.json\n"
            )
            try:
                os.replace(temp_path, entry_path)
            except OSError:
                if not entry_path.exists():
                    raise
                shutil.rmtree(temp_path, ignore_errors=True)
            self._append_index(metadata, entry_rel)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
        return ReferenceRecord(self, reference_id, entry_path)

    def import_record(self, record: ReferenceRecord) -> ReferenceRecord:
        """Copy an immutable entry and its review into this library.

        Parameters
        ----------
        record : ReferenceRecord
            Entry from another library, normally a shared cluster reference
            library being snapshotted into a portable run.

        Returns
        -------
        ReferenceRecord
            Equivalent entry owned by this library.
        """
        self.initialize()
        metadata = record.metadata
        source_metadata = dict(metadata)
        source_metadata.pop("review", None)
        source_path = record.path
        method_slug = _slug(str(source_metadata["method"]))
        state_slug = _slug(str(source_metadata["state_id"]))
        compound_slug = _slug(str(source_metadata["compound_name"]))
        entry_rel = (
            Path("entries")
            / method_slug
            / state_slug
            / compound_slug
            / str(source_metadata["reference_id"])
        )
        destination = self.root / entry_rel
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{record.reference_id}-", dir=destination.parent)
            )
            shutil.rmtree(temporary)
            shutil.copytree(source_path, temporary)
            try:
                os.replace(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                shutil.rmtree(temporary, ignore_errors=True)
        self._append_index(source_metadata, entry_rel)
        decision = record.library.review_status(record.reference_id)
        if (
            decision in {"approved", "rejected"}
            and self.review_status(record.reference_id) != decision
        ):
            self.set_review(record.reference_id, decision)
        return ReferenceRecord(self, record.reference_id, destination)

    def review_status(self, reference_id: str) -> ReviewDecision:
        """Return the latest review decision for an entry."""
        reviews = self._reviews()
        matches = reviews[reviews["reference_id"].eq(str(reference_id))]
        if matches.empty:
            return "unreviewed"
        return str(matches.iloc[-1]["decision"])  # type: ignore[return-value]

    def review_queue(self) -> pd.DataFrame:
        """Return auto-valid entries that have not been manually reviewed."""
        return self.index().query("review == 'unreviewed'").reset_index(drop=True)

    def set_review(
        self,
        reference_id: str,
        decision: Literal["approved", "rejected"],
        *,
        note: str = "",
        reviewer: str = "",
    ) -> None:
        """Append a review decision without modifying the immutable entry."""
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        self.get(reference_id)
        row = pd.DataFrame(
            [{
                "reference_id": reference_id,
                "decision": decision,
                "note": str(note),
                "reviewer": str(reviewer),
                "reviewed_at": _utc_now(),
            }]
        )
        with self._locked():
            reviews = self._reviews()
            _atomic_write_csv(
                pd.concat([reviews, row], ignore_index=True),
                self.reviews_path,
            )

    def _append_index(self, metadata: Mapping[str, Any], entry_rel: Path) -> None:
        row = {
            column: metadata.get(column)
            for column in INDEX_COLUMNS
            if column != "entry_path"
        }
        row["entry_path"] = str(entry_rel)
        with self._locked():
            index = pd.read_parquet(self.index_path)
            if str(metadata["reference_id"]) in set(index.get("reference_id", [])):
                return
            updated = (
                pd.DataFrame([row], columns=INDEX_COLUMNS)
                if index.empty
                else pd.concat([index, pd.DataFrame([row])], ignore_index=True)
            )
            _atomic_write_parquet(updated, self.index_path)

    def _reviews(self) -> pd.DataFrame:
        if not self.reviews_path.exists():
            self.initialize()
        return pd.read_csv(self.reviews_path, dtype=str).fillna("")

    def _locked(self):
        """Return an exclusive file-lock context manager."""
        self.root.mkdir(parents=True, exist_ok=True)
        return _FileLock(self.lock_path)

    @staticmethod
    def _validate_checksums(entry_path: Path) -> None:
        metadata_path = entry_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Reference metadata is missing: {metadata_path}")
        metadata_digest_path = entry_path / "metadata.sha256"
        if not metadata_digest_path.exists():
            raise ValueError(f"Reference metadata checksum is missing: {metadata_digest_path}")
        digest_fields = metadata_digest_path.read_text().split()
        if not digest_fields:
            raise ValueError(f"Reference metadata checksum is empty: {metadata_digest_path}")
        expected_metadata_digest = digest_fields[0]
        if _file_sha256(metadata_path) != expected_metadata_digest:
            raise ValueError(f"Reference checksum failed for {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        for relative, expected in metadata.get("checksums", {}).items():
            path = entry_path / relative
            if not path.exists() or _file_sha256(path) != expected:
                raise ValueError(f"Reference checksum failed for {path}")


class _FileLock:
    """Small Unix file-lock context manager for library sidecars."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_FileLock":
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def open_reference_library(path: str | Path) -> ReferenceLibrary:
    """Open or initialize an inspectable reference library."""
    return ReferenceLibrary(path).initialize()


def reference_identity(
    target: StructureTarget,
    method: MethodPlan,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a stable compatibility key and its scientific identity."""
    if method.thermochemistry is None:
        raise ValueError(
            f"Method plan {method.name!r} has no thermochemistry specification"
        )
    identity = {
        "schema_version": 1,
        "state_id": target.state_id,
        "state_kind": target.state_kind,
        "scope": target.scope,
        "rpos": target.rpos,
        "builder_spec": target.builder_spec,
        "chemical_identity": _target_chemical_identity(target),
        "charge": int(target.builder_options.get("charge", 0)),
        "multiplicity": int(target.builder_options.get("multiplicity", 1)),
        "method_fingerprint": method.fingerprint(),
        "thermochemistry": method.thermochemistry.to_dict(),
        "protocol": _json_compatible(dict(protocol or {})),
    }
    return f"key_{_content_hash(identity)[:16]}", identity


def _target_chemical_identity(target: StructureTarget) -> dict[str, Any]:
    system = target.system
    scope = target.scope
    identity: dict[str, Any] = {}
    if scope in {"substrate", "substrate_rpos", "system", "system_rpos"}:
        identity["substrate_smiles"] = _canonical_smiles(system.substrate_smiles)
    if scope in {"catalyst", "system", "system_rpos"}:
        identity["catalyst_smiles"] = _canonical_smiles(system.catalyst_smiles)
    if scope == "global":
        identity["global_state"] = target.state_id
    return identity


def _canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Cannot canonicalize SMILES {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _validate_reference_result(
    df: pd.DataFrame,
    method: MethodPlan,
) -> dict[str, Any]:
    nt_columns = normal_termination_columns(df)
    if not nt_columns:
        raise ValueError("Reference has no normal-termination provenance")
    failed_nt = [column for column in nt_columns if not bool(df[column].fillna(False).all())]
    if failed_nt:
        raise ValueError(f"Reference has non-normal termination columns: {failed_nt}")
    components = free_energy_components(
        df,
        thermochemistry=method.thermochemistry,
    )
    if components["free_energy_hartree"].isna().any():
        raise ValueError("Reference has missing assembled free energy")
    vibration_columns = [column for column in df.columns if str(column).endswith("-vibs")]
    if not vibration_columns:
        raise ValueError("Reference has no vibration data")
    vibrations = next(
        (
            df[column].iloc[0]
            for column in reversed(vibration_columns)
            if _usable_vibrations(df[column].iloc[0])
        ),
        None,
    )
    if vibrations is None:
        raise ValueError("Reference has no usable vibration data")
    negative = [float(mode["frequency"]) for mode in vibrations if float(mode["frequency"]) < 0]
    if negative:
        raise ValueError(f"Reference minimum has {len(negative)} imaginary frequencies")
    return {
        "status": "auto_valid",
        "normal_termination_columns": nt_columns,
        "n_imag": 0,
    }


def _reference_result_content(
    df: pd.DataFrame,
    method: MethodPlan,
) -> dict[str, Any]:
    """Return the scientific result content used for immutable entry IDs."""
    row = df.iloc[0]
    coords_column = result_column(df, "coords", purpose="optimized")
    vibration_columns = [column for column in df.columns if str(column).endswith("-vibs")]
    vibration_column = next(
        (
            column
            for column in reversed(vibration_columns)
            if _usable_vibrations(row[column])
        ),
        None,
    )
    energies = free_energy_components(
        df,
        thermochemistry=method.thermochemistry,
    ).iloc[0]
    return _json_compatible(
        {
            "atoms": row["atoms"],
            "optimized_coords": row[coords_column],
            "energies": energies.to_dict(),
            "vibrations": None if vibration_column is None else row[vibration_column],
            "normal_termination": {
                column: row[column]
                for column in normal_termination_columns(df)
            },
        }
    )


def _usable_vibrations(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0


def _copy_scientific_calculator_files(
    source_target_dir: str | Path | None,
    destination: Path,
) -> None:
    if source_target_dir is None:
        return
    source = Path(source_target_dir)
    if not source.is_dir():
        return
    stage_names = {"dft_opt", "dft_freq", "dft_solv_sp"}
    candidates = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name.lower() in {"orca.out", "orca.inp", "input.inp", "output.out"}
        and any(stage in path.parts for stage in stage_names)
    ]
    for path in candidates:
        stage = next(stage for stage in stage_names if stage in path.parts)
        target = destination / stage / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{path.parent.name}_{path.name}")
        shutil.copy2(path, target)


def _entry_checksums(entry_path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(entry_path.rglob("*")):
        if path.is_file() and path.name != "metadata.json":
            checksums[str(path.relative_to(entry_path))] = _file_sha256(path)
    return checksums


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically replace a parquet sidecar after writing it completely."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomically replace a CSV sidecar after writing it completely."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compound_name(target: StructureTarget) -> str:
    if target.scope == "global":
        return {"HH": "H2", "HBpin-mol": "HBpin"}.get(target.state_id, target.state_id)
    if target.scope.startswith("substrate"):
        return target.system.substrate_name
    if target.scope == "catalyst":
        return target.system.catalyst_name
    return target.system.system_name


def _formula(atoms: Any) -> str:
    counts = Counter(str(atom) for atom in atoms)
    order = ["C", "H"] + sorted(element for element in counts if element not in {"C", "H"})
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in order
        if counts.get(element)
    )


def _slug(value: str) -> str:
    text = "".join(character if character.isalnum() else "_" for character in str(value))
    return text.strip("_").lower() or "reference"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_compatible(value: Any) -> Any:
    if value is pd.NA:
        return None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"Reference value {value!r} is not JSON-compatible")
