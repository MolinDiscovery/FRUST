"""Extract reviewable tsguess2 geometry candidates from optimized results."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import acos, degrees
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frust.tsguess2.models import GeometryKey
from frust.tsguess2.topologies import CORE_TOPOLOGIES


def main() -> None:
    """Extract deterministic geometry candidates and validation metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquets", nargs="+", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--basis")
    parser.add_argument("--solvation-model")
    parser.add_argument("--solvent")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--mode-reviewed", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    key = GeometryKey(
        method=args.method,
        basis=args.basis,
        solvation_model=args.solvation_model,
        solvent=args.solvent,
    )
    excluded = {state.upper() for state in args.exclude}
    reviewed = {state.upper() for state in args.mode_reviewed}

    rows: list[tuple[pd.Series, Path]] = []
    for path in args.parquets:
        frame = pd.read_parquet(path)
        rows.extend((row, path) for _, row in frame.iterrows())

    candidates: dict[str, dict[str, Any]] = {}
    for row, source in rows:
        state = _state(row)
        if state in excluded:
            continue
        if state in candidates:
            raise ValueError(f"Multiple source rows found for {state}")
        topology = CORE_TOPOLOGIES[state]
        coordinates_column = "dft_ts_opt-oc" if state.startswith("TS") else "dft_opt-oc"
        vibrations_column = "dft_freq-vibs"
        coords = np.vstack(row[coordinates_column]).astype(float)
        roles = {
            str(role): int(index)
            for role, index in row["constraint_roles"].items()
            if index is not None and not pd.isna(index)
        }
        required_roles = sorted(
            {role for constraint in topology.constraints for role in constraint.roles}
        )
        missing_roles = sorted(set(required_roles) - set(roles))
        if missing_roles:
            raise ValueError(f"{state} is missing roles: {missing_roles}")

        negative_frequencies = tuple(
            float(mode["frequency"])
            for mode in row[vibrations_column]
            if float(mode["frequency"]) < 0
        )
        expected_negative = 1 if state.startswith("TS") else 0
        frequency_valid = len(negative_frequencies) == expected_negative
        mode_reviewed = state in reviewed if state.startswith("TS") else True

        candidates[state] = {
            "status": "candidate" if frequency_valid and mode_reviewed else "quarantined",
            "geometry_key": {
                "method": key.method,
                "basis": key.basis,
                "solvation_model": key.solvation_model,
                "solvent": key.solvent,
            },
            "role_coordinates": {
                role: [float(value) for value in coords[roles[role]]]
                for role in required_roles
            },
            "constraint_values": {
                constraint.name: _internal_coordinate(
                    coords,
                    [roles[role] for role in constraint.roles],
                )
                for constraint in topology.constraints
            },
            "reference": {
                "substrate_name": _text(row.get("substrate_name")),
                "catalyst_name": _text(row.get("catalyst_name")),
                "coordinates_column": coordinates_column,
                "vibrations_column": vibrations_column,
                "negative_frequencies": list(negative_frequencies),
                "frequency_count_valid": frequency_valid,
                "mode_reviewed": mode_reviewed,
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
            },
        }

    payload = {
        "schema_version": 1,
        "profile": key.profile_id,
        "candidates": dict(sorted(candidates.items())),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


def _state(row: pd.Series) -> str:
    """Return a supported state identifier from one source row."""
    for column in ("state_id", "structure_type"):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            state = str(value).upper()
            if state in CORE_TOPOLOGIES:
                return state
    raise ValueError("Could not infer a supported state from source row")


def _internal_coordinate(coords: np.ndarray, atom_indices: list[int]) -> float:
    """Return a distance or angle for two or three atom indices."""
    if len(atom_indices) == 2:
        return float(np.linalg.norm(coords[atom_indices[0]] - coords[atom_indices[1]]))
    first, center, last = (coords[index] for index in atom_indices)
    left = first - center
    right = last - center
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return float(degrees(acos(np.clip(cosine, -1.0, 1.0))))


def _sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a source artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str | None:
    """Return optional scalar metadata as text."""
    if value is None or pd.isna(value):
        return None
    return str(value)


if __name__ == "__main__":
    main()
