"""Compact, geometry-bound native optimization evidence."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from frust.utils.RMSD import _coerce_coords_array


def _geometry_digest(atoms, coords):
    xyz = _coerce_coords_array(coords, "optimization coordinates")
    header = json.dumps(list(atoms), separators=(",", ":")).encode()
    return hashlib.sha256(header + np.asarray(xyz, dtype="<f8").tobytes()).hexdigest()


def _candidate_key(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _orca_optimization_requested(inputs):
    options = inputs.get("options") or {}
    keys = options if isinstance(options, dict) else str(options).split()
    return bool({str(k).lower() for k in keys} & {"opt", "optts", "copt", "optxyz", "optcart"}) and not re.search(
        r"\bscan\b", str(inputs.get("xtra_inp_str", "")), re.I
    )


def _optimization_record(row, stage, coords, log):
    text = re.sub(r"\s+", " ", log).upper()
    return {
        "stage": stage,
        "structure_id": str(row.get("structure_id", "")),
        "cid": _candidate_key(row.get("cid", "")),
        "geometry_sha256": _geometry_digest(row["atoms"], coords),
        "status": "converged" if "THE OPTIMIZATION HAS CONVERGED" in text else "not_converged",
        "engine": "orca",
        "output_sha256": hashlib.sha256(log.encode()).hexdigest(),
    }


def _optimization_check(records, row, stage):
    candidates = [r for r in records if r["stage"] == stage
                  and r["structure_id"] == str(row.get("structure_id", ""))
                  and r["cid"] == _candidate_key(row.get("cid", ""))]
    if not candidates:
        return None
    coords = row.get(f"{stage}-oc")
    try:
        digest = _geometry_digest(row["atoms"], coords)
    except (ValueError, TypeError):
        return {"status": "stale", "stage": stage}
    matches = [r for r in candidates if r["geometry_sha256"] == digest]
    if not matches:
        return {"status": "stale", "stage": stage}
    if len({r["status"] for r in matches}) > 1:
        return {"status": "conflicting", "stage": stage}
    return {"status": matches[0]["status"], "stage": stage,
            "geometry_sha256": digest,
            "output_sha256": sorted({r["output_sha256"] for r in matches})}
