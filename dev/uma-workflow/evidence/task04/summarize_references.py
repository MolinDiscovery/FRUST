"""Summarize saved UMA geometry-reference results for scientific review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frust.tsguess2.topologies import CORE_TOPOLOGIES


def summarize_result(path: Path) -> dict[str, object]:
    """Read one final reference result and summarize its geometry and modes.

    Parameters
    ----------
    path : pathlib.Path
        Single-row ``final.parquet`` copied from a submitted reference job.

    Returns
    -------
    dict
        State, source, stage termination, imaginary modes, and reactive-core
        distances and angles. Mode-distance changes are measured between
        ``-0.25`` and ``+0.25`` displacements along each imaginary mode.
    """
    frame = pd.read_parquet(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one reference row in {path}, found {len(frame)}")
    row = frame.iloc[0]
    state = str(row["state_id"]).upper()
    opt_stage = "uma_ts_opt" if state.startswith("TS") else "uma_opt"
    opt_coordinates = np.vstack(row[f"{opt_stage}-oc"]).astype(float)
    roles = {str(role): int(index) for role, index in row["constraint_roles"].items()}
    vibrations = row["uma_freq-vibs"]
    negative_modes = []
    for index, vibration in enumerate(vibrations):
        frequency = float(vibration["frequency"])
        if frequency >= 0:
            continue
        displacement = np.vstack(vibration["mode"]).astype(float)
        distance_changes = {}
        for constraint in CORE_TOPOLOGIES[state].constraints:
            if constraint.kind != "distance":
                continue
            first, second = (roles[role] for role in constraint.roles)
            before = np.linalg.norm(
                (opt_coordinates - 0.25 * displacement)[first]
                - (opt_coordinates - 0.25 * displacement)[second]
            )
            after = np.linalg.norm(
                (opt_coordinates + 0.25 * displacement)[first]
                - (opt_coordinates + 0.25 * displacement)[second]
            )
            distance_changes[constraint.name] = round(float(after - before), 4)
        negative_modes.append(
            {
                "index": index,
                "frequency_cm1": frequency,
                "distance_changes_angstrom": distance_changes,
            }
        )
    return {
        "state": state,
        "environment": "alpb-chloroform" if "alpb-chloroform" in path.stem else "gas",
        "source_xyz": str(row["source_xyz"]),
        "parquet": str(path.resolve()),
        "optimization_normal_termination": bool(row[f"{opt_stage}-NT"]),
        "frequency_normal_termination": bool(row["uma_freq-NT"]),
        "negative_modes": negative_modes,
        "lowest_frequency_cm1": float(vibrations[0]["frequency"]),
    }


def main() -> None:
    """Write a compact review JSON for all copied reference parquets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.parquet"))
    records = [summarize_result(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        modes = ", ".join(
            f"{mode['index']}: {mode['frequency_cm1']:.2f}"
            for mode in record["negative_modes"]
        ) or "none"
        print(
            f"{record['environment']:16} {record['state']:4} "
            f"opt={record['optimization_normal_termination']} "
            f"freq={record['frequency_normal_termination']} "
            f"negative={modes}"
        )


if __name__ == "__main__":
    main()
