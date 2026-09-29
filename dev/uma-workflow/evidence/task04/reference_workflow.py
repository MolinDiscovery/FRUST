"""Reoptimize the legacy ωB97 profile structures with UMA on a compute node."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frust.tsguess2.profiles.wb97xd3_631g_gas import GEOMETRIES
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from frust.workflows.methods import CalculatorSpec, MethodPlan


REPOSITORY = Path(__file__).resolve().parents[4]
SOURCE_FILES = {
    "TS1": "ts1.xyz",
    "TS2": "ts2.xyz",
    "TS3": "ts3_TMP.xyz",
    "TS4": "ts4_TMP.xyz",
    "INT3": "int3_TMP.xyz",
}
TS_MODES = {
    "TS1": ("transfer_H", "substrate_C"),
    "TS2": ("cat_B", "B_transfer_H"),
    "TS3": ("pin_B", "substrate_C"),
    "TS4": ("cat_B", "transfer_H"),
}


def read_reference_xyz(state: str) -> tuple[list[str], list[list[float]], dict[str, int]]:
    """Read a profile source and recover the exact reactive-role atom map.

    Parameters
    ----------
    state : str
        One of ``TS1``–``TS4`` or ``INT3``.

    Returns
    -------
    tuple
        Element symbols, Cartesian coordinates, and zero-based role indices.

    Raises
    ------
    ValueError
        If the XYZ is malformed or no longer matches the existing profile.
    """
    path = REPOSITORY / "structures" / SOURCE_FILES[state]
    lines = path.read_text().splitlines()
    atom_count = int(lines[0].strip())
    records = [line.split() for line in lines[2:] if line.strip()]
    if len(records) != atom_count or any(len(record) < 4 for record in records):
        raise ValueError(f"Malformed XYZ: {path}")
    atoms = [record[0] for record in records]
    coordinates = np.asarray(
        [[float(value) for value in record[1:4]] for record in records],
        dtype=float,
    )
    roles: dict[str, int] = {}
    for role, position in GEOMETRIES[state].role_coordinates.items():
        distances = np.linalg.norm(coordinates - np.asarray(position), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > 1e-5:
            raise ValueError(f"{state} {role} does not match {path}")
        roles[role] = index
    if len(set(roles.values())) != len(roles):
        raise ValueError(f"Duplicate role atom indices in {path}")
    return atoms, coordinates.tolist(), roles


class UMAReferenceWorkflow(BaseWorkflow):
    """Run one legacy profile structure through UMA optimization and NumFreq.

    Parameters
    ----------
    state : str
        ``TS1``–``TS4`` or ``INT3``.
    environment : {"gas", "alpb-chloroform"}
        UMA potential used throughout the job. ALPB adds the GFN2-xTB
        chloroform correction to UMA energies and gradients.
    output_root : pathlib.Path
        Directory in which the reusable UMA server writes its log.
    """

    workflow_name = "uma_geometry_reference"

    def __init__(self, *, state: str, environment: str, output_root: Path) -> None:
        if state not in SOURCE_FILES:
            raise ValueError(f"Unsupported state: {state}")
        if environment not in {"gas", "alpb-chloroform"}:
            raise ValueError(f"Unsupported environment: {environment}")
        self.state = state
        self.environment = environment
        self.output_root = Path(output_root)
        common = {
            "uma": "omol@uma-s-1p2p1",
            "uma_offline": True,
            "uma_inference_settings": "batch",
            "uma_keep_logs": "always",
            "uma_log_dir": str(self.output_root / "server-logs"),
        }
        if environment == "alpb-chloroform":
            import os

            common["uma_xtb_alpb"] = "chloroform"
            common["uma_xtb_exe"] = os.environ["XTB_EXE"]
        stage = "uma_opt" if state == "INT3" else "uma_ts_opt"
        opt_kwargs = dict(common)
        if state in TS_MODES:
            opt_kwargs["ts_mode"] = TS_MODES[state]
        method = MethodPlan(
            f"uma-s-1p2p1-{environment}-{state.lower()}",
            {
                stage: CalculatorSpec(
                    "orca",
                    {"ExtOpt": None, "Opt" if state == "INT3" else "OptTS": None},
                    kwargs=opt_kwargs,
                ),
                "uma_freq": CalculatorSpec(
                    "orca",
                    {"ExtOpt": None, "NumFreq": None},
                    kwargs=common,
                ),
            },
        )
        super().__init__(method=method)

    def _build_targets(self) -> list[WorkflowTarget]:
        """Return the selected state as a single scheduler target."""
        return [WorkflowTarget(f"{self.state.lower()}_{self.environment}", self.state)]

    def _prepare_initial_df(self, target, *, save_dir, options) -> pd.DataFrame:
        """Load the exact ωB97 profile source without generating new guesses."""
        atoms, coordinates, roles = read_reference_xyz(self.state)
        frame = pd.DataFrame(
            {
                "state_id": [self.state],
                "structure_type": [self.state],
                "substrate_name": [f"legacy-wb97-source-{self.state}"],
                "catalyst_name": [GEOMETRIES[self.state].reference.catalyst_name],
                "source_xyz": [SOURCE_FILES[self.state]],
                "atoms": [atoms],
                "coords_embedded": [coordinates],
                "constraint_roles": [roles],
            }
        )
        return frame

    def _stage_defs(self) -> list[StageDef]:
        """Keep UMA optimization and frequency in one server lifetime."""
        stage = "uma_opt" if self.state == "INT3" else "uma_ts_opt"
        files = ["input.inp", "orca.out", "input_EXT.uma.json", "input.xyz"]
        return [
            StageDef("prepare", "Load ωB97 reference XYZ", kind="prepare"),
            StageDef(stage, "UMA reference optimization", save_files=files),
            StageDef("uma_freq", "UMA numerical frequencies", save_files=files),
        ]
