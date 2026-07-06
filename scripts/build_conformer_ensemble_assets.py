"""Build documentation assets for conformer ensemble visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import frust as ft


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
HTML_PATH = ASSET_DIR / "conformer-ensemble-example.html"


def _rotated(coords: np.ndarray) -> np.ndarray:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return coords @ rotation.T + np.array([4.0, -2.0, 1.5])


def build_example_dataframe() -> pd.DataFrame:
    """Return a compact TS-like conformer dataframe for documentation."""
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [1.0, 1.0, 0.0],
            [1.4, 1.2, 1.1],
            [1.6, 1.0, 2.0],
        ],
        dtype=float,
    )
    rows = []
    for cid, energy in enumerate([0.0, 0.9, 1.8, 3.2, 5.6]):
        coords = base.copy()
        coords[3:] += np.array([0.0, 0.18 * cid, 0.22 * cid])
        if cid == 2:
            coords = _rotated(coords)
        rows.append(
            {
                "system_name": "anisole_cat_a",
                "substrate_name": "anisole",
                "catalyst_name": "cat_a",
                "structure_type": "TS1",
                "molecule_role": "ts",
                "rpos": 4,
                "cid": cid,
                "atoms": ["B", "N", "C", "C", "O", "H"],
                "connectivity_bonds": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                "coords_embedded": coords.tolist(),
                "constraint_roles": {"cat_B": 0, "cat_N": 1, "substrate_C": 2},
                "energy_uff": energy,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the conformer ensemble HTML asset."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = build_example_dataframe()
    ft.plot_conformers(
        df,
        mode="representatives+cloud",
        top_n=5,
        energy_window_kcal=4.0,
        export_HTML=str(HTML_PATH),
        background_color=("white", 1.0),
    )


if __name__ == "__main__":
    main()
