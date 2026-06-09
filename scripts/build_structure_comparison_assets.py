"""Build structure-comparison documentation assets.

Run from the repository root:

    python scripts/build_structure_comparison_assets.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import frust as ft


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
HTML_PATH = ASSET_DIR / "structure-comparison-example.html"


def _embedded_ethanol() -> tuple[list[str], np.ndarray]:
    """Create a small reproducible structure for the docs viewer."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=23)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    return atoms, coords


def _write_xyz(path: Path, atoms: list[str], coords: np.ndarray) -> None:
    """Write a compact XYZ file."""
    lines = [str(len(atoms)), path.stem]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_assets() -> list[Path]:
    """Generate the structure-comparison HTML preview."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    atoms, ref_coords = _embedded_ethanol()
    probe_coords = ref_coords.copy()
    probe_coords[1] += np.array([0.25, -0.10, 0.05])
    probe_coords[2] += np.array([-0.08, 0.12, -0.18])

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        probe_path = tmpdir_path / "gxtb.xyz"
        ref_path = tmpdir_path / "orca.xyz"
        _write_xyz(probe_path, atoms, probe_coords)
        _write_xyz(ref_path, atoms, ref_coords)

        ft.vis.compare_xyz_rmsd(
            str(probe_path),
            str(ref_path),
            render=False,
            print_summary=False,
            top_n=3,
            cell_size=(420, 360),
            export_HTML=str(HTML_PATH),
        )

    return [HTML_PATH]


if __name__ == "__main__":
    for path in build_assets():
        print(path.relative_to(ROOT))
