"""Build XYZ export tutorial assets.

Run from the repository root:

    python scripts/build_xyz_export_assets.py
"""

from pathlib import Path

from rdkit import Chem

from frust.embedder import embed_mols
from frust.stepper import Stepper
from frust.utils.io import write_xyz
from frust.vis import MolTo3DGrid


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
XYZ_DIR = ASSET_DIR / "xyz-export"
HTML_PATH = ASSET_DIR / "xyz-export-structures.html"

MOLECULES = {
    "acetanilide": "CC(=O)Nc1ccccc1",
    "benzamide": "NC(=O)c1ccccc1",
}


def _build_input_molecules() -> dict[str, tuple[Chem.Mol, dict[str, str]]]:
    """Create tutorial molecules with FRUST dataframe metadata."""
    mols = {}
    for name, smiles in MOLECULES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse tutorial SMILES for {name}: {smiles}")
        mols[name] = (
            mol,
            {
                "custom_name": name,
                "substrate_name": name,
                "structure_type": "MOL",
                "molecule_role": "product(anin)",
                "smiles": smiles,
                "input_smiles": smiles,
            },
        )
    return mols


def build_assets() -> list[Path]:
    """Generate tutorial XYZ files and the matching 3D HTML preview."""
    embedded = embed_mols(
        _build_input_molecules(),
        n_confs=1,
        n_cores=1,
        optimization="MMFF94",
    )
    df_assets = Stepper(save_output_dir=False, n_cores=1).build_initial_df(embedded)
    df_assets["moltype"] = "product(anin)"
    df_assets["xtb_opt-oc"] = df_assets["coords_embedded"]
    df_assets["orca_opt-oc"] = df_assets["coords_embedded"]
    df_assets = df_assets.sort_values("substrate_name").reset_index(drop=True)

    paths = write_xyz(
        df_assets,
        XYZ_DIR,
        coords_col="orca_opt-oc",
        overwrite=True,
    )

    legend_order = [path.stem for path in paths]
    MolTo3DGrid(
        paths,
        legends=legend_order,
        columns=2,
        cell_size=(320, 300),
        linked=False,
        export_HTML=str(HTML_PATH),
        show_charges=False,
    )
    return paths + [HTML_PATH]


if __name__ == "__main__":
    for path in build_assets():
        print(path.relative_to(ROOT))
