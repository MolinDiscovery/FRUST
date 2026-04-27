import re
from pathlib import Path
from typing import Mapping

import pandas as pd
from tooltoad.chemutils import ac2xyz
from frust.vis import MolTo3DGrid
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def dump_df(df: pd.DataFrame, step: str, base_dir: Path) -> Path:
    """
    If dump_each_step is True, writes DataFrame to `base_dir/{step}.csv`.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{step}.csv"
    df.to_csv(path, index=False)
    return path


def read_ts_type_from_xyz(xyz_file: str):
    """
    Reads the transition state (TS) type from the comment line of an XYZ file.

    Args:
        xyz_file (str): Path to the XYZ file containing the transition state structure.
            The TS type must be specified in the second line as 'TS' followed by a number
            (e.g., 'TS1 guess', 'TS2').

    Returns:
        str: The transition state type in uppercase (e.g., 'TS1', 'TS2').
    """    
    try:
        with open(xyz_file, 'r') as file:
            file.readline()  # Skip first line
            comment = file.readline()  # Read second line
            
            if not comment: 
                raise ValueError("XYZ file must have at least 2 lines with a comment on the second line")
                
    except FileNotFoundError:
        print(f"Error: Transition state structure file not found: {xyz_file}")
        raise
    except PermissionError:
        print(f"Error: Permission denied when accessing file: {xyz_file}")
        raise
    except IOError as e:
        print(f"Error: Failed to read transition state structure file {xyz_file}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading transition state structure from {xyz_file}: {e}")
        raise

    match = re.search(r'\b(?:TS|INT)\d+\b', comment, re.IGNORECASE)
    if match:
        return match.group().upper()
    else:
        raise ValueError(
            "XYZ file must specify a structure type in the comments. "
            "Please include TSX or INTX in the comment (e.g. TS1, INT3)."
        )


def write_xyz_structures(
    df: pd.DataFrame,
    path: Path | str,
    coord_options: Mapping[str, str],
    name_col: str = "custom_name",
    atoms_col: str = "atoms",
    show_mols: bool = False,
    **molto3d_kwargs,
) -> None:
    """Write XYZ structure files from coordinate columns in a dataframe.

    Args:
        df: Dataframe containing atoms, names, and coordinate columns.
        path: Base directory where the XYZ folders should be created.
        coord_options: Mapping from output folder/suffix to coordinate column.
            The key is used both as the subfolder name and filename suffix.
        name_col: Column containing the base structure name.
        atoms_col: Column containing atomic symbols.
        show_mols: Whether to display the written structures.
        **molto3d_kwargs: Additional keyword arguments passed to MolTo3DGrid.
    """
    path = Path(path)

    for option in coord_options:
        (path / option).mkdir(parents=True, exist_ok=True)

    mols = []
    legends = []

    for _, row in df.iterrows():
        name = row[name_col]
        atoms = row[atoms_col]

        for option, coord_col in coord_options.items():
            coords = row[coord_col]
            xyz_str = ac2xyz(atoms, coords)

            xyz_path = path / option / f"{name}_{option}.xyz"

            with open(xyz_path, "w") as f:
                f.write(xyz_str)

            if show_mols:
                mol = Chem.MolFromXYZBlock(xyz_str)
                if mol is not None:
                    rdDetermineBonds.DetermineConnectivity(mol)
                    mols.append(mol)
                    legends.append(f"{name}_{option}")

    if show_mols and mols:
        MolTo3DGrid(mols, legends=legends, **molto3d_kwargs)