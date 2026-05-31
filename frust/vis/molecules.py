from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from tooltoad.chemutils import ac2mol
from tooltoad.vis import DrawMolSvg, MolTo3DGrid, RxnTo3DGrid
from tooltoad.scene3d import Py3DmolGridRenderer

from frust.schema import normalize_dataframe
from frust.vis.scenes import molecule_scene_from_dataframe


def plot_mols(
    df: pd.DataFrame,
    row_indices: Optional[List[int]] = None,
    substrate_filter: Optional[List[str]] = None,
    rpos_filter: Optional[List[Union[str, int]]] = None,
    exclude_coords: Optional[List[str]] = None,
    include_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    dark: bool = False,
    **molto3d_kwargs: Any
) -> None:
    """
    Display molecules from a dataframe with filtering capabilities.

    Args:
        df: DataFrame with molecular data
        row_indices: List of row indices to display (if None, displays all
            rows)
        substrate_filter: List of substrate names to include (if None, includes all)
        rpos_filter: List of rpos values to include (if None, includes all)
        exclude_coords: List of coordinate column patterns to exclude
        include_coords: List of coordinate column patterns to include
            (overrides exclude)
        coord_indices: List of indices or a slice for coordinate columns
            (overrides include/exclude).
        dark: If True, use a dark background by default. Ignored if
            'background_color' is explicitly provided in molto3d_kwargs.
        **molto3d_kwargs: Additional arguments to pass to MolTo3DGrid

    Returns:
        None
    """
    scene_kwargs = {
        "cell_size": molto3d_kwargs.pop("cell_size", (400, 400)),
        "columns": molto3d_kwargs.pop("columns", None),
        "linked": molto3d_kwargs.pop("linked", False),
        "show_labels": molto3d_kwargs.pop("show_labels", False),
        "show_charges": molto3d_kwargs.pop("show_charges", True),
        "kekulize": molto3d_kwargs.pop("kekulize", True),
    }
    export_HTML = molto3d_kwargs.pop("export_HTML", "none")
    if "background_color" in molto3d_kwargs:
        scene_kwargs["background_color"] = molto3d_kwargs.pop("background_color")
    elif dark:
        scene_kwargs["background_color"] = ("black", 1.0)

    if molto3d_kwargs:
        ignored = ", ".join(sorted(molto3d_kwargs))
        print(f"Ignoring unsupported scene-grid options: {ignored}")

    try:
        scene = molecule_scene_from_dataframe(
            df,
            row_indices=row_indices,
            substrate_filter=substrate_filter,
            rpos_filter=rpos_filter,
            exclude_coords=exclude_coords,
            include_coords=include_coords,
            coord_indices=coord_indices,
            **scene_kwargs,
        )
    except ValueError as exc:
        print(str(exc))
        return None

    print(f"Generated {len(scene.cells)} molecules for display")
    renderer = Py3DmolGridRenderer(scene)
    renderer.show()
    if export_HTML != "none":
        try:
            renderer.write_html(export_HTML)
            print(f"HTML export successful: {export_HTML}")
        except Exception as e:
            print(f"Error exporting HTML to '{export_HTML}': {e}")
    return None


def _row_to_mol(row: pd.Series, atoms: list[str], coords: list[tuple[float, float, float]]):
    """Create a molecule for plotting one dataframe row.

    Parameters
    ----------
    row : pandas.Series
        Dataframe row.
    atoms : list of str
        Atom symbols.
    coords : list of tuple of float
        Cartesian coordinates.

    Returns
    -------
    rdkit.Chem.Mol or None
        Molecule with row-provided connectivity when available.
    """
    bonds = row.get("connectivity_bonds")
    if isinstance(bonds, list) and bonds:
        return _mol_from_connectivity(atoms, coords, bonds)
    return ac2mol(atoms, coords)


def _mol_from_connectivity(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    bonds: list[tuple[int, int]],
) -> Chem.Mol:
    """Build an RDKit molecule from explicit atom and bond connectivity.

    Parameters
    ----------
    atoms : list of str
        Atom symbols.
    coords : list of tuple of float
        Cartesian coordinates.
    bonds : list of tuple of int
        Zero-based atom-index pairs.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with one conformer.
    """
    editable = Chem.RWMol()
    for symbol in atoms:
        editable.AddAtom(Chem.Atom(str(symbol)))
    for begin, end in bonds:
        if begin == end:
            continue
        if editable.GetBondBetweenAtoms(int(begin), int(end)) is None:
            editable.AddBond(int(begin), int(end), Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    conformer = Chem.Conformer(len(atoms))
    for atom_idx, coord in enumerate(coords):
        conformer.SetAtomPosition(
            int(atom_idx),
            Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
        )
    mol.AddConformer(conformer, assignId=True)
    return mol

def plot_row(
    df: pd.DataFrame,
    row_index: int = 0,
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display all coordinate types for a single row."""
    plot_mols(
        df,
        row_indices=[row_index],
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )

def plot_lig(
    df: pd.DataFrame,
    substrate_names: Union[str, List[str]],
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display molecules filtered by substrate name(s)."""
    if isinstance(substrate_names, str):
        substrate_names = [substrate_names]

    plot_mols(
        df,
        substrate_filter=substrate_names,
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )

def plot_rpos(
    df: pd.DataFrame,
    rpos_values: Union[str, int, List[Union[str, int]]],
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display molecules filtered by rpos value(s)."""
    if isinstance(rpos_values, (str, int)):
        rpos_values = [rpos_values]

    plot_mols(
        df,
        rpos_filter=rpos_values,
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )
