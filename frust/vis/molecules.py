import os
from collections.abc import Sequence
from numbers import Integral
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Geometry import Point3D
from tooltoad.chemutils import ac2mol
from tooltoad.vis import DrawMolSvg, RxnTo3DGrid
from tooltoad.scene3d import (
    AtomHighlight,
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    SceneCell,
    coerce_to_mol,
    ensure_3d_mol,
)

from frust.schema import normalize_dataframe
from frust.vis.scenes import molecule_scene_from_dataframe


def MolTo3DGrid(
    mols: Chem.Mol | str | os.PathLike | list[Chem.Mol | str | os.PathLike],
    show_labels: bool = False,
    show_confs: bool = True,
    background_color: tuple[str, float] = ("blue", 0.1),
    export_HTML: str = "none",
    cell_size: tuple[int, int] = (400, 400),
    columns: int = 3,
    linked: bool = False,
    kekulize: bool = True,
    legends: list[str] | None = None,
    highlightAtoms: list[int] | list[list[int]] | None = None,
    bonds_to_remove: list[tuple[int, int]] | None = None,
    show_charges: bool = True,
    verbose: bool = False,
    decimals_of_measure: int = 3,
) -> None:
    """Display one or more molecules in an interactive 3D grid.

    Examples
    --------
    Measure distances and angles with one decimal place in the interactive
    labels::

        import frust as ft

        ft.MolTo3DGrid("structures/ts1.xyz", decimals_of_measure=1)

    Show two molecules with synchronized rotation::

        ft.MolTo3DGrid(["C1=CC=CO1", "CN1C=CC=C1"], linked=True)

    Parameters
    ----------
    mols
        A molecule, SMILES string, ``.xyz`` path, or a list of these.
    show_labels
        If ``True``, draw atom labels before rendering. Labels use
        ``atomNote`` when present, otherwise atom indices.
    show_confs
        If ``True``, show every conformer of each molecule. If ``False``, show
        only conformer ``0``.
    background_color
        Viewer background as ``(color, opacity)``, for example
        ``("blue", 0.1)`` or ``("white", 1.0)``.
    export_HTML
        Output path for HTML export. Use ``"none"`` to disable export.
    cell_size
        Width and height of each grid cell in pixels.
    columns
        Number of grid columns. For one or two single-conformer molecules,
        FRUST chooses a compact one- or two-column layout.
    linked
        If ``True``, link rotation and zoom across all cells.
    kekulize
        Whether to kekulize RDKit mol blocks before display.
    legends
        Legend text for each input molecule. If omitted, default labels are
        used. Conformer numbers are appended automatically when needed.
    highlightAtoms
        Atom indices to mark with translucent cyan halo overlays. Provide one
        list per molecule, or a single list for a single molecule.
    bonds_to_remove
        Bonds to remove before display, given as atom-index pairs. These are
        removed only on temporary visualization copies.
    show_charges
        If ``True``, display non-zero formal charges in 3D.
    verbose
        If ``True``, allow RDKit embedding messages to surface.
    decimals_of_measure
        Number of decimals in Ctrl-click distance labels and Shift-click angle
        labels. The default is ``3``.

    Returns
    -------
    None
        The viewer is displayed directly in notebook contexts.
    """

    decimals_of_measure = _validate_decimals_of_measure(decimals_of_measure)

    if not isinstance(mols, list):
        mols = [mols]
    mols = [ensure_3d_mol(coerce_to_mol(mol), verbose=verbose) for mol in mols]

    if legends is None or not legends:
        legends = [f"Mol {idx + 1}" for idx in range(len(mols))]
    if len(legends) != len(mols):
        raise ValueError("Length of legends must match the number of molecules.")

    normalized_highlights = _normalize_scene_highlights(highlightAtoms, len(mols))

    cells = []
    mols_with_multiple_confs = any(mol.GetNumConformers() > 1 for mol in mols)
    for mol_idx, mol in enumerate(mols):
        conf_ids = list(range(mol.GetNumConformers())) if show_confs else [0]
        for conf_id in conf_ids:
            title = legends[mol_idx]
            if len(conf_ids) > 1:
                title += f" c{conf_id + 1}"

            overlays = [
                AtomHighlight(atom=atom_idx)
                for atom_idx in (normalized_highlights[mol_idx] or [])
            ]
            cells.append(
                SceneCell(
                    title=title,
                    models=[
                        MoleculeModel(
                            mol=mol,
                            conf_id=conf_id,
                            kekulize=kekulize,
                            show_atom_labels=show_labels,
                            show_charges=show_charges,
                            bonds_to_remove=bonds_to_remove,
                        )
                    ],
                    overlays=overlays,
                )
            )

    if len(mols) == 1 and not mols_with_multiple_confs:
        columns = 1
    elif len(mols) == 2 and not mols_with_multiple_confs:
        columns = 2

    scene = GridScene(
        cells=cells,
        columns=columns,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
    )
    renderer = Py3DmolGridRenderer(scene)
    renderer._CLICK_HANDLER = _click_handler_with_measure_decimals(decimals_of_measure)
    renderer.show()

    if export_HTML != "none":
        try:
            renderer.write_html(export_HTML)
            print(f"HTML export successful: {export_HTML}")
        except Exception as e:
            print(f"Error exporting HTML to '{export_HTML}': {e}")
    return None


def _validate_decimals_of_measure(decimals_of_measure: int) -> int:
    """Return a validated number of measurement-label decimals."""
    if isinstance(decimals_of_measure, bool) or not isinstance(decimals_of_measure, Integral):
        raise TypeError("decimals_of_measure must be a non-negative integer.")
    decimals = int(decimals_of_measure)
    if decimals < 0:
        raise ValueError("decimals_of_measure must be a non-negative integer.")
    if decimals > 100:
        raise ValueError("decimals_of_measure cannot be greater than 100.")
    return decimals


def _click_handler_with_measure_decimals(decimals_of_measure: int) -> str:
    """Return the py3Dmol click handler with measurement precision set."""
    angstrom = chr(0x00C5)
    degree = chr(0x00B0)
    replacements = {
        f".toFixed(3) + ' {angstrom}'": (
            f".toFixed({decimals_of_measure}) + ' {angstrom}'"
        ),
        f".toFixed(2) + '{degree}'": (
            f".toFixed({decimals_of_measure}) + '{degree}'"
        ),
    }
    handler = Py3DmolGridRenderer._CLICK_HANDLER
    for pattern, replacement in replacements.items():
        if pattern not in handler:
            raise RuntimeError("Could not locate measurement formatting hook.")
        handler = handler.replace(pattern, replacement, 1)
    return handler


def _normalize_scene_highlights(
    highlight_atoms: list[int] | list[list[int]] | None,
    n_mols: int,
) -> list[list[int] | None]:
    """Normalize atom-highlight inputs to one optional list per molecule."""
    if highlight_atoms is None:
        return [None] * n_mols
    if all(isinstance(atom_idx, int) for atom_idx in highlight_atoms):
        if n_mols != 1:
            raise ValueError(
                "For multiple molecules, highlightAtoms must contain one "
                "sequence per molecule."
            )
        return [list(highlight_atoms)]
    try:
        normalized = [list(seq) if seq is not None else None for seq in highlight_atoms]
    except TypeError as exc:
        raise ValueError(
            "highlightAtoms must be a sequence of ints or a sequence of sequences."
        ) from exc
    if len(normalized) != n_mols:
        raise ValueError("Length of highlightAtoms must match number of molecules.")
    return normalized


def plot_mols(
    df: pd.DataFrame,
    row_indices: Optional[List[int]] = None,
    substrate_filter: Optional[List[str]] = None,
    rpos_filter: Optional[List[Union[str, int]]] = None,
    exclude_coords: Optional[List[str]] = None,
    include_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    dark: bool = False,
    *,
    legends: Sequence[str] | None = None,
    cell_size: tuple[int, int] = (400, 400),
    columns: int | None = None,
    linked: bool = False,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    background_color: str | tuple[str, float] | None = None,
    export_HTML: str = "none",
) -> None:
    """Display molecules from a dataframe in an interactive 3D grid.

    Examples
    --------
    Label the last available coordinates for two selected rows::

        import frust as ft

        ft.plot_mols(
            df,
            row_indices=[0, 1],
            legends=["Reactant", "Product"],
            columns=2,
        )

    Parameters
    ----------
    df
        Dataframe containing ``atoms`` and one or more coordinate columns.
    row_indices
        Positional dataframe rows to display. When omitted, display all rows.
    substrate_filter
        Substrate names to include. When omitted, include every substrate.
    rpos_filter
        Reactive-position values to include. When omitted, include every
        reactive position.
    exclude_coords
        Coordinate-column name fragments to exclude when ``coord_indices`` and
        ``include_coords`` are both ``None``.
    include_coords
        Coordinate-column name fragments to include when ``coord_indices`` is
        ``None``.
    coord_indices
        Positional coordinate-column selection. The default selects only the
        last coordinate column. Set this to ``None`` to use ``include_coords``
        or ``exclude_coords``.
    dark
        If ``True``, use an opaque black background unless
        ``background_color`` is provided.
    legends
        Custom title for each valid rendered cell. Titles follow dataframe-row
        order and then coordinate-column order. The number of titles must
        equal the number of rendered cells after missing coordinates are
        skipped.
    cell_size
        Width and height of each grid cell in pixels.
    columns
        Number of grid columns. When omitted, FRUST chooses a layout from the
        coordinate selection.
    linked
        If ``True``, link rotation and zoom across all cells.
    show_labels
        If ``True``, draw atom labels.
    show_charges
        If ``True``, display non-zero formal charges.
    kekulize
        Whether to kekulize generated RDKit mol blocks.
    background_color
        Viewer background as a color string or ``(color, opacity)`` tuple.
        This takes precedence over ``dark``.
    export_HTML
        Output path for HTML export. Use ``"none"`` to disable export.

    Returns
    -------
    None
        The viewer is displayed directly in notebook contexts.

    Raises
    ------
    ValueError
        If filters or coordinate selection produce no molecules, or if the
        number of legends does not match the rendered cells.
    """
    scene_background = (
        background_color
        if background_color is not None
        else ("black", 1.0) if dark else ("blue", 0.1)
    )
    scene = molecule_scene_from_dataframe(
        df,
        row_indices=row_indices,
        substrate_filter=substrate_filter,
        rpos_filter=rpos_filter,
        exclude_coords=exclude_coords,
        include_coords=include_coords,
        coord_indices=coord_indices,
        legends=legends,
        cell_size=cell_size,
        columns=columns,
        linked=linked,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        background_color=scene_background,
    )

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
