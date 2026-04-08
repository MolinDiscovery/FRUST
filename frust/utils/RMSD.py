from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import py3Dmol
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign


def read_xyz(xyz_path: str) -> tuple[list[str], np.ndarray]:
    """Read symbols and coordinates from an XYZ file.

    Args:
        xyz_path: Path to the XYZ file.

    Returns:
        Tuple of atom symbols and coordinates as an (N, 3) array.

    Raises:
        ValueError: If the XYZ file is malformed.
    """
    with open(xyz_path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip() for line in f]

    if len(raw_lines) < 3:
        raise ValueError(f"XYZ file is too short: {xyz_path}")

    try:
        n_atoms = int(raw_lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"First line is not a valid atom count in: {xyz_path}"
        ) from exc

    atom_lines = raw_lines[2:2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atom lines in {xyz_path}, found "
            f"{len(atom_lines)}"
        )

    symbols: list[str] = []
    coords = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line in {xyz_path}: {line}")

        symbol = parts[0]
        x, y, z = map(float, parts[1:4])

        symbols.append(symbol)
        coords.append([x, y, z])

    return symbols, np.asarray(coords, dtype=float)


def xyz_to_rdkit_mol(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
) -> Chem.Mol:
    """Build an RDKit molecule with coordinates and perceived bonds.

    Args:
        symbols: Atom symbols.
        coords: Cartesian coordinates with shape (N, 3).
        charge: Total molecular charge used for bond perception.

    Returns:
        RDKit molecule with one conformer.

    Raises:
        ValueError: If bond perception fails.
    """
    mol = Chem.RWMol()

    for symbol in symbols:
        mol.AddAtom(Chem.Atom(symbol))

    mol = mol.GetMol()
    conf = Chem.Conformer(len(symbols))

    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    mol.AddConformer(conf, assignId=True)
    mol = Chem.Mol(mol)

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception as exc:
        raise ValueError(
            "RDKit bond perception failed. This can happen for some "
            "transition-state geometries."
        ) from exc

    return mol


def get_heavy_mol_with_parent_map(
    mol: Chem.Mol,
) -> tuple[Chem.Mol, list[int]]:
    """Return heavy-atom-only molecule and original atom index mapping.

    Args:
        mol: Input RDKit molecule.

    Returns:
        Tuple of heavy-atom-only RDKit molecule and a list mapping heavy-atom
        indices back to the original atom indices.
    """
    heavy_atom_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    heavy_mol = Chem.RemoveHs(mol)
    return heavy_mol, heavy_atom_indices


def get_best_heavy_atom_map(
    prb_heavy_mol: Chem.Mol,
    ref_heavy_mol: Chem.Mol,
    prb_parent_map: list[int],
    ref_parent_map: list[int],
) -> list[tuple[int, int]]:
    """Find the best heavy-atom map using topology-aware matching.

    Args:
        prb_heavy_mol: Probe heavy-atom-only molecule.
        ref_heavy_mol: Reference heavy-atom-only molecule.
        prb_parent_map: Probe heavy-atom index to original atom index.
        ref_parent_map: Reference heavy-atom index to original atom index.

    Returns:
        Atom map as original-atom-index pairs.

    Raises:
        ValueError: If no valid substructure match is found.
    """
    matches = ref_heavy_mol.GetSubstructMatches(prb_heavy_mol, uniquify=False)
    if not matches:
        raise ValueError(
            "Could not find a heavy-atom substructure match between the two "
            "structures."
        )

    best_rmsd = None
    best_atom_map = None

    for match in matches:
        heavy_atom_map = list(enumerate(match))
        prb_copy = Chem.Mol(prb_heavy_mol)

        rmsd = rdMolAlign.AlignMol(
            prb_copy,
            ref_heavy_mol,
            atomMap=heavy_atom_map,
        )

        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_atom_map = heavy_atom_map

    if best_atom_map is None:
        raise ValueError("Failed to determine a heavy-atom mapping.")

    atom_map = [
        (prb_parent_map[prb_idx], ref_parent_map[ref_idx])
        for prb_idx, ref_idx in best_atom_map
    ]
    atom_map.sort(key=lambda pair: pair[0])

    return atom_map


def mol_to_xyz_block(mol: Chem.Mol, comment: str = "") -> str:
    """Convert an RDKit molecule with one conformer to an XYZ block.

    Args:
        mol: RDKit molecule with one conformer.
        comment: XYZ comment line.

    Returns:
        XYZ-format string.
    """
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        lines.append(
            f"{atom.GetSymbol():<2} "
            f"{pos.x: .10f} {pos.y: .10f} {pos.z: .10f}"
        )

    return "\n".join(lines)


def get_atom_pair_deviations(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    atom_map: list[tuple[int, int]],
) -> pd.DataFrame:
    """Return per-atom mapped deviations after alignment.

    Args:
        prb_mol: Aligned probe molecule.
        ref_mol: Reference molecule.
        atom_map: Atom map as (probe_idx, ref_idx) pairs.

    Returns:
        DataFrame sorted by largest atom-pair deviation first.
    """
    prb_conf = prb_mol.GetConformer()
    ref_conf = ref_mol.GetConformer()

    rows = []
    for prb_idx, ref_idx in atom_map:
        prb_atom = prb_mol.GetAtomWithIdx(prb_idx)
        ref_atom = ref_mol.GetAtomWithIdx(ref_idx)

        prb_pos = prb_conf.GetAtomPosition(prb_idx)
        ref_pos = ref_conf.GetAtomPosition(ref_idx)

        dx = prb_pos.x - ref_pos.x
        dy = prb_pos.y - ref_pos.y
        dz = prb_pos.z - ref_pos.z
        dist = float((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)

        rows.append(
            {
                "probe_idx": prb_idx,
                "ref_idx": ref_idx,
                "probe_symbol": prb_atom.GetSymbol(),
                "ref_symbol": ref_atom.GetSymbol(),
                "distance_A": dist,
                "probe_x": prb_pos.x,
                "probe_y": prb_pos.y,
                "probe_z": prb_pos.z,
                "ref_x": ref_pos.x,
                "ref_y": ref_pos.y,
                "ref_z": ref_pos.z,
            }
        )

    df_dev = pd.DataFrame(rows)
    df_dev = df_dev.sort_values("distance_A", ascending=False).reset_index(
        drop=True
    )
    return df_dev


def show_overlay(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    rmsd: float,
    width: int = 900,
    height: int = 650,
) -> Any:
    """Show a clean aligned overlay.

    Args:
        prb_mol: Aligned probe molecule.
        ref_mol: Reference molecule.
        rmsd: RMSD value to display.
        label: Label prefix for the RMSD box.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol view object.
    """
    prb_xyz = mol_to_xyz_block(prb_mol, "Probe aligned")
    ref_xyz = mol_to_xyz_block(ref_mol, "Reference")

    view = py3Dmol.view(width=width, height=height)

    view.addModel(ref_xyz, "xyz")
    view.setStyle(
        {"model": 0},
        {"stick": {"radius": 0.16}, "sphere": {"scale": 0.23}},
    )

    view.addModel(prb_xyz, "xyz")
    view.setStyle(
        {"model": 1},
        {"stick": {"radius": 0.08}, "sphere": {"scale": 0.14}},
    )

    view.zoomTo()
    view.show()
    return view


def show_overlay_with_deviation_lines(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    df_dev: pd.DataFrame,
    rmsd: float,
    n_lines: int = 10,
    width: int = 950,
    height: int = 700,
) -> Any:
    """Show overlay with dashed lines for the largest mapped deviations.

    Args:
        prb_mol: Aligned probe molecule.
        ref_mol: Reference molecule.
        df_dev: Per-atom deviation DataFrame.
        rmsd: RMSD value to display.
        label: Label prefix for the RMSD box.
        n_lines: Number of largest deviations to draw.
        width: Viewer width in pixels.
        height: Viewer height in pixels.

    Returns:
        py3Dmol view object.
    """
    prb_xyz = mol_to_xyz_block(prb_mol, "Probe aligned")
    ref_xyz = mol_to_xyz_block(ref_mol, "Reference")

    view = py3Dmol.view(width=width, height=height)

    view.addModel(ref_xyz, "xyz")
    view.setStyle(
        {"model": 0},
        {"stick": {"radius": 0.16}, "sphere": {"scale": 0.23}},
    )

    view.addModel(prb_xyz, "xyz")
    view.setStyle(
        {"model": 1},
        {"stick": {"radius": 0.08}, "sphere": {"scale": 0.14}},
    )

    top_dev = df_dev.head(n_lines)

    for _, row in top_dev.iterrows():
        start = {
            "x": float(row["probe_x"]),
            "y": float(row["probe_y"]),
            "z": float(row["probe_z"]),
        }
        end = {
            "x": float(row["ref_x"]),
            "y": float(row["ref_y"]),
            "z": float(row["ref_z"]),
        }
        mid = {
            "x": 0.5 * (start["x"] + end["x"]),
            "y": 0.5 * (start["y"] + end["y"]),
            "z": 0.5 * (start["z"] + end["z"]),
        }

        label_text = (
            f'{row["probe_symbol"]}{int(row["probe_idx"])} → '
            f'{row["ref_symbol"]}{int(row["ref_idx"])}: '
            f'{row["distance_A"]:.3f} Å'
        )

        view.addLine(
            {
                "start": start,
                "end": end,
                "dashed": True,
                "linewidth": 3.0,
            }
        )
        view.addLabel(
            label_text,
            {
                "position": mid,
                "backgroundColor": "white",
                "fontColor": "black",
                "fontSize": 11,
                "showBackground": True,
            },
        )

    view.zoomTo()
    view.show()
    return view


def compare_xyz_rmsd(
    probe_xyz_path: str,
    ref_xyz_path: str,
    atom_scope: str = "heavy",
    charge: int = 0,
    show_overlay_plot: bool = False,
    show_deviation_overlay: bool = False,
    show_table: bool = False,
    top_n: int = 10,
    table_rows: int = 15,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Compare two XYZ structures and optionally visualize the result.

    Parameters
    ----------
    probe_xyz_path : str
        Path to the probe XYZ structure.
    ref_xyz_path : str
        Path to the reference XYZ structure.
    atom_scope : str, optional
        Atom scope to use. Currently supports only "heavy".
    charge : int, optional
        Total molecular charge used during bond perception.
    show_overlay_plot : bool, optional
        Whether to show the aligned overlay.
    show_deviation_overlay : bool, optional
        Whether to show the overlay with deviation lines for the worst
        mapped atoms.
    show_table : bool, optional
        Whether to display the per-atom deviation table.
    top_n : int, optional
        Number of largest deviations to highlight in the deviation overlay.
    table_rows : int, optional
        Number of rows to show in the displayed deviation table.
    print_summary : bool, optional
        Whether to print a short textual summary.

    Returns
    -------
    dict
        Dictionary containing RMSD results, mapping information, aligned
        molecules, heavy-atom molecules, deviation table, and view objects.

    Raises
    ------
    ValueError
        If unsupported options are requested or mapping fails.
    """
    if atom_scope != "heavy":
        raise ValueError(
            "Currently only atom_scope='heavy' is supported."
        )

    probe_symbols, probe_coords = read_xyz(probe_xyz_path)
    ref_symbols, ref_coords = read_xyz(ref_xyz_path)

    probe_mol = xyz_to_rdkit_mol(probe_symbols, probe_coords, charge=charge)
    ref_mol = xyz_to_rdkit_mol(ref_symbols, ref_coords, charge=charge)

    probe_heavy_mol, probe_parent_map = get_heavy_mol_with_parent_map(
        probe_mol
    )
    ref_heavy_mol, ref_parent_map = get_heavy_mol_with_parent_map(ref_mol)

    atom_map = get_best_heavy_atom_map(
        probe_heavy_mol,
        ref_heavy_mol,
        probe_parent_map,
        ref_parent_map,
    )

    probe_mol_aligned = Chem.Mol(probe_mol)
    rmsd = rdMolAlign.AlignMol(
        probe_mol_aligned,
        ref_mol,
        atomMap=atom_map,
    )

    probe_heavy_mol_aligned = Chem.RemoveHs(probe_mol_aligned)
    ref_heavy_mol_final = Chem.RemoveHs(ref_mol)

    df_dev = get_atom_pair_deviations(
        probe_mol_aligned,
        ref_mol,
        atom_map,
    )

    worst_row: Optional[pd.Series]
    worst_row = None if df_dev.empty else df_dev.iloc[0]

    overlay_view = None
    deviation_view = None

    if print_summary:
        print(f"Probe file: {Path(probe_xyz_path).name}")
        print(f"Reference file: {Path(ref_xyz_path).name}")
        print(f"Atom scope: {atom_scope}")
        print(f"Mapped atoms: {len(atom_map)}")
        print(f"RMSD: {rmsd:.6f} Å")
        if worst_row is not None:
            print(
                "Largest mapped deviation: "
                f'{worst_row["probe_symbol"]}{int(worst_row["probe_idx"])} -> '
                f'{worst_row["ref_symbol"]}{int(worst_row["ref_idx"])} = '
                f'{worst_row["distance_A"]:.4f} Å'
            )

    if show_table:
        display(
            df_dev[
                [
                    "probe_idx",
                    "ref_idx",
                    "probe_symbol",
                    "ref_symbol",
                    "distance_A",
                ]
            ]
            .head(table_rows)
            .style.format({"distance_A": "{:.4f}"})
        )

    if show_overlay_plot:
        overlay_view = show_overlay(
            probe_heavy_mol_aligned,
            ref_heavy_mol_final,
            rmsd,
        )

    if show_deviation_overlay:
        deviation_view = show_overlay_with_deviation_lines(
            probe_heavy_mol_aligned,
            ref_heavy_mol_final,
            df_dev,
            rmsd,
            n_lines=top_n,
        )

    return {
        "rmsd": rmsd,
        "atom_scope": atom_scope,
        "atom_map": atom_map,
        "df_dev": df_dev,
        "probe_symbols": probe_symbols,
        "ref_symbols": ref_symbols,
        "probe_mol": probe_mol,
        "ref_mol": ref_mol,
        "probe_mol_aligned": probe_mol_aligned,
        "probe_heavy_mol_aligned": probe_heavy_mol_aligned,
        "ref_heavy_mol": ref_heavy_mol_final,
        "overlay_view": overlay_view,
        "deviation_view": deviation_view,
    }