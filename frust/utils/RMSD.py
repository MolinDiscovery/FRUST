from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign


def read_xyz(xyz_path: str) -> tuple[list[str], np.ndarray]:
    """Read symbols and coordinates from an XYZ file.

    Parameters
    ----------
    xyz_path
        Path to the XYZ file.

    Returns
    -------
    tuple of list of str and numpy.ndarray
        Atom symbols and coordinates as an ``(n_atoms, 3)`` array.

    Raises
    ------
    ValueError
        If the XYZ file is malformed.
    """
    with open(xyz_path, "r", encoding="utf-8") as f:
        return read_xyz_block(f.read(), source=str(xyz_path))


def read_xyz_block(xyz_block: str, *, source: str = "XYZ block") -> tuple[list[str], np.ndarray]:
    """Read symbols and coordinates from an XYZ-format text block.

    Parameters
    ----------
    xyz_block
        XYZ-format text. The first line must be the atom count, the second line
        is treated as the comment, and the next ``n_atoms`` lines must start
        with ``symbol x y z``.
    source
        Human-readable source label used in error messages.

    Returns
    -------
    tuple of list of str and numpy.ndarray
        Atom symbols and coordinates as an ``(n_atoms, 3)`` array.

    Raises
    ------
    ValueError
        If the XYZ block is malformed.
    """
    raw_lines = [line.rstrip() for line in str(xyz_block).splitlines()]
    if len(raw_lines) < 3:
        raise ValueError(f"XYZ input is too short: {source}")

    try:
        n_atoms = int(raw_lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"First line is not a valid atom count in: {source}"
        ) from exc

    atom_lines = raw_lines[2:2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atom lines in {source}, found "
            f"{len(atom_lines)}"
        )

    symbols: list[str] = []
    coords = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line in {source}: {line}")

        symbol = parts[0]
        x, y, z = map(float, parts[1:4])

        symbols.append(symbol)
        coords.append([x, y, z])

    return symbols, np.asarray(coords, dtype=float)


def xyz_to_rdkit_mol(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    *,
    perceive_bonds: bool = True,
) -> Chem.Mol:
    """Build an RDKit molecule with coordinates and perceived bonds.

    Parameters
    ----------
    symbols
        Atom symbols.
    coords
        Cartesian coordinates with shape ``(n_atoms, 3)``.
    charge
        Total molecular charge used for bond perception.
    perceive_bonds
        If ``True``, ask RDKit to infer bonds from the 3D geometry. If
        ``False``, return a molecule with atoms and a conformer but no bonds.
        Bond perception is required for topology-based atom mapping, but it is
        not required when the atom map is supplied explicitly or derived from
        matching atom order.

    Returns
    -------
    rdkit.Chem.Mol
        RDKit molecule with one conformer.

    Raises
    ------
    ValueError
        If bond perception is requested and fails.
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

    if perceive_bonds:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception as exc:
            raise ValueError(
                "RDKit bond perception failed. This can happen for some "
                "transition-state geometries. Use mapping='index' or pass "
                "atom_map=... when atom order or atom correspondence is "
                "already known."
            ) from exc

    return mol


def compare_symbols_coords_rmsd(
    probe_symbols: Sequence[str],
    probe_coords: Sequence[Sequence[float]],
    ref_symbols: Sequence[str],
    ref_coords: Sequence[Sequence[float]],
    atom_scope: str = "heavy",
    charge: int = 0,
    mapping: str = "topology",
    atom_map: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compare two structures from atomic symbols and coordinates.

    Parameters
    ----------
    probe_symbols
        Atomic symbols for the structure that will be aligned.
    probe_coords
        Probe Cartesian coordinates with shape ``(n_atoms, 3)``.
    ref_symbols
        Atomic symbols for the reference structure.
    ref_coords
        Reference Cartesian coordinates with shape ``(n_atoms, 3)``.
    atom_scope
        Atom scope used for topology matching and RMSD. Currently only
        ``"heavy"`` is supported, meaning hydrogens are ignored during atom
        mapping and RMSD calculation.
    charge
        Total molecular charge used during RDKit bond perception.
    mapping
        Automatic atom-mapping strategy used when ``atom_map`` is not supplied.
        Use ``"topology"`` to infer bonds, match the heavy-atom molecular graph,
        and choose the lowest-RMSD substructure match. Use ``"index"`` when
        atom order already defines correspondence; this maps heavy atoms by
        their order in the two inputs and does not require bond perception.
    atom_map
        Explicit atom-index pairs as ``(probe_idx, ref_idx)`` in the original
        input atom ordering. When supplied, this takes precedence over
        ``mapping`` and does not require bond perception. For
        ``atom_scope="heavy"``, every mapped pair must contain non-hydrogen
        atoms with matching element symbols.

    Returns
    -------
    dict
        RMSD result containing the aligned probe molecule, reference molecule,
        atom map, heavy-atom parent maps, and per-atom deviation dataframe.

    Raises
    ------
    ValueError
        If coordinates are malformed, ``atom_scope`` is unsupported, bond
        perception fails, or topology-aware atom mapping fails.
    """
    if atom_scope != "heavy":
        raise ValueError(
            "Currently only atom_scope='heavy' is supported."
        )
    if mapping not in {"topology", "index"}:
        raise ValueError("mapping must be either 'topology' or 'index'.")

    probe_symbols_list = [str(symbol) for symbol in probe_symbols]
    ref_symbols_list = [str(symbol) for symbol in ref_symbols]
    probe_coords_arr = np.asarray(probe_coords, dtype=float)
    ref_coords_arr = np.asarray(ref_coords, dtype=float)

    _validate_symbols_coords(probe_symbols_list, probe_coords_arr, "probe")
    _validate_symbols_coords(ref_symbols_list, ref_coords_arr, "reference")

    needs_bonds = atom_map is None and mapping == "topology"
    probe_mol = xyz_to_rdkit_mol(
        probe_symbols_list,
        probe_coords_arr,
        charge=charge,
        perceive_bonds=needs_bonds,
    )
    ref_mol = xyz_to_rdkit_mol(
        ref_symbols_list,
        ref_coords_arr,
        charge=charge,
        perceive_bonds=needs_bonds,
    )

    probe_heavy_mol, probe_parent_map = get_heavy_mol_with_parent_map(
        probe_mol
    )
    ref_heavy_mol, ref_parent_map = get_heavy_mol_with_parent_map(ref_mol)

    if atom_map is not None:
        resolved_atom_map = _validate_atom_map(
            atom_map,
            probe_symbols_list,
            ref_symbols_list,
            atom_scope=atom_scope,
        )
        mapping_used = "explicit"
    elif mapping == "index":
        resolved_atom_map = get_index_atom_map(
            probe_symbols_list,
            ref_symbols_list,
            atom_scope=atom_scope,
        )
        mapping_used = "index"
    else:
        resolved_atom_map = get_best_heavy_atom_map(
            probe_heavy_mol,
            ref_heavy_mol,
            probe_parent_map,
            ref_parent_map,
        )
        mapping_used = "topology"

    probe_mol_aligned = Chem.Mol(probe_mol)
    rmsd = rdMolAlign.AlignMol(
        probe_mol_aligned,
        ref_mol,
        atomMap=resolved_atom_map,
    )

    probe_heavy_mol_aligned = Chem.RemoveHs(probe_mol_aligned)
    ref_heavy_mol_final = Chem.RemoveHs(ref_mol)

    df_dev = get_atom_pair_deviations(
        probe_mol_aligned,
        ref_mol,
        resolved_atom_map,
    )

    return {
        "rmsd": rmsd,
        "atom_scope": atom_scope,
        "mapping": mapping_used,
        "atom_map": resolved_atom_map,
        "df_dev": df_dev,
        "probe_symbols": probe_symbols_list,
        "ref_symbols": ref_symbols_list,
        "probe_coords": probe_coords_arr,
        "ref_coords": ref_coords_arr,
        "probe_mol": probe_mol,
        "ref_mol": ref_mol,
        "probe_mol_aligned": probe_mol_aligned,
        "probe_heavy_mol": probe_heavy_mol,
        "ref_heavy_mol_input": ref_heavy_mol,
        "probe_heavy_parent_map": probe_parent_map,
        "ref_heavy_parent_map": ref_parent_map,
        "probe_heavy_mol_aligned": probe_heavy_mol_aligned,
        "ref_heavy_mol": ref_heavy_mol_final,
    }


def _validate_symbols_coords(
    symbols: Sequence[str],
    coords: np.ndarray,
    label: str,
) -> None:
    """Validate symbol and coordinate array shape."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"{label} coordinates must have shape (n_atoms, 3); "
            f"got {coords.shape}."
        )
    if len(symbols) != coords.shape[0]:
        raise ValueError(
            f"{label} symbols and coordinates have different lengths: "
            f"{len(symbols)} symbols vs {coords.shape[0]} coordinate rows."
        )


def get_heavy_mol_with_parent_map(
    mol: Chem.Mol,
) -> tuple[Chem.Mol, list[int]]:
    """Return heavy-atom-only molecule and original atom index mapping.

    Parameters
    ----------
    mol
        Input RDKit molecule.

    Returns
    -------
    tuple of rdkit.Chem.Mol and list of int
        Heavy-atom-only RDKit molecule and a list mapping heavy-atom indices
        back to the original atom indices.
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

    Parameters
    ----------
    prb_heavy_mol
        Probe heavy-atom-only molecule.
    ref_heavy_mol
        Reference heavy-atom-only molecule.
    prb_parent_map
        Probe heavy-atom index to original atom index.
    ref_parent_map
        Reference heavy-atom index to original atom index.

    Returns
    -------
    list of tuple of int
        Atom map as original-atom-index pairs.

    Raises
    ------
    ValueError
        If no valid substructure match is found.
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


def get_index_atom_map(
    probe_symbols: Sequence[str],
    ref_symbols: Sequence[str],
    *,
    atom_scope: str = "heavy",
) -> list[tuple[int, int]]:
    """Map atoms by input order.

    Parameters
    ----------
    probe_symbols
        Atomic symbols for the structure that will be aligned.
    ref_symbols
        Atomic symbols for the reference structure.
    atom_scope
        Atom scope used for the generated map. Currently only ``"heavy"`` is
        supported, so hydrogens are skipped before comparing atom order.

    Returns
    -------
    list of tuple of int
        Atom-index pairs as ``(probe_idx, ref_idx)`` in original input order.

    Raises
    ------
    ValueError
        If the heavy-atom counts differ or paired heavy atoms have different
        element symbols.
    """
    if atom_scope != "heavy":
        raise ValueError("Currently only atom_scope='heavy' is supported.")

    probe_heavy = [
        (idx, symbol)
        for idx, symbol in enumerate(probe_symbols)
        if str(symbol).upper() != "H"
    ]
    ref_heavy = [
        (idx, symbol)
        for idx, symbol in enumerate(ref_symbols)
        if str(symbol).upper() != "H"
    ]
    if len(probe_heavy) != len(ref_heavy):
        raise ValueError(
            "Cannot use mapping='index' because the structures have different "
            f"heavy-atom counts: probe has {len(probe_heavy)}, reference has "
            f"{len(ref_heavy)}."
        )

    atom_map = [
        (probe_idx, ref_idx)
        for (probe_idx, _), (ref_idx, _) in zip(probe_heavy, ref_heavy)
    ]
    return _validate_atom_map(
        atom_map,
        probe_symbols,
        ref_symbols,
        atom_scope=atom_scope,
    )


def _validate_atom_map(
    atom_map: Sequence[tuple[int, int]],
    probe_symbols: Sequence[str],
    ref_symbols: Sequence[str],
    *,
    atom_scope: str,
) -> list[tuple[int, int]]:
    """Validate an explicit atom map and normalize it to integer pairs."""
    resolved: list[tuple[int, int]] = []
    seen_probe: set[int] = set()
    seen_ref: set[int] = set()

    for pair in atom_map:
        if len(pair) != 2:
            raise ValueError(
                "atom_map entries must be two-item pairs of "
                "(probe_idx, ref_idx)."
            )
        probe_idx = int(pair[0])
        ref_idx = int(pair[1])
        if probe_idx < 0 or probe_idx >= len(probe_symbols):
            raise ValueError(
                f"Probe atom index {probe_idx} is outside the input range "
                f"0..{len(probe_symbols) - 1}."
            )
        if ref_idx < 0 or ref_idx >= len(ref_symbols):
            raise ValueError(
                f"Reference atom index {ref_idx} is outside the input range "
                f"0..{len(ref_symbols) - 1}."
            )
        if probe_idx in seen_probe:
            raise ValueError(f"Probe atom index {probe_idx} appears more than once.")
        if ref_idx in seen_ref:
            raise ValueError(f"Reference atom index {ref_idx} appears more than once.")

        probe_symbol = str(probe_symbols[probe_idx])
        ref_symbol = str(ref_symbols[ref_idx])
        if probe_symbol != ref_symbol:
            raise ValueError(
                "Mapped atoms must have matching element symbols; got "
                f"probe {probe_symbol}{probe_idx} -> reference "
                f"{ref_symbol}{ref_idx}."
            )
        if atom_scope == "heavy" and probe_symbol.upper() == "H":
            raise ValueError(
                "atom_map contains hydrogen atoms, but atom_scope='heavy' "
                "compares only non-hydrogen atoms."
            )

        resolved.append((probe_idx, ref_idx))
        seen_probe.add(probe_idx)
        seen_ref.add(ref_idx)

    if not resolved:
        raise ValueError("atom_map must contain at least one atom pair.")

    return resolved


def mol_to_xyz_block(mol: Chem.Mol, comment: str = "") -> str:
    """Convert an RDKit molecule with one conformer to an XYZ block.

    Parameters
    ----------
    mol
        RDKit molecule with one conformer.
    comment
        XYZ comment line.

    Returns
    -------
    str
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

    Parameters
    ----------
    prb_mol
        Aligned probe molecule.
    ref_mol
        Reference molecule.
    atom_map
        Atom map as ``(probe_idx, ref_idx)`` pairs.

    Returns
    -------
    pandas.DataFrame
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
