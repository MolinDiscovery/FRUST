"""Substructure matching helpers for TS guess generation."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from rdkit import Chem

from frust.utils.mols import find_ch, find_unique_ch

CATALYST_SCAFFOLD = Chem.MolFromSmarts("[#5]~c1ccccc1~[#7]")


def mol_from_smiles(smiles: str, *, label: str) -> Chem.Mol:
    """Parse a SMILES string into an RDKit molecule.

    Parameters
    ----------
    smiles : str
        Input SMILES.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    rdkit.Chem.Mol
        Parsed molecule.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for {label!r}: {smiles!r}")
    return mol


def match_catalyst_roles(mol: Chem.Mol, *, catalyst_name: str) -> dict[str, int]:
    """Find the strict B-aryl-N catalyst scaffold.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Catalyst molecule, with or without explicit hydrogens.
    catalyst_name : str
        Catalyst name used in diagnostics.

    Returns
    -------
    dict
        Role mapping containing ``cat_B`` and ``cat_N``.
    """
    matches = mol.GetSubstructMatches(CATALYST_SCAFFOLD)
    if len(matches) != 1:
        raise ValueError(
            f"Catalyst {catalyst_name!r} must contain exactly one B-aryl-N scaffold; "
            f"found {len(matches)} matches"
        )
    match = matches[0]
    return {"cat_B": int(match[0]), "cat_N": int(match[-1])}


def parse_rpos_value(value: Any, smiles: str) -> tuple[int, ...]:
    """Parse and validate a substrate ``rpos`` value.

    Parameters
    ----------
    value : object
        Missing value, integer, sequence, or comma/semicolon-separated string.
    smiles : str
        Substrate SMILES used for validation.

    Returns
    -------
    tuple of int
        Valid aromatic C-H atom indices.
    """
    if _is_missing(value):
        requested = tuple(int(i) for i in find_unique_ch(smiles))
    elif isinstance(value, (list, tuple, set)):
        requested = tuple(int(i) for i in value)
    else:
        text = str(value).strip()
        parts = [part for part in re.split(r"[;,]", text) if part.strip()]
        try:
            requested = tuple(int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError("rpos must contain integers separated by ';' or ','") from exc

    valid = tuple(int(i) for i in find_ch(smiles))
    invalid = sorted(set(requested) - set(valid))
    if invalid:
        raise ValueError(
            f"Invalid rpos values {invalid} for SMILES {smiles!r}. "
            f"Valid aromatic C-H positions: {valid}"
        )
    return requested


def substrate_hydrogen_for_rpos(mol: Chem.Mol, rpos: int, *, substrate_name: str) -> int:
    """Return one explicit hydrogen attached to a substrate reactive atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Substrate molecule with explicit hydrogens.
    rpos : int
        Reactive heavy-atom index.
    substrate_name : str
        Substrate name used in diagnostics.

    Returns
    -------
    int
        Hydrogen atom index.
    """
    atom = mol.GetAtomWithIdx(int(rpos))
    hydrogens = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]
    if not hydrogens:
        raise ValueError(
            f"Substrate {substrate_name!r} rpos {rpos} has no explicit attached hydrogen"
        )
    return int(hydrogens[0])


def hydrogens_on_atom(mol: Chem.Mol, atom_idx: int, *, role: str, minimum: int = 1) -> list[int]:
    """Return explicit hydrogen neighbors for one atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit hydrogens.
    atom_idx : int
        Atom whose hydrogens should be returned.
    role : str
        Role used in error messages.
    minimum : int, optional
        Minimum number of hydrogens required.

    Returns
    -------
    list of int
        Hydrogen atom indices.
    """
    atom = mol.GetAtomWithIdx(int(atom_idx))
    hydrogens = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]
    if len(hydrogens) < minimum:
        raise ValueError(
            f"Atom role {role!r} expected at least {minimum} explicit hydrogens; "
            f"found {len(hydrogens)}"
        )
    return [int(idx) for idx in hydrogens]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
