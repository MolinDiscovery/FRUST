"""Assemble catalyst/substrate systems into TS guess dataframes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from rdkit import Chem

from frust.tsguess.diagnostics import core_metrics
from frust.tsguess.embedding import embed_with_coord_map
from frust.tsguess.matching import (
    hydrogens_on_atom,
    match_catalyst_roles,
    mol_from_smiles,
    parse_rpos_value,
    substrate_hydrogen_for_rpos,
)
from frust.tsguess.specs import BUILTIN_TS_SPECS, TSSpec

HBPIN_SMILES = "CC1(C)OB([H])OC1(C)C"


def create_ts_guess_dataframes(
    systems: pd.DataFrame,
    *,
    ts_types: Iterable[str] = ("TS1", "TS2", "TS3", "TS4"),
    n_confs: int = 1,
    n_cores: int = 1,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate grouped TS guess dataframes from expanded screen systems.

    Parameters
    ----------
    systems : pandas.DataFrame
        Expanded substrate-catalyst systems from :func:`frust.screen.expand`.
    ts_types : iterable of str, optional
        TS types to generate.
    n_confs : int, optional
        Number of conformers per generated TS guess.
    n_cores : int, optional
        RDKit embedding threads.
    validate : bool, optional
        If ``True``, validate required system columns before generation.

    Returns
    -------
    dict
        Mapping from TS type to FRUST initial dataframe.
    """
    if validate:
        _validate_systems(systems)

    requested_specs = [_resolve_spec(ts_type) for ts_type in ts_types]
    rows_by_type: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in requested_specs}

    for _, system in systems.iterrows():
        substrate_smiles = str(system["substrate_smiles"])
        rpos_values = parse_rpos_value(system.get("rpos"), substrate_smiles)
        for rpos in rpos_values:
            for spec in requested_specs:
                rows_by_type[spec.name].extend(
                    _rows_for_system_rpos(
                        system,
                        spec,
                        int(rpos),
                        n_confs=int(n_confs),
                        n_cores=int(n_cores),
                    )
                )

    return {ts_type: pd.DataFrame(rows) for ts_type, rows in rows_by_type.items()}


def _rows_for_system_rpos(
    system: pd.Series,
    spec: TSSpec,
    rpos: int,
    *,
    n_confs: int,
    n_cores: int,
) -> list[dict[str, Any]]:
    mol, roles = _assemble_system_mol(system, spec, rpos)
    coord_map = {
        roles[role]: spec.role_coordinates[role]
        for role in _placement_roles(spec)
        if role in roles
    }
    embedded, cids = embed_with_coord_map(
        mol,
        coord_map,
        n_confs=n_confs,
        n_cores=n_cores,
        allowed_contact_pairs=_allowed_contact_pairs(spec, roles),
    )
    atoms = [atom.GetSymbol() for atom in embedded.GetAtoms()]
    connectivity_bonds = _connectivity_bonds(embedded)
    constraint_spec = spec.constraint_dicts()
    constraint_atoms = [roles[role] for role in spec.constraint_order if role in roles]

    rows: list[dict[str, Any]] = []
    for cid in cids:
        conf = embedded.GetConformer(int(cid))
        coords = [
            tuple(float(v) for v in conf.GetAtomPosition(idx))
            for idx in range(embedded.GetNumAtoms())
        ]
        system_name = str(system["system_name"])
        custom_name = f"{spec.name}({system_name}_rpos({rpos}))"
        row = {
            "custom_name": custom_name,
            "structure_id": f"{spec.name}:{system_name}:r{rpos}",
            "structure_type": spec.name,
            "molecule_role": "ts",
            "system_name": system_name,
            "substrate_name": str(system["substrate_name"]),
            "catalyst_name": str(system["catalyst_name"]),
            "substrate_smiles": str(system["substrate_smiles"]),
            "catalyst_smiles": str(system["catalyst_smiles"]),
            "smiles": str(system["substrate_smiles"]),
            "input_smiles": str(system["substrate_smiles"]),
            "rpos": int(rpos),
            "cid": int(cid),
            "atoms": atoms,
            "connectivity_bonds": connectivity_bonds,
            "coords_embedded": coords,
            "constraint_roles": dict(roles),
            "constraint_spec": constraint_spec,
            "constraint_atoms": constraint_atoms,
            "ts_spec_id": spec.spec_id,
            "ts_core_metrics": core_metrics(coords, roles, constraint_spec),
        }
        for col, value in system.items():
            if col not in row and col not in {"rpos", "smiles"}:
                row[col] = value
        rows.append(row)
    return rows


def _assemble_system_mol(
    system: pd.Series,
    spec: TSSpec,
    rpos: int,
) -> tuple[Chem.Mol, dict[str, int]]:
    catalyst = Chem.AddHs(
        mol_from_smiles(str(system["catalyst_smiles"]), label=str(system["catalyst_name"]))
    )
    substrate_with_h = Chem.AddHs(
        mol_from_smiles(str(system["substrate_smiles"]), label=str(system["substrate_name"]))
    )
    substrate, substrate_rpos = _remove_substrate_hydrogen_for_rpos(
        substrate_with_h,
        int(rpos),
        substrate_name=str(system["substrate_name"]),
    )
    cat_roles = match_catalyst_roles(catalyst, catalyst_name=str(system["catalyst_name"]))

    parts = [catalyst, substrate]
    offsets = [0, catalyst.GetNumAtoms()]
    extra_offset = catalyst.GetNumAtoms() + substrate.GetNumAtoms()
    extra = _extra_fragment(spec)
    if extra is not None:
        parts.append(extra)
        offsets.append(extra_offset)

    combined = _combine_mols(parts)
    roles: dict[str, int] = {
        "cat_B": cat_roles["cat_B"],
        "cat_N": cat_roles["cat_N"],
        "substrate_C": offsets[1] + substrate_rpos,
    }
    catalyst_b_h = hydrogens_on_atom(catalyst, cat_roles["cat_B"], role="cat_B", minimum=1)

    if spec.name == "TS1":
        roles["transfer_H"] = _single_h_role(extra, extra_offset)
    elif spec.name == "TS2":
        roles["cat_H"] = catalyst_b_h[0]
        roles.update(_h2_roles(extra, extra_offset))
    elif spec.name == "TS3":
        roles["transfer_H"] = catalyst_b_h[0]
        roles["cat_H"] = catalyst_b_h[1] if len(catalyst_b_h) > 1 else catalyst_b_h[0]
        roles["pin_B"] = _pin_b_role(extra, extra_offset)
    elif spec.name == "TS4":
        roles["cat_H"] = catalyst_b_h[0]
        roles["transfer_H"] = catalyst_b_h[1] if len(catalyst_b_h) > 1 else catalyst_b_h[0]
        roles["pin_B"] = _pin_b_role(extra, extra_offset)
    else:
        raise ValueError(f"Unsupported TS spec: {spec.name}")

    missing = sorted(set(spec.role_coordinates) - set(roles))
    if missing:
        raise ValueError(f"Could not assign required roles for {spec.name}: {missing}")
    return combined, roles


def _extra_fragment(spec: TSSpec) -> Chem.Mol | None:
    if spec.extra_fragment is None:
        return None
    if spec.extra_fragment == "H":
        return Chem.MolFromSmiles("[H]", sanitize=False)
    if spec.extra_fragment == "H2":
        return Chem.MolFromSmiles("[H][H]", sanitize=False)
    if spec.extra_fragment == "HBpin":
        return Chem.AddHs(Chem.MolFromSmiles(HBPIN_SMILES))
    raise ValueError(f"Unknown extra fragment: {spec.extra_fragment!r}")


def _remove_substrate_hydrogen_for_rpos(
    substrate: Chem.Mol,
    rpos: int,
    *,
    substrate_name: str,
) -> tuple[Chem.Mol, int]:
    """Remove the explicit hydrogen attached to a substrate reactive atom.

    Parameters
    ----------
    substrate : rdkit.Chem.Mol
        Substrate molecule with explicit hydrogens.
    rpos : int
        Reactive heavy-atom index.
    substrate_name : str
        Substrate name used in diagnostics.

    Returns
    -------
    tuple
        Dehydrogenated substrate and adjusted reactive atom index.
    """
    hydrogen_idx = substrate_hydrogen_for_rpos(
        substrate,
        int(rpos),
        substrate_name=substrate_name,
    )
    editable = Chem.RWMol(substrate)
    editable.RemoveAtom(int(hydrogen_idx))
    dehydrogenated = editable.GetMol()
    dehydrogenated.UpdatePropertyCache(strict=False)
    adjusted_rpos = int(rpos) - int(hydrogen_idx < int(rpos))
    return dehydrogenated, adjusted_rpos


def _single_h_role(extra: Chem.Mol | None, offset: int) -> int:
    """Return the role atom for a built-in single-hydrogen fragment.

    Parameters
    ----------
    extra : rdkit.Chem.Mol or None
        Built-in extra fragment.
    offset : int
        Atom offset for the extra fragment.

    Returns
    -------
    int
        Atom index for the hydrogen role.
    """
    if extra is None or extra.GetNumAtoms() != 1:
        raise ValueError("TS1 requires a built-in single-hydrogen fragment")
    atom = extra.GetAtomWithIdx(0)
    if atom.GetAtomicNum() != 1:
        raise ValueError("TS1 built-in fragment must be hydrogen")
    return int(offset)


def _h2_roles(extra: Chem.Mol | None, offset: int) -> dict[str, int]:
    if extra is None or extra.GetNumAtoms() != 2:
        raise ValueError("TS2 requires a built-in H2 fragment")
    return {"n_transfer_H": offset, "transfer_H": offset + 1}


def _pin_b_role(extra: Chem.Mol | None, offset: int) -> int:
    if extra is None:
        raise ValueError("TS3/TS4 require a built-in HBpin fragment")
    matches = extra.GetSubstructMatches(Chem.MolFromSmarts("[#5]"))
    if len(matches) != 1:
        raise ValueError(f"HBpin fragment must contain exactly one boron; found {len(matches)}")
    return offset + int(matches[0][0])


def _combine_mols(parts: list[Chem.Mol]) -> Chem.Mol:
    combined = parts[0]
    for part in parts[1:]:
        combined = Chem.CombineMols(combined, part)
    return combined


def _connectivity_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Return covalent connectivity for dataframe storage and plotting.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule whose bonds should be stored.

    Returns
    -------
    list of tuple of int
        Zero-based atom-index pairs.
    """
    return [
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    ]


def _placement_roles(spec: TSSpec) -> tuple[str, ...]:
    """Return the role anchors used to place disconnected TS fragments.

    Parameters
    ----------
    spec : TSSpec
        Transition-state specification.

    Returns
    -------
    tuple of str
        Role names used as hard placement anchors.
    """
    if spec.name == "TS1":
        return ("cat_B", "cat_N", "transfer_H", "substrate_C")
    if spec.name == "TS2":
        return ("cat_B", "cat_N", "cat_H", "transfer_H", "n_transfer_H", "substrate_C")
    if spec.name == "TS3":
        return ("cat_B", "transfer_H", "pin_B", "substrate_C")
    if spec.name == "TS4":
        return ("cat_B", "transfer_H", "pin_B", "substrate_C")
    return tuple(spec.role_coordinates)


def _allowed_contact_pairs(spec: TSSpec, roles: dict[str, int]) -> set[tuple[int, int]]:
    """Return nonbonded role pairs expected to be close in a TS guess.

    Parameters
    ----------
    spec : TSSpec
        Transition-state specification.
    roles : dict
        Mapping from role name to atom index.

    Returns
    -------
    set of tuple of int
        Atom-index pairs to ignore during clash scoring.
    """
    pairs = {
        tuple(sorted((roles[left], roles[right])))
        for constraint in spec.constraints
        if constraint.kind == "distance"
        for left, right in [constraint.roles[:2]]
        if left in roles and right in roles
    }
    if spec.name == "TS2" and {"cat_B", "substrate_C"} <= set(roles):
        pairs.add(tuple(sorted((roles["cat_B"], roles["substrate_C"]))))
    return pairs


def _resolve_spec(ts_type: str) -> TSSpec:
    key = str(ts_type).upper()
    try:
        return BUILTIN_TS_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_TS_SPECS))
        raise ValueError(f"Unsupported ts_type {ts_type!r}; expected one of {known}") from exc


def _validate_systems(systems: pd.DataFrame) -> None:
    required = {
        "system_name",
        "substrate_name",
        "catalyst_name",
        "substrate_smiles",
        "catalyst_smiles",
    }
    missing = sorted(required - set(systems.columns))
    if missing:
        raise ValueError("systems dataframe is missing required columns: " + ", ".join(missing))
