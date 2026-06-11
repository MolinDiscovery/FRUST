"""Dataframe API for the tsguess2 SMILES-roundtrip backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
from rdkit.Geometry import Point3D

from frust.tsguess.diagnostics import core_metrics
from frust.tsguess.matching import parse_rpos_value
from frust.tsguess2.builders import (
    build_ts1_ts2_connected_smiles,
    build_ts3_ts4_connected_smiles,
)
from frust.tsguess2.specs import BUILTIN_TS_SPECS_V2, TSGuess2Spec

PRUNE_RMS_THRESH = 0.1
RANDOM_SEED = 0xF00D

_BUILDERS: dict[str, Callable[[str, str, tuple[int, ...] | list[int] | None], dict[int, str]]] = {
    "ts1_ts2": build_ts1_ts2_connected_smiles,
    "ts3_ts4": build_ts3_ts4_connected_smiles,
}


def create_ts_guess_dataframes(
    systems: pd.DataFrame,
    *,
    ts_types: Iterable[str] = ("TS1", "TS2", "TS3", "TS4"),
    n_confs: int | None = 1,
    n_cores: int = 1,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate grouped TS guess dataframes from expanded screen systems.

    Parameters
    ----------
    systems : pandas.DataFrame
        Expanded substrate-catalyst systems from :func:`frust.screen.expand`.
    ts_types : iterable of str, optional
        TS types to generate. Supported values are ``"TS1"``, ``"TS2"``,
        ``"TS3"``, and ``"TS4"``.
    n_confs : int or None, optional
        Number of conformers per generated TS guess. If ``None``, choose a
        count from the generated molecule's rotatable-bond count.
    n_cores : int, optional
        RDKit embedding threads forwarded to ``numThreads``.
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
    conformers_by_type: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in requested_specs}
    smiles_by_type: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in requested_specs}

    for _, system in systems.iterrows():
        substrate_smiles = str(system["substrate_smiles"])
        rpos_values = parse_rpos_value(system.get("rpos"), substrate_smiles)
        for rpos in rpos_values:
            for spec in requested_specs:
                rows, conformer_meta, smiles_meta = _rows_for_system_rpos(
                    system,
                    spec,
                    int(rpos),
                    n_confs=n_confs,
                    n_cores=int(n_cores),
                )
                rows_by_type[spec.name].extend(rows)
                conformers_by_type[spec.name].append(conformer_meta)
                smiles_by_type[spec.name].append(smiles_meta)

    dataframes: dict[str, pd.DataFrame] = {}
    for ts_type, rows in rows_by_type.items():
        df = pd.DataFrame(rows)
        conformer_records = conformers_by_type[ts_type]
        spec = _resolve_spec(ts_type)
        if conformer_records:
            df.attrs["frust_conformers"] = {
                "schema_version": 1,
                "source": "screen.create_ts_guesses",
                "backend": "tsguess2",
                "requested_n_confs": n_confs,
                "n_cores": int(n_cores),
                "n_structures": len(conformer_records),
                "total_generated_confs": int(
                    sum(record["generated_n_confs"] for record in conformer_records)
                ),
                "structures": conformer_records,
            }
        df.attrs["frust_tsguess2"] = {
            "schema_version": 1,
            "backend": "tsguess2",
            "builder": _BUILDERS[spec.builder_key].__name__,
            "spec_id": spec.spec_id,
            "embedding": {
                "maxAttempts": 0,
                "randomSeed": RANDOM_SEED,
                "useRandomCoords": False,
                "pruneRmsThresh": PRUNE_RMS_THRESH,
                "ignoreSmoothingFailures": True,
                "enforceChirality": True,
                "useSmallRingTorsions": True,
                "numThreads": int(n_cores),
            },
            "generated_smiles": smiles_by_type[ts_type],
        }
        dataframes[ts_type] = df
    return dataframes


def _rows_for_system_rpos(
    system: pd.Series,
    spec: TSGuess2Spec,
    rpos: int,
    *,
    n_confs: int | None,
    n_cores: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generated_smiles = _generated_smiles_for_system_rpos(system, spec, rpos)
    mol = Chem.MolFromSmiles(generated_smiles)
    if mol is None:
        raise ValueError(f"Could not parse generated {spec.name} SMILES: {generated_smiles}")
    mol = Chem.AddHs(mol)
    roles = _role_mapping_from_mol(mol, spec)
    resolved_n_confs = _resolve_n_confs(mol, n_confs)
    coord_map = {
        int(roles[role]): Point3D(*spec.role_coordinates[role])
        for role in spec.role_coordinates
    }
    cids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=resolved_n_confs,
            maxAttempts=0,
            randomSeed=RANDOM_SEED,
            useRandomCoords=False,
            pruneRmsThresh=PRUNE_RMS_THRESH,
            coordMap=coord_map,
            ignoreSmoothingFailures=True,
            enforceChirality=True,
            useSmallRingTorsions=True,
            numThreads=int(n_cores),
        )
    )
    if not cids:
        raise ValueError(f"RDKit embedding produced no conformers for {spec.name}")

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    connectivity_bonds = _connectivity_bonds(mol)
    constraint_spec = spec.constraint_dicts()
    constraint_atoms = [roles[role] for role in spec.constraint_order if role in roles]

    rows: list[dict[str, Any]] = []
    system_name = str(system["system_name"])
    custom_name = f"{spec.name}({system_name}_rpos({rpos}))"
    for cid in cids:
        coords = _conf_coords(mol, int(cid))
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
            "smiles": generated_smiles,
            "input_smiles": generated_smiles,
            "rpos": int(rpos),
            "cid": int(cid),
            "atoms": atoms,
            "connectivity_bonds": connectivity_bonds,
            "coords_embedded": coords,
            "constraint_roles": dict(roles),
            "constraint_spec": constraint_spec,
            "constraint_atoms": constraint_atoms,
            "ts_spec_id": spec.spec_id,
            "tsguess_backend": "tsguess2",
            "ts_core_metrics": core_metrics(coords, roles, constraint_spec),
        }
        for col, value in system.items():
            if col not in row and col not in {"rpos", "smiles"}:
                row[col] = value
        rows.append(row)

    conformer_meta = {
        "structure_id": f"{spec.name}:{system_name}:r{rpos}",
        "custom_name": custom_name,
        "structure_type": spec.name,
        "system_name": system_name,
        "substrate_name": str(system["substrate_name"]),
        "catalyst_name": str(system["catalyst_name"]),
        "rpos": int(rpos),
        "requested_n_confs": n_confs,
        "resolved_n_confs": int(resolved_n_confs),
        "generated_n_confs": len(cids),
        "cids": [int(cid) for cid in cids],
    }
    smiles_meta = {
        "structure_id": f"{spec.name}:{system_name}:r{rpos}",
        "rpos": int(rpos),
        "smiles": generated_smiles,
    }
    return rows, conformer_meta, smiles_meta


def _generated_smiles_for_system_rpos(system: pd.Series, spec: TSGuess2Spec, rpos: int) -> str:
    builder = _BUILDERS[spec.builder_key]
    generated = builder(
        str(system["catalyst_smiles"]),
        str(system["substrate_smiles"]),
        rpos_list=(int(rpos),),
    )
    try:
        return generated[int(rpos)]
    except KeyError as exc:
        raise ValueError(f"{spec.name} builder did not return rpos {rpos}") from exc


def _role_mapping_from_mol(mol: Chem.Mol, spec: TSGuess2Spec) -> dict[str, int]:
    query = Chem.MolFromSmarts(spec.core_smarts)
    matches = mol.GetSubstructMatches(query)
    if not matches:
        raise ValueError(f"Could not match {spec.name} core SMARTS: {spec.core_smarts}")
    match = matches[0]

    if spec.name == "TS1":
        return {
            "transfer_H": int(match[0]),
            "cat_N": int(match[1]),
            "cat_B": int(match[4]),
            "substrate_C": int(match[5]),
        }
    if spec.name == "TS2":
        cat_b = int(match[4])
        b_hydrogens = _hydrogen_neighbors(mol, cat_b)
        if not b_hydrogens:
            raise ValueError("Could not find TS2 B_transfer_H on catalyst boron")
        return {
            "N_transfer_H": int(match[0]),
            "cat_N": int(match[1]),
            "cat_B": cat_b,
            "substrate_C": int(match[5]),
            "B_transfer_H": int(b_hydrogens[0]),
        }
    if spec.name in {"TS3", "TS4"}:
        return {
            "cat_B": int(match[0]),
            "transfer_H": int(match[1]),
            "pin_B": int(match[2]),
            "substrate_C": int(match[3]),
        }
    raise ValueError(f"Unsupported tsguess2 spec: {spec.name}")


def _hydrogen_neighbors(mol: Chem.Mol, atom_idx: int) -> list[int]:
    return [
        int(neighbor.GetIdx())
        for neighbor in mol.GetAtomWithIdx(int(atom_idx)).GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    ]


def _conf_coords(mol: Chem.Mol, cid: int) -> list[tuple[float, float, float]]:
    conf = mol.GetConformer(int(cid))
    return [
        tuple(float(v) for v in conf.GetAtomPosition(idx))
        for idx in range(mol.GetNumAtoms())
    ]


def _connectivity_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    return [
        (int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
        for bond in mol.GetBonds()
    ]


def _resolve_n_confs(mol: Chem.Mol, n_confs: int | None) -> int:
    if n_confs is not None:
        return int(n_confs)

    rdmolops.FastFindRings(mol)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if rotatable_bonds <= 7:
        return 50
    if rotatable_bonds <= 12:
        return 200
    return 300


def _resolve_spec(ts_type: str) -> TSGuess2Spec:
    key = str(ts_type).upper()
    try:
        return BUILTIN_TS_SPECS_V2[key]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_TS_SPECS_V2))
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
