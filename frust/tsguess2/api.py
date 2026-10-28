"""Dataframe API for the tsguess2 SMILES-roundtrip backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import warnings

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
from rdkit.Geometry import Point3D

from frust.tsguess.diagnostics import core_metrics
from frust.tsguess.matching import parse_rpos_value
from frust.tsguess2.builders import (
    BISARYL_ROLE_MAPS,
    BISARYL_EXCHANGE_ROLE_MAPS,
    build_bisaryl_adduct_connected_smiles,
    build_bisaryl_diaryl_connected_smiles,
    build_bisaryl_exchange_connected_smiles,
    build_ts1_ts2_connected_smiles,
    build_ts3_ts4_connected_smiles,
)
from frust.tsguess2.specs import (
    SpecSelection,
    TSGuess2Spec,
    normalize_profile_id,
    resolve_profile_specs,
)

PRUNE_RMS_THRESH = -1
RANDOM_SEED = 0xF00D

_BISARYL_ADDUCT_BUILDERS = frozenset({"bisaryl_adduct", "bisaryl_h2"})
_BISARYL_BUILDERS = _BISARYL_ADDUCT_BUILDERS | {"bisaryl_diaryl", "bisaryl_exchange"}

_BUILDERS: dict[str, Callable[[str, str, tuple[int, ...] | list[int] | None], dict[int, str]]] = {
    "bisaryl_adduct": build_bisaryl_adduct_connected_smiles,
    "bisaryl_h2": build_bisaryl_adduct_connected_smiles,
    "bisaryl_diaryl": build_bisaryl_diaryl_connected_smiles,
    "bisaryl_exchange": build_bisaryl_exchange_connected_smiles,
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
    spec_profile: str = "wb97xd3-631g/gas",
    spec_match: str = "prefer-exact",
    candidate_specs: Mapping[str, TSGuess2Spec] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Generate grouped TS guess dataframes from expanded screen systems.

    Parameters
    ----------
    systems : pandas.DataFrame
        Expanded substrate-catalyst systems from :func:`frust.screen.expand`.
    ts_types : iterable of str, optional
        States to generate. Supported values are ``"TS1"``, ``"TS2"``,
        ``"TS3"``, ``"TS4"``, and the constrained minimum ``"INT3"``.
    n_confs : int or None, optional
        Number of conformers per generated TS guess. If ``None``, choose a
        count from the generated molecule's rotatable-bond count.
    n_cores : int, optional
        RDKit embedding threads forwarded to ``numThreads``.
    validate : bool, optional
        If ``True``, validate required system columns before generation.
    spec_profile : str, optional
        Method/environment geometry profile. The default preserves the
        original wB97X-D3/6-31G** gas-phase specification.
    spec_match : {"prefer-exact", "exact"}, optional
        Profile resolution policy. ``"prefer-exact"`` permits the other
        environment of the same method and records the fallback. It never
        crosses method families.
    candidate_specs : mapping of str to TSGuess2Spec or None, optional
        Explicit experimental geometry specifications for requested states.
        Each specification must match its state and the requested method/profile.
        They are used only for this call, are recorded as unvalidated candidates
        in dataframe attributes, and never enter the active profile registry.
        Experimental ``"TS5"`` (second C-H activation) and ``"TS6"`` (H2
        elimination) require explicit specifications. Both use a bisaryl
        N-H/B-H construction graph with different reactive geometries. Systems may
        specify ``bound_substrate_smiles`` and ``bound_rpos`` independently of
        the incoming substrate; omitted values use a second molecule of the
        same substrate at the same site. ``"TS7"``, ``"INT6"`` and ``"TS8"``
        candidates use a bisaryl/HBpin exchange construction graph. This shared
        graph does not assign a reaction or distinguish a saddle from a minimum;
        the selected template supplies geometry, and validation supplies identity.
        ``"INT4"`` uses the bisaryl N-H/B-H adduct, while ``"INT5"`` is the
        neutral diaryl catalyst after H2 elimination. Intermediate candidates
        require a minimum workflow for subsequent optimization.
    random_seed : int, optional
        RDKit embedding seed, retained in conformer provenance.

    Returns
    -------
    dict
        Mapping from TS type to FRUST initial dataframe.
    """
    if validate:
        _validate_systems(systems)
    if spec_match not in {"prefer-exact", "exact"}:
        raise ValueError("spec_match must be 'prefer-exact' or 'exact'")

    requested_types = [str(ts_type).upper() for ts_type in ts_types]
    candidates = {str(state).upper(): spec for state, spec in (candidate_specs or {}).items()}
    unknown_candidates = set(candidates) - set(requested_types)
    if unknown_candidates:
        raise ValueError(f"candidate_specs includes unrequested states: {sorted(unknown_candidates)}")
    selections = resolve_profile_specs(
        [state for state in requested_types if state not in candidates],
        spec_profile,
        match=spec_match,
    )
    for state, candidate in candidates.items():
        if candidate.name != state or candidate.profile_id != normalize_profile_id(spec_profile):
            raise ValueError("Candidate state and method/profile must match the requested state/profile")
        if candidate.builder_key not in _BUILDERS:
            raise ValueError(f"Unsupported candidate builder: {candidate.builder_key}")
        selections[state] = SpecSelection(state, candidate.profile_id, candidate.profile_id, "exact", candidate)
    requested_specs = [selections[ts_type].spec for ts_type in requested_types]
    fallback_profiles = {
        (selection.requested_profile, selection.resolved_profile)
        for selection in selections.values()
        if selection.match != "exact"
    }
    for requested, resolved in sorted(fallback_profiles):
        warnings.warn(
            f"No exact selectable tsguess2 geometry was available for every requested "
            f"state in {requested!r}; using same-method profile {resolved!r} where needed.",
            UserWarning,
            stacklevel=2,
        )
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
                    random_seed=int(random_seed),
                )
                rows_by_type[spec.name].extend(rows)
                conformers_by_type[spec.name].append(conformer_meta)
                smiles_by_type[spec.name].append(smiles_meta)

    dataframes: dict[str, pd.DataFrame] = {}
    for ts_type, rows in rows_by_type.items():
        df = pd.DataFrame(rows)
        conformer_records = conformers_by_type[ts_type]
        selection = selections[ts_type]
        spec = selection.spec
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
            "preserve_explicit_hydrogens": spec.builder_key in _BISARYL_BUILDERS,
            "spec_id": spec.spec_id,
            "spec_selection": {
                "requested_profile": selection.requested_profile,
                "resolved_profile": selection.resolved_profile,
                "match": selection.match,
                "state_specs": {ts_type: spec.spec_id},
            },
            "embedding": {
                "maxAttempts": 0,
                "randomSeed": int(random_seed),
                "useRandomCoords": False,
                "pruneRmsThresh": PRUNE_RMS_THRESH,
                "ignoreSmoothingFailures": True,
                "enforceChirality": True,
                "useSmallRingTorsions": True,
                "numThreads": int(n_cores),
            },
            "generated_smiles": smiles_by_type[ts_type],
            "candidate_spec": ts_type in candidates,
            "scientific_status": "unvalidated_candidate" if ts_type in candidates else "registered_profile",
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
    random_seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    generated_smiles = _generated_smiles_for_system_rpos(system, spec, rpos)
    if spec.builder_key in _BISARYL_BUILDERS:
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(generated_smiles, parser)
    else:
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
            randomSeed=int(random_seed),
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

    rows: list[dict[str, Any]] = []
    system_name = str(system["system_name"])
    custom_name = f"{spec.name}({system_name}_rpos({rpos}))"
    for cid in cids:
        coords = _conf_coords(mol, int(cid))
        row = {
            "custom_name": custom_name,
            "structure_id": f"{spec.name}:{system_name}:r{rpos}",
            "structure_type": spec.name,
            "molecule_role": "ts" if spec.name.startswith("TS") else spec.name.lower(),
            "state_id": spec.name,
            "state_kind": (
                "transition_state" if spec.name.startswith("TS") else "constrained_minimum"
            ),
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
            "ts_spec_id": spec.spec_id,
            "tsguess_backend": "tsguess2",
            "ts_core_metrics": core_metrics(coords, roles, constraint_spec),
        }
        input_only = {"rpos", "smiles"}
        if spec.builder_key in _BISARYL_BUILDERS:
            input_only.update({"bound_substrate_smiles", "bound_rpos"})
        for col, value in system.items():
            if col not in row and col not in input_only:
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
    if spec.builder_key in _BISARYL_BUILDERS:
        bound_smiles = system.get("bound_substrate_smiles")
        bound_rpos = system.get("bound_rpos")
        smiles_meta["builder_inputs"] = {
            "bound_substrate_smiles": (str(system["substrate_smiles"])
                                       if bound_smiles is None or pd.isna(bound_smiles) else str(bound_smiles)),
            "bound_rpos": int(rpos if bound_rpos is None or pd.isna(bound_rpos) else bound_rpos),
            "incoming_substrate_smiles": str(system["substrate_smiles"]),
            "incoming_rpos": int(rpos),
        }
    return rows, conformer_meta, smiles_meta


def _generated_smiles_for_system_rpos(system: pd.Series, spec: TSGuess2Spec, rpos: int) -> str:
    builder = _BUILDERS[spec.builder_key]
    extra = {}
    if spec.builder_key in _BISARYL_BUILDERS:
        bound_smiles = system.get("bound_substrate_smiles")
        bound_rpos = system.get("bound_rpos")
        if bound_smiles is not None and not pd.isna(bound_smiles):
            extra["bound_substrate_smiles"] = str(bound_smiles)
        if bound_rpos is not None and not pd.isna(bound_rpos):
            extra["bound_rpos"] = int(bound_rpos)
    generated = builder(
        str(system["catalyst_smiles"]),
        str(system["substrate_smiles"]),
        rpos_list=(int(rpos),),
        **extra,
    )
    try:
        return generated[int(rpos)]
    except KeyError as exc:
        raise ValueError(f"{spec.name} builder did not return rpos {rpos}") from exc


def _role_mapping_from_mol(mol: Chem.Mol, spec: TSGuess2Spec) -> dict[str, int]:
    if spec.builder_key == "bisaryl_diaryl":
        mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()
                  if atom.GetAtomMapNum()}
        tags = {role: tag for role, tag in BISARYL_ROLE_MAPS.items()
                if role not in {"N_transfer_H", "B_transfer_H"}}
        missing = set(tags.values()) - set(mapped)
        if missing:
            raise ValueError(f"Diaryl construction lost chemical role maps: {sorted(missing)}")
        roles = {role: int(mapped[tag]) for role, tag in tags.items()}
        boron = mol.GetAtomWithIdx(roles["cat_B"])
        nitrogen = mol.GetAtomWithIdx(roles["cat_N"])
        if (boron.GetAtomicNum() != 5 or nitrogen.GetAtomicNum() != 7
                or boron.GetFormalCharge() or nitrogen.GetFormalCharge()
                or boron.GetTotalNumHs(includeNeighbors=True)
                or nitrogen.GetTotalNumHs(includeNeighbors=True)):
            raise ValueError("Diaryl catalyst requires neutral B/N without N-H or B-H")
        for role in ("bound_C", "incoming_C", "scaffold_B_C"):
            if (mol.GetAtomWithIdx(roles[role]).GetAtomicNum() != 6
                    or mol.GetBondBetweenAtoms(roles["cat_B"], roles[role]) is None):
                raise ValueError(f"Invalid diaryl connectivity: cat_B--{role}")
        return roles
    if spec.builder_key == "bisaryl_exchange":
        mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
        missing = set(BISARYL_EXCHANGE_ROLE_MAPS.values()) - set(mapped)
        if missing:
            raise ValueError(f"Bisaryl exchange lost chemical role maps: {sorted(missing)}")
        roles = {role: int(mapped[tag]) for role, tag in BISARYL_EXCHANGE_ROLE_MAPS.items()}
        expected = {"cat_B": 5, "cat_N": 7, "pin_B": 5, "transfer_H": 1,
                    "exchanging_C": 6, "retained_C": 6, "pin_O1": 8, "pin_O2": 8}
        for role, atomic_number in expected.items():
            if mol.GetAtomWithIdx(roles[role]).GetAtomicNum() != atomic_number:
                raise ValueError(f"Invalid exchange chemical role: {role}")
        for a, b in [("cat_B", "transfer_H"), ("pin_B", "transfer_H"),
                     ("cat_B", "exchanging_C"), ("pin_B", "exchanging_C"),
                     ("cat_B", "retained_C")]:
            if mol.GetBondBetweenAtoms(roles[a], roles[b]) is None:
                raise ValueError(f"Invalid exchange connectivity: {a}--{b}")
        return roles
    if spec.builder_key in _BISARYL_ADDUCT_BUILDERS:
        mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
        for h_role, parent_role in [("N_transfer_H", "cat_N"), ("B_transfer_H", "cat_B")]:
            tag = BISARYL_ROLE_MAPS[h_role]
            parent = mapped.get(BISARYL_ROLE_MAPS[parent_role])
            if tag not in mapped and parent is not None:
                hydrogens = _hydrogen_neighbors(mol, parent)
                if len(hydrogens) == 1:
                    mapped[tag] = hydrogens[0]
        missing = set(BISARYL_ROLE_MAPS.values()) - set(mapped)
        if missing:
            raise ValueError(f"Bisaryl construction lost chemical role maps: {sorted(missing)}")
        roles = {role: int(mapped[tag]) for role, tag in BISARYL_ROLE_MAPS.items()}
        expected = {"cat_B": 5, "cat_N": 7, "N_transfer_H": 1, "B_transfer_H": 1,
                    "bound_C": 6, "incoming_C": 6}
        for role, atomic_number in expected.items():
            if mol.GetAtomWithIdx(roles[role]).GetAtomicNum() != atomic_number:
                raise ValueError(f"Invalid bisaryl chemical role: {role}")
        for a, b in [("cat_B", "bound_C"), ("cat_B", "incoming_C"),
                     ("cat_B", "B_transfer_H"), ("cat_N", "N_transfer_H")]:
            if mol.GetBondBetweenAtoms(roles[a], roles[b]) is None:
                raise ValueError(f"Invalid bisaryl connectivity: {a}--{b}")
        return roles
    query = Chem.MolFromSmarts(spec.core_smarts)
    if query is None:
        raise ValueError(f"Invalid {spec.name} core SMARTS: {spec.core_smarts}")
    matches = mol.GetSubstructMatches(query)
    if not matches:
        raise ValueError(f"Could not match {spec.name} core SMARTS: {spec.core_smarts}")
    if spec.name in {"TS3", "TS4", "INT3"} and len(matches) != 1:
        raise ValueError(
            f"Expected one chemically directed {spec.name} core SMARTS match; "
            f"found {len(matches)}"
        )
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
    if spec.name in {"TS3", "TS4", "INT3"}:
        roles = {
            "cat_B": int(match[0]),
            "transfer_H": int(match[1]),
            "pin_B": int(match[2]),
            "substrate_C": int(match[3]),
        }
        _validate_hbpin_bridged_roles(mol, roles, state=spec.name)
        return roles
    raise ValueError(f"Unsupported tsguess2 spec: {spec.name}")


def _validate_hbpin_bridged_roles(
    mol: Chem.Mol,
    roles: dict[str, int],
    *,
    state: str,
) -> None:
    """Validate chemically directed TS3/TS4/INT3 role assignments.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Hydrogen-explicit connected TS molecule.
    roles : dict
        Candidate mapping for catalyst B, HBpin B, bridge H, and substrate C.
    state : str
        State name used in diagnostic messages.
    """
    expected_atomic_numbers = {
        "cat_B": 5,
        "pin_B": 5,
        "transfer_H": 1,
        "substrate_C": 6,
    }
    for role, atomic_number in expected_atomic_numbers.items():
        atom = mol.GetAtomWithIdx(roles[role])
        if atom.GetAtomicNum() != atomic_number:
            raise ValueError(
                f"Invalid {state} role mapping: {role} is not atomic number "
                f"{atomic_number}"
            )

    pin_b = roles["pin_B"]
    pin_oxygen_count = sum(
        neighbor.GetAtomicNum() == 8
        for neighbor in mol.GetAtomWithIdx(pin_b).GetNeighbors()
    )
    if pin_oxygen_count != 2:
        raise ValueError(
            f"Invalid {state} role mapping: pin_B must be bonded to two oxygens; "
            f"found {pin_oxygen_count}"
        )

    cat_b = roles["cat_B"]
    for role in ("transfer_H", "substrate_C"):
        role_idx = roles[role]
        if (
            mol.GetBondBetweenAtoms(cat_b, role_idx) is None
            or mol.GetBondBetweenAtoms(pin_b, role_idx) is None
        ):
            raise ValueError(
                f"Invalid {state} role mapping: {role} must connect catalyst B "
                "and HBpin B"
            )


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
