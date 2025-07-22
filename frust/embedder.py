import logging
logger = logging.getLogger(__name__)

from typing import Dict, Tuple, List, Union
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdDistGeom, rdMolDescriptors
from rdkit.Chem.rdchem import Mol, RWMol

from tooltoad.vis import MolTo3DGrid

TSValue = Tuple[
    Mol,                 # ts_with_H
    List[int],           # conformer IDs
    List[int],           # atom_indices_to_keep
    str,                 # smiles
    List[Tuple[float,int]]  # energies list (may be empty)
]


def embed_ts(
    ts_data: Union[Dict[str, Tuple[Mol, List[int], str]], Mol],
    atom_indices_to_keep: List[int] = None,
    *,
    ts_type: str = "TS1",
    n_confs: None | int = None,
    n_cores: int = 1,
    optimize: bool = False,
    force_constant: float = 1e6,
    energy_tol: float = 1e-4,
    force_tol: float = 1e-3,
    extra_iterations: int = 4,
    smi: str = ""
) -> Union[Dict[str, TSValue], TSValue]:
    """
    Embed conformers for one TS or a dict of TS entries and (optionally) run UFF.

    Parameters
    ----------
    ts_data
        • Single Mol  – call with `atom_indices_to_keep` and optionally `smi`\n
        • Dict[str, (Mol, keep_idxs, smiles)] – typically what `transformer_ts` returns.
    atom_indices_to_keep
        Constraint atoms (needed only in single-Mol mode).
    n_confs
        Number of conformers to generate. If None, automatically determined
        based on rotatable bonds (≤7: 50, ≤12: 200, >12: 300).
    n_cores
        Passed to RDKit EmbedMultipleConfs.
    optimize
        If True, each embedded conformer is minimized with UFF **inside this function**.
    force_constant, energy_tol, force_tol, extra_iterations
        Parameters for the UFF call when `optimize=True`.
    smi
        SMILES (single-Mol mode only).

    Returns
    -------
    • Single-Mol call   → (mol_with_H, cids, keep_idxs, smi, energies)\n
    • Dict call         → { key: (mol_with_H, cids, keep_idxs, smi, energies) }
      where `energies` is [] if `optimize=False`.
    """

    # ---------- DICT BRANCH ----------
    if isinstance(ts_data, dict):
        result: Dict[str, TSValue] = {}
        for name, (mol, keep_idxs, smi_in) in ts_data.items():
            # Calculate n_confs if not provided
            confs_to_use = n_confs
            if confs_to_use is None:
                rdmolops.FastFindRings(mol)
                N_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                if N_rot <= 7:
                    confs_to_use = 50
                elif N_rot <= 12:
                    confs_to_use = 200
                else:
                    confs_to_use = 300

            result[name] = embed_ts(
                mol,
                keep_idxs,
                ts_type=ts_type,
                n_confs=confs_to_use,
                n_cores=n_cores,
                optimize=optimize,
                force_constant=force_constant,
                energy_tol=energy_tol,
                force_tol=force_tol,
                extra_iterations=extra_iterations,
                smi=smi_in,
            )
        return result

    # ---------- SINGLE-MOL BRANCH ----------
    ts_mol: Mol = ts_data
    atom_indices_to_keep = atom_indices_to_keep or []

    # Build coord map to freeze selected atoms during embedding
    coord_map = {}
    for idx in atom_indices_to_keep:
        if ts_type.upper() in ("TS3", "TS4"):
            atom = ts_mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "N":
                continue
        pos = ts_mol.GetConformer().GetAtomPosition(idx)
        coord_map[idx] = pos

    # Calculate conformations to create
    if n_confs is None:
        rdmolops.FastFindRings(ts_mol)
        N_rot = rdMolDescriptors.CalcNumRotatableBonds(ts_mol)
        if N_rot <= 7:
            n_confs = 50
        elif N_rot <= 12:
            n_confs = 200
        else:
            n_confs = 300

    # Prepare molecule with explicit Hs
    ts_mol.UpdatePropertyCache(strict=True)
    ts_with_H = Chem.AddHs(ts_mol)
    ts_with_H.UpdatePropertyCache(strict=False)

    # Embed
    cids = rdDistGeom.EmbedMultipleConfs(
        ts_with_H,
        n_confs,
        maxAttempts=0,
        randomSeed=0xF00D,
        useRandomCoords=False,
        pruneRmsThresh=0.5,
        coordMap=coord_map,
        ignoreSmoothingFailures=True,
        enforceChirality=True,
        useSmallRingTorsions=True,
        numThreads=n_cores
    )

    ts_with_H = RWMol(ts_with_H)
    if ts_type.upper() == "TS1":
        # Remove temporary bonds (hard-coded: 10-reactive_C / 10-reactive_H)
        reactive_C = atom_indices_to_keep[-1]
        reactive_H = atom_indices_to_keep[-2]
        ts_with_H.RemoveBond(10, reactive_C)
        ts_with_H.RemoveBond(10, reactive_H)

    # if ts_type.upper() == "TS2":
    #     reactive_C = atom_indices_to_keep[-1]
    #     reactive_H = atom_indices_to_keep[-2]
    #     cat_B      = atom_indices_to_keep[0]
    #     pin_B      = atom_indices_to_keep[-3]
    #     ts_with_H.RemoveBond(reactive_C, cat_B)
    #     ts_with_H.RemoveBond(reactive_H, cat_B)
    #     ts_with_H.RemoveBond(pin_B, cat_B)

    if ts_type.upper() == "TS2":
        reactive_C = atom_indices_to_keep[-1]
        reactive_H = atom_indices_to_keep[-3]
        cat_B      = atom_indices_to_keep[0]
        ts_with_H.RemoveBond(cat_B, reactive_H)

    if ts_type.upper() == "TS3":
        reactive_C = atom_indices_to_keep[-1]
        cat_B      = atom_indices_to_keep[0]
        pin_B      = atom_indices_to_keep[-3]
        ts_with_H.RemoveBond(pin_B, cat_B)
    
    if ts_type.upper() == "TS4":
        reactive_C = atom_indices_to_keep[-1]
        cat_B      = atom_indices_to_keep[0]
        pin_B      = atom_indices_to_keep[-2]
        ts_with_H.RemoveBond(pin_B, cat_B)    

    if ts_type.upper() == "INT3":
        reactive_C = atom_indices_to_keep[-1]
        cat_B      = atom_indices_to_keep[0]
        pin_B      = atom_indices_to_keep[-3]   
        H_pin = atom_indices_to_keep[-2]

        ts_with_H.AddBond(pin_B, H_pin, Chem.BondType.SINGLE) # only for visualization
        ts_with_H.AddBond(cat_B, reactive_C, Chem.BondType.SINGLE) # only for visualization

        ts_with_H.RemoveBond(cat_B, pin_B)

    print(f"Embedded {len(cids)} conformers on atom {reactive_C}")

    # ---------- OPTIONAL UFF ----------
    energies: List[Tuple[float, int]] = []
    if optimize:
        rdmolops.FastFindRings(ts_with_H)
        for cid in cids:
            ff = AllChem.UFFGetMoleculeForceField(
                ts_with_H, confId=cid, ignoreInterfragInteractions=False
            )
            for idx in atom_indices_to_keep:
                ff.UFFAddPositionConstraint(idx, 0, force_constant)

            ff.Initialize()
            converged = ff.Minimize(energyTol=energy_tol, forceTol=force_tol)
            for _ in range(extra_iterations):
                if not converged:
                    break
                converged = ff.Minimize(energyTol=energy_tol, forceTol=force_tol)

            energies.append((ff.CalcEnergy(), cid))

    return ts_with_H, list(cids), atom_indices_to_keep, smi, energies


def embed_mols(
    mols_dict: Dict[str, Chem.Mol],
    n_confs: int | None = None,
    n_cores: int      = 5,
    optimization: str = 'none',
    max_iters: int    = 100
) -> Dict[str, Tuple[Chem.Mol, List[int]]]:
    """
    For each molecule in `mols_dict`:
      1. Add explicit Hs.
      2. Embed `n_confs` conformers (using RDKit's DG). If n_confs is None, 
         automatically determine based on rotatable bonds.
      3. Optionally minimize each conformer with UFF or MMFF94.
      4. Return a dict mapping name -> (molecule_with_conformers, conformer_IDs).

    - n_confs: Number of conformers to generate. If None, automatically determined
      based on rotatable bonds (≤7: 50, ≤12: 200, >12: 300).
    - optimization: 'UFF', 'MMFF94', or 'MMFF94s' (case‐insensitive).
    - If the molecule cannot be optimized by the chosen force field, we skip that molecule
      with a warning.
    - If `name.lower() == "dimer"`, we skip UFF entirely (by design), but still embed.
    """
    mols_dict_embedded: Dict[str, Tuple[Chem.Mol, List[int]]] = {}

    for name, raw_mol in mols_dict.items():
        mol = Chem.AddHs(raw_mol)

        # Calculate n_confs if not provided
        confs_to_use = n_confs
        if confs_to_use is None:
            rdmolops.FastFindRings(mol)
            N_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if N_rot <= 7:
                confs_to_use = 50
            elif N_rot <= 12:
                confs_to_use = 200
            else:
                confs_to_use = 300

        try:
            cids = rdDistGeom.EmbedMultipleConfs(
                mol, 
                numConfs   = confs_to_use,
                randomSeed = 0xF00D,
                numThreads = n_cores,
            )
        except Exception as e:
            logger.warning(f"[{name}] EmbedMultipleConfs failed: {e}")
            mols_dict_embedded[name] = (mol, [])
            continue

        opt_method = optimization.strip().upper()
        skip_uff = (opt_method == 'UFF' and name.lower() == 'dimer')

        if opt_method in ('MMFF94', 'MMFF94S'):
            mmff_variant = 'MMFF94' if opt_method == 'MMFF94' else 'MMFF94s'
            for cid in cids:
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
                if props is None:
                    logger.warning(f"[{name}][conf {cid}] MMFF ({mmff_variant}) not supported. Skipping.")
                    continue
                _rc = AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant=mmff_variant,
                    confId=cid,
                    maxIters=max_iters
                )
                if _rc != 0:
                    logger.debug(f"[{name}][conf {cid}] MMFF optimization return code {_rc}")

        elif opt_method == 'UFF' and not skip_uff:
            for cid in cids:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff is None:
                    logger.warning(f"[{name}][conf {cid}] UFF not supported. Skipping.")
                    continue
                rc = ff.Minimize(maxIts=max_iters)
                if rc != 0:
                    logger.debug(f"[{name}][conf {cid}] UFF Minimize return code {rc}")

        mols_dict_embedded[name] = (mol, list(cids))

    return mols_dict_embedded