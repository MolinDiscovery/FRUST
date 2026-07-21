import logging
from typing import Any, Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdDistGeom, rdMolDescriptors

logger = logging.getLogger(__name__)


def embed_mols(
    mols_dict: Dict[str, Chem.Mol | tuple[Chem.Mol, dict[str, Any]]],
    n_confs: int | None = None,
    n_cores: int = 5,
    optimization: str = 'none',
    max_iters: int = 100
) -> Dict[str, Tuple[Chem.Mol, List[int]] | Tuple[Chem.Mol, List[int], dict[str, Any]]]:
    """
    For each molecule in `mols_dict`:
      1. Add explicit Hs.
      2. Embed `n_confs` conformers (using RDKit's DG). If n_confs is None,
         automatically determine based on rotatable bonds.
      3. Optionally minimize each conformer with UFF or MMFF94.
      4. Return a dict mapping name -> (molecule_with_conformers, conformer_IDs).

    - n_confs: Number of conformers to generate. If None, automatically determined
      based on rotatable bonds (≤7: 50, ≤12: 200, >12: 300).
    - optimization: 'UFF', 'MMFF94', or 'MMFF94s' (case-insensitive).
    - If the molecule cannot be optimized by the chosen force field, we skip that molecule
      with a warning.
    - If `name.lower() == "dimer"`, we skip UFF entirely (by design), but still embed.
    """
    mols_dict_embedded: Dict[str, Tuple[Chem.Mol, List[int]] | Tuple[Chem.Mol, List[int], dict[str, Any]]] = {}

    for name, raw_value in mols_dict.items():
        metadata = None
        raw_mol = raw_value
        if isinstance(raw_value, tuple) and len(raw_value) == 2 and isinstance(raw_value[1], dict):
            raw_mol, metadata = raw_value
        mol = Chem.AddHs(raw_mol)

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
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 0xF00D
            params.numThreads = n_cores

            cids = rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=confs_to_use,
                params=params,
            )

            if len(cids) == 0:
                params = rdDistGeom.ETKDGv3()
                params.randomSeed = 0xF00D
                params.numThreads = n_cores
                params.useRandomCoords = True

                cids = rdDistGeom.EmbedMultipleConfs(
                    mol,
                    numConfs=confs_to_use,
                    params=params,
                )

        except Exception as e:
            logger.warning(f"[{name}] EmbedMultipleConfs failed: {e}")
            mols_dict_embedded[name] = (mol, [], metadata) if metadata else (mol, [])
            continue

        if len(cids) == 0:
            logger.warning(
                f"[{name}] Embedding produced no conformers."
            )
            mols_dict_embedded[name] = (mol, [], metadata) if metadata else (mol, [])
            continue

        opt_method = optimization.strip().upper()
        skip_uff = (opt_method == 'UFF' and name.lower() == 'dimer')

        if opt_method in ('MMFF94', 'MMFF94S'):
            mmff_variant = 'MMFF94' if opt_method == 'MMFF94' else 'MMFF94s'
            for cid in cids:
                props = AllChem.MMFFGetMoleculeProperties(
                    mol,
                    mmffVariant=mmff_variant
                )
                if props is None:
                    logger.warning(
                        f"[{name}][conf {cid}] MMFF ({mmff_variant}) "
                        f"not supported. Skipping."
                    )
                    continue
                _rc = AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant=mmff_variant,
                    confId=cid,
                    maxIters=max_iters
                )
                if _rc != 0:
                    logger.debug(
                        f"[{name}][conf {cid}] MMFF optimization "
                        f"return code {_rc}"
                    )

        elif opt_method == 'UFF' and not skip_uff:
            for cid in cids:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff is None:
                    logger.warning(
                        f"[{name}][conf {cid}] UFF not supported. Skipping."
                    )
                    continue
                rc = ff.Minimize(maxIts=max_iters)
                if rc != 0:
                    logger.debug(
                        f"[{name}][conf {cid}] UFF Minimize return code {rc}"
                    )

        mols_dict_embedded[name] = (mol, list(cids), metadata) if metadata else (mol, list(cids))

    return mols_dict_embedded


def embed_mols_old(
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
