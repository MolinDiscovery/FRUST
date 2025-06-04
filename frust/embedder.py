from typing import Dict, Tuple, List, Any, Union
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import Mol, RWMol


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
    n_confs: int = 2,
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
    n_confs, n_cores
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
            result[name] = embed_ts(
                mol,
                keep_idxs,
                n_confs=n_confs,
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
    coord_map = {
        idx: ts_mol.GetConformer().GetAtomPosition(idx)
        for idx in atom_indices_to_keep
    }

    # Prepare molecule with explicit Hs
    ts_mol.UpdatePropertyCache(strict=True)
    ts_with_H = Chem.AddHs(ts_mol)
    ts_with_H.UpdatePropertyCache(strict=False)

    # Embed
    from rdkit.Chem import rdDistGeom

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
        numThreads=n_cores
    )

    # Remove temporary bonds (hard-coded: 10-reactive_C / 10-reactive_H)
    reactive_C = atom_indices_to_keep[-1]
    reactive_H = atom_indices_to_keep[-2]
    ts_with_H = RWMol(ts_with_H)
    ts_with_H.RemoveBond(10, reactive_C)
    ts_with_H.RemoveBond(10, reactive_H)

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