# frust/optimizers/uff.py

from typing import List, Tuple
from rdkit.Chem import rdmolops
from rdkit.Chem.AllChem import UFFGetMoleculeForceField
from rdkit import Chem

def optimize_mol_uff(
    mol,
    constraint_atoms: List[int] = None,
    force_constant: float = 1e6,
    energy_tol: float = 1e-4,
    force_tol: float = 1e-3,
    extra_iterations: int = 4,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Minimize the geometry of `mol` using the UFF force field.

    Args:
        mol (rdkit.Chem.Mol or RWMol): A molecule with at least one conformer (confId=0).
        constraint_atoms (List[int], optional): Atom indices to constrain during optimization.
        force_constant (float): Force constant (kcal/mol·Å²) for positional constraints.
        energy_tol (float): Energy convergence tolerance (kcal/mol).
        force_tol (float): Force convergence tolerance (kcal/mol·Å).
        extra_iterations (int): Number of additional minimize steps if initial step converges.

    Returns:
        energy (float): The final UFF-calculated energy.
        coords (List[Tuple[float, float, float]]): Optimized 3D coordinates for each atom.
    """
    # 1) Sanitize & update properties (computes implicit valence)
    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache(strict=True)

    # 2) Now add explicit hydrogens so valences remain correct
    mol = Chem.AddHs(mol)    

    # 3) Perceive rings (improves geometry hints)
    rdmolops.FastFindRings(mol)

    # 4) Set up UFF on confId 0
    ff = UFFGetMoleculeForceField(
        mol,
        confId=0,
        ignoreInterfragInteractions=False
    )

    # 5) Apply positional constraints if requested
    if constraint_atoms:
        for idx in constraint_atoms:
            ff.UFFAddPositionConstraint(idx, 0, force_constant)

    # 6) Minimize
    ff.Initialize()
    converged = ff.Minimize(energyTol=energy_tol, forceTol=force_tol)
    for _ in range(extra_iterations):
        if not converged:
            break
        converged = ff.Minimize(energyTol=energy_tol, forceTol=force_tol)

    # 7) Read energy
    energy = ff.CalcEnergy()

    # 8) Extract optimized coordinates
    conf = mol.GetConformer(0)
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    return energy, coords