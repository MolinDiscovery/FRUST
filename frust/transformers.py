# frust/transformers.py
from copy import copy

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign, rdmolops
from rdkit.Chem.AllChem import ETKDGv3, EmbedMolecule, UFFGetMoleculeForceField, EmbedMultipleConfs
from rdkit.Chem.rdchem import RWMol

from .transformer_utils import rotated_maps
from .utils.mols import get_molecule_name

def transformer_ts(
    ligand_smiles="CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct="ts1_guess.xyz",
    bonds_to_remove=[(10, 41), (10, 12), (11, 41)],
    pre_name="TS",
    embed_ready=True,
):
    
    # --- Read TS Guess Structure --- #
    try:
        with open(ts_guess_struct, 'r') as file:
            xyz_block = file.read()
    except FileNotFoundError:
        print(f"Error: Transition state structure file not found: {ts_guess_struct}")
        raise
    except PermissionError:
        print(f"Error: Permission denied when accessing file: {ts_guess_struct}")
        raise
    except IOError as e:
        print(f"Error: Failed to read transition state structure file {ts_guess_struct}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading transition state structure from {ts_guess_struct}: {e}")
        raise

    # --- Determine Connectivity --- #
    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)
    ts_rw = RWMol(ts)

    # --- Remove Bonds --- #
    bonds_to_remove = bonds_to_remove # Finding these bonds might need to be automated.
    for bond in bonds_to_remove:
        ts_rw.RemoveBond(bond[0], bond[1])
    ts_rw_origin = copy(ts_rw)

    # --- Find ligand in guess ts structure --- #
    ts_ligand_pattern = Chem.MolFromSmarts("N1CCCC1")
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)  # e.g. (5,6,7,8,9)

    # --- Find unique positions and check that they are valid cH --- #
    lig_mol = Chem.MolFromSmiles(ligand_smiles)
    lig_mol = Chem.AddHs(lig_mol)

    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = lig_mol.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]

    atom_rank = list(Chem.CanonicalRankAtoms(lig_mol,breakTies=False))
    
    def find_unique_atoms(lst):
        seen = set()
        result = []
        for i, x in enumerate(lst):
            if x not in seen:
                result.append(i)
                seen.add(x)
        return result

    unique_atoms = find_unique_atoms(atom_rank)
    unique_cH = set(unique_atoms).intersection(set(cH_atoms))
    unique_cH = tuple(unique_cH)

    # --- Create aligned maps --- #
    old_active_site = old_ring_match[0:3]

    maps = []
    for a in unique_cH:
        C_pos = lig_mol.GetAtomWithIdx(a)
        nbs = []
        for nb in C_pos.GetNeighbors():
            if nb.GetAtomicNum() == 1:
                pass # hydrogen
            else:
                nbs.append(nb.GetIdx())
        
        nbs.insert(1, C_pos.GetIdx())
        
        map = []
        for nb, aa in zip(nbs, old_active_site):
            map.append((nb, aa))
        maps.append(map)

    # --- Loop through each map a.k.a reactive position and create the molecule --- #
    params = ETKDGv3()
    params.randomSeed = 0xF00D  # Use any integer seed
    ts_mols = {}
    for map in maps:
        rpos = map[1][0]
        EmbedMolecule(lig_mol, params)
        ts_rw = Chem.RWMol(ts_rw_origin)
        rdMolAlign.AlignMol(lig_mol, ts_rw, atomMap=map)

        # --- remove hydrogen from the reacting carbon --- #
        chosen_carbon_idx = rpos
        chosen_carbon = lig_mol.GetAtomWithIdx(chosen_carbon_idx)

        for nb in chosen_carbon.GetNeighbors():
            if nb.GetAtomicNum() == 1:  # hydrogen
                lig_mol_rw = RWMol(lig_mol)
                lig_mol_rw.RemoveAtom(nb.GetIdx())
                lig_mol = lig_mol_rw.GetMol()
                break

        # --- Remove old ligand and determine bond order (to get aromaticity correct for the catalyst) --- #
        n_pattern_full = Chem.MolFromSmiles("CN1CCCC1")
        n_old_indices = ts_rw.GetSubstructMatch(n_pattern_full)

        atoms_to_remove = set()
        for idx in n_old_indices:
            atom = ts_rw.GetAtomWithIdx(idx)
            atoms_to_remove.add(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # Check if hydrogen
                    atoms_to_remove.add(neighbor.GetIdx())

        for idx in sorted(atoms_to_remove, reverse=True):
            ts_rw.RemoveAtom(idx)

        atom_idx_to_remove = 10
        atom_to_remove = ts_rw.GetAtomWithIdx(atom_idx_to_remove)
        atom_symbol = atom_to_remove.GetSymbol()
        atom_coords = ts_rw.GetConformer().GetAtomPosition(atom_idx_to_remove)

        ts_rw.RemoveAtom(atom_idx_to_remove)

        rdDetermineBonds.DetermineBonds(ts_rw)

        #  --- Add reactive H back --- #
        new_atom_idx = ts_rw.AddAtom(Chem.Atom(atom_symbol))
        ts_rw.GetConformer().SetAtomPosition(new_atom_idx, atom_coords)          

        # --- Combine ligand and catalyst, add temporary bonds, and set temporary formal charges ---
        ts_combined = Chem.CombineMols(ts_rw, lig_mol)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_H = offset - 1 # the reactive H is the offset - 1, because it was the last atom added to the mol.
        reactive_C = rpos + offset
        # atom_indices_to_keep = [10, 11, 39, 40, 41, reactive_C]
        atom_indices_to_keep = [10, 11, 39, 40, reactive_H, reactive_C]

        if embed_ready:
            ts_rw_combined.AddBond(10, reactive_C, Chem.BondType.ZERO)
            ts_rw_combined.AddBond(10, reactive_H, Chem.BondType.SINGLE)
            b_atom = ts_rw_combined.GetAtomWithIdx(10)
            b_atom.SetFormalCharge(2)
            c_atom = ts_rw_combined.GetAtomWithIdx(reactive_C)
            c_atom.SetFormalCharge(-1)

        # rpos = lig_match.index(rpos) # reset the index for rpos.

        mol_name = get_molecule_name(ligand_smiles)  
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep)
    
    return ts_mols