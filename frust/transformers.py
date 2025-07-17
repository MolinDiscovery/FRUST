# frust/transformers.py
from copy import copy
import logging
import math
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign, rdchem
from rdkit.Chem.AllChem import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdchem import RWMol
from rdkit.Geometry.rdGeometry import Point3D

from typing import Dict, Tuple, List
from .utils.mols import (
    combine_rw_mols,
    remove_one_h,
    get_molecule_name,
    fix_cat_frag,
    fix_pin_frag,
)

logger = logging.getLogger(__name__)


def transformer_ts1(
    ligand_smiles       = "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct     = "ts1.xyz",
    bonds_to_remove     = [(10, 41), (10, 12), (11, 41)],
    H_idx               = 10,
    pre_name            = "TS1",
    embed_ready         = True,
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
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)

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

        atom_idx_to_remove = H_idx
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
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)
    
    return ts_mols


def transformer_ts2(
    ligand_smiles       = "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct     = "ts2.xyz",
    pre_name            = "TS2",
    embed_ready         = True,
):
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

    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)

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

    ts_rw = RWMol(ts)
    ts_rw_origin = copy(ts_rw)
    ts_ligand_pattern = Chem.MolFromSmarts("N1CCCC1")
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)

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

    params = ETKDGv3()
    params.randomSeed = 0xF00D
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

        frags = Chem.GetMolFrags(ts_rw, asMols=True)
        fixed_cat = fix_cat_frag(frags[0])
        fixed_Hs  = frags[1]

        ts_rw = RWMol(Chem.CombineMols(fixed_cat, fixed_Hs))
        offset = ts_rw.GetNumAtoms()
        reactive_H1 = offset - 1
        reactive_H2 = offset - 2

        ts_combined = Chem.CombineMols(ts_rw, lig_mol_rw)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_C = rpos + offset

        cat_pat = Chem.MolFromSmarts('[B]-c1ccccc1-[N]')
        B_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][0]
        N_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][7]

        B_nbs = ts_rw_combined.GetAtomWithIdx(B_cat_idx).GetNeighbors()
        Hs_on_B = [nb.GetIdx() for nb in B_nbs if nb.GetAtomicNum() == 1]

        ts_rw_combined.AddBond(B_cat_idx, reactive_C, Chem.BondType.SINGLE)
        
        atom_indices_to_keep = [B_cat_idx, N_cat_idx]
        atom_indices_to_keep.extend(Hs_on_B)
        atom_indices_to_keep.extend([reactive_H1, reactive_H2, reactive_C])

        if embed_ready:
            ts_rw_combined.AddBond(B_cat_idx, reactive_H1, Chem.BondType.ZERO)

        mol_name = get_molecule_name(ligand_smiles)
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)
    
    return ts_mols


def transformer_ts3(
    ligand_smiles="C1=CC=CO1",
    ts_guess_struct="../structures/ts3.xyz",
    bonds_to_remove = [(10, 11), (10,20)],
    pre_name="TS3",
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
    from rdkit import Chem
    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)
    ts_rw = RWMol(ts)

    bonds_to_remove = bonds_to_remove
    for bond in bonds_to_remove:
        ts_rw.RemoveBond(bond[0], bond[1])
    ts_rw_origin = copy(ts_rw)

    # --- Find ligand in guess ts structure --- #
    ts_ligand_pattern = Chem.MolFromSmarts("S1CCCC1")
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)  # e.g. (5,6,7,8,9)

    # --- Find unique positions and check that they are valid cH --- #
    lig_mol = Chem.MolFromSmiles(ligand_smiles)
    lig_mol = Chem.AddHs(lig_mol)

    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = lig_mol.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]

    atom_rank = list(Chem.CanonicalRankAtoms(lig_mol, breakTies=False))

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

    lig_mol_original = copy(lig_mol)

    for map in maps:
        lig_mol = lig_mol_original
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
        n_pattern_full = Chem.MolFromSmiles("C1CCCS1")
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

        frags = Chem.GetMolFrags(ts_rw, asMols=True)

        frag0 = fix_cat_frag(frags[0])
        rdDetermineBonds.DetermineBonds(frags[1])

        ts_rw = RWMol(Chem.CombineMols(frag0, frags[1]))

        # --- Combine ligand and catalyst, add temporary bonds, and set temporary formal charges ---
        ts_combined = Chem.CombineMols(ts_rw, lig_mol)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_C = rpos + offset

        cat_pat = Chem.MolFromSmarts('[B]-c1ccccc1-[N]')
        B_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][0]
        N_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][7]
        B_nbs = ts_rw_combined.GetAtomWithIdx(B_cat_idx).GetNeighbors()
        Hs_on_B = [nb.GetIdx() for nb in B_nbs if nb.GetAtomicNum() == 1]

        ts_rw_combined.AddBond(reactive_C, B_cat_idx, Chem.BondType.SINGLE)

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw_combined.GetSubstructMatches(pin_pat)[0][0]
        B_pin_nbs = ts_rw_combined.GetAtomWithIdx(B_pin_idx).GetNeighbors()
        H_pin_idx = B_pin_nbs[0].GetIdx()

        atom_indices_to_keep = [B_cat_idx, N_cat_idx]
        atom_indices_to_keep.extend(Hs_on_B)
        atom_indices_to_keep.extend([B_pin_idx])
        atom_indices_to_keep.extend([H_pin_idx, reactive_C])

        if embed_ready:
            pass
            ts_rw_combined.AddBond(B_cat_idx, B_pin_idx, Chem.BondType.SINGLE)
            b_pin_atom = ts_rw_combined.GetAtomWithIdx(B_pin_idx)
            b_cat_atom = ts_rw_combined.GetAtomWithIdx(B_cat_idx)
            b_pin_atom.SetFormalCharge(2)
            b_cat_atom.SetFormalCharge(2)
            
            # c_atom = ts_rw_combined.GetAtomWithIdx(reactive_C)
            # c_atom.SetFormalCharge(0)

        mol_name = get_molecule_name(ligand_smiles)
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)

    return ts_mols


def transformer_ts4(
    ligand_smiles="C1=CC=CO1",
    ts_guess_struct="../structures/ts4.xyz",
    bonds_to_remove = [(11,23)],
    pre_name="TS4",
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
    from rdkit import Chem
    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)
    ts_rw = RWMol(ts)

    bonds_to_remove = bonds_to_remove
    for bond in bonds_to_remove:
        ts_rw.RemoveBond(bond[0], bond[1])
    ts_rw_origin = copy(ts_rw)

    # --- Find ligand in guess ts structure --- #
    ts_ligand_pattern = Chem.MolFromSmarts("S1CCCC1")
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)  # e.g. (5,6,7,8,9)

    # --- Find unique positions and check that they are valid cH --- #
    lig_mol = Chem.MolFromSmiles(ligand_smiles)
    lig_mol = Chem.AddHs(lig_mol)

    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = lig_mol.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]

    atom_rank = list(Chem.CanonicalRankAtoms(lig_mol, breakTies=False))

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

    lig_mol_original = copy(lig_mol)

    for map in maps:
        lig_mol = lig_mol_original
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
        n_pattern_full = Chem.MolFromSmiles("C1CCCS1")
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

        frags = Chem.GetMolFrags(ts_rw, asMols=True)

        rdDetermineBonds.DetermineBonds(frags[0])
        frag1 = fix_pin_frag(frags[1])

        ts_rw = RWMol(Chem.CombineMols(frags[0], frag1))

        # --- Combine ligand and catalyst, add temporary bonds, and set temporary formal charges ---
        ts_combined = Chem.CombineMols(ts_rw, lig_mol)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_C = rpos + offset

        cat_pat = Chem.MolFromSmarts('[B]-c1ccccc1-[N]')
        B_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][0]
        N_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][7]
        B_nbs = ts_rw_combined.GetAtomWithIdx(B_cat_idx).GetNeighbors()
        Hs_on_B = [nb.GetIdx() for nb in B_nbs if nb.GetAtomicNum() == 1]

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw_combined.GetSubstructMatches(pin_pat)[0][0]
        #B_pin_nbs = ts_rw_combined.GetAtomWithIdx(B_pin_idx).GetNeighbors()
        #H_pin_idx = B_pin_nbs[1].GetIdx()

        ts_rw_combined.AddBond(reactive_C, B_pin_idx, Chem.BondType.SINGLE)

        atom_indices_to_keep = [B_cat_idx, N_cat_idx]
        atom_indices_to_keep.extend(Hs_on_B)
        atom_indices_to_keep.extend([B_pin_idx])
        atom_indices_to_keep.extend([reactive_C])

        if embed_ready:
            pass
            ts_rw_combined.AddBond(B_cat_idx, B_pin_idx, Chem.BondType.SINGLE)
            b_pin_atom = ts_rw_combined.GetAtomWithIdx(B_pin_idx)
            b_cat_atom = ts_rw_combined.GetAtomWithIdx(B_cat_idx)
            b_pin_atom.SetFormalCharge(2)
            b_cat_atom.SetFormalCharge(2)
            
            # c_atom = ts_rw_combined.GetAtomWithIdx(reactive_C)
            # c_atom.SetFormalCharge(0)

        mol_name = get_molecule_name(ligand_smiles)
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)

    return ts_mols


def transformer_int3(
    ligand_smiles="C1=CC=CO1",
    ts_guess_struct="../structures/int3.xyz",
    bonds_to_remove = [(10, 19)],
    pre_name="INT3",
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
    from rdkit import Chem
    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)
    ts_rw = RWMol(ts)

    bonds_to_remove = bonds_to_remove
    for bond in bonds_to_remove:
        ts_rw.RemoveBond(bond[0], bond[1])
    ts_rw_origin = copy(ts_rw)

    # --- Find ligand in guess ts structure --- #
    ts_ligand_pattern = Chem.MolFromSmarts("S1CCCC1")
    old_ring_match    = ts_rw.GetSubstructMatch(ts_ligand_pattern)  # e.g. (5,6,7,8,9)

    # --- Find unique positions and check that they are valid cH --- #
    lig_mol = Chem.MolFromSmiles(ligand_smiles)
    lig_mol = Chem.AddHs(lig_mol)

    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = lig_mol.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]

    atom_rank = list(Chem.CanonicalRankAtoms(lig_mol, breakTies=False))

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

    lig_mol_original = copy(lig_mol)

    for map in maps:
        lig_mol = lig_mol_original
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
        n_pattern_full = Chem.MolFromSmiles("C1CCCS1")
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

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw.GetSubstructMatches(pin_pat)[0][0]
        B_pin_nbs = ts_rw.GetAtomWithIdx(B_pin_idx).GetNeighbors()
        H_pin_idx = B_pin_nbs[0].GetIdx()
        
        ts_rw.RemoveBond(B_pin_idx, H_pin_idx)
        frag0, frag1 = Chem.GetMolFrags(ts_rw, asMols=True) 

        frag1 = fix_pin_frag(frag1)

        rdDetermineBonds.DetermineBonds(frag0)

        ts_rw = RWMol(Chem.CombineMols(frag0, frag1))

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw.GetSubstructMatches(pin_pat)[0][0]
        B_pin_nbs = ts_rw.GetAtomWithIdx(B_pin_idx).GetNeighbors()
        H_pin_idx = B_pin_nbs[0].GetIdx()

        ts_rw.AddBond(11, 22, Chem.BondType.SINGLE)

        # --- Combine ligand and catalyst, add temporary bonds, and set temporary formal charges ---
        ts_combined = Chem.CombineMols(ts_rw, lig_mol)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_C = rpos + offset

        cat_pat = Chem.MolFromSmarts('[B]-c1ccccc1-[N]')
        B_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][0]
        N_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][7]
        B_nbs = ts_rw_combined.GetAtomWithIdx(B_cat_idx).GetNeighbors()
        Hs_on_B = [nb.GetIdx() for nb in B_nbs if nb.GetAtomicNum() == 1]

        ts_rw_combined.AddBond(reactive_C, B_cat_idx, Chem.BondType.SINGLE)

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw_combined.GetSubstructMatches(pin_pat)[0][0]
        B_pin_nbs = ts_rw_combined.GetAtomWithIdx(B_pin_idx).GetNeighbors()
        H_pin_idx = B_pin_nbs[2].GetIdx()

        ts_rw_combined.AddBond(reactive_C, B_pin_idx, Chem.BondType.SINGLE)
        atom_indices_to_keep = [B_cat_idx, N_cat_idx]
        atom_indices_to_keep.extend([Hs_on_B[1]])
        atom_indices_to_keep.extend([B_pin_idx])
        atom_indices_to_keep.extend([H_pin_idx, reactive_C])

        ts_rw_combined.GetAtomWithIdx(reactive_C).SetFormalCharge(0)
        ts_rw_combined.GetAtomWithIdx(B_cat_idx).SetFormalCharge(-1)
        ts_rw_combined.GetAtomWithIdx(B_pin_idx).SetFormalCharge(-1)
        ts_rw_combined.GetAtomWithIdx(H_pin_idx).SetFormalCharge(1)

        het_ring = Chem.MolFromSmarts("[!#6;R]")
        hetero_atom_idx = lig_mol_original.GetSubstructMatches(het_ring)
        hetero_atom_idx = offset+hetero_atom_idx[0][0]

        ts_rw_combined.GetAtomWithIdx(hetero_atom_idx).SetFormalCharge(1)

        Chem.SanitizeMol(ts_rw_combined)

        mol_name = get_molecule_name(ligand_smiles)
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)
    
    return ts_mols


def transformer_mols(
    ligand_smiles = "CCCCCCN(CCCCCC)C(=O)c1ccccc1C(F)(F)F",
    catalyst_smiles = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
    only_uniques = False,
    only_generics = False,
    show_IUPAC = True,
    select: str | list[str] | None = None,
    key_prefix: str | None = None,
):
    """
    Build the standard set of cycle molecules:
      dimer, HH, ligand, catalyst, int2_rpos(#), mol2_rpos(#), HBpin-ligand_rpos(#), HBpin-mol

    Flags:
      - only_uniques   : drop any _rpos variants
      - only_generics  : keep only the bare names (and select int2/mol2/HBpin-ligand variants)
      - select         : a name or list from
                         ['dimer','HH','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']
                         to return only those entries (including their _rpos(...) variants).
    """

    # --- normalize select to a list if given ---
    base_names = ['dimer','HH','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']
    if select is not None:
        if isinstance(select, str):
            select = [select]
        bad = set(select) - set(base_names)
        if bad:
            raise ValueError(f"select must be from {base_names}, got {bad}")

    # --- prepare input molecules ---
    catalyst_mol = Chem.MolFromSmiles(catalyst_smiles)
    ligand_mol   = Chem.MolFromSmiles(ligand_smiles)
    catalyst_rw  = RWMol(catalyst_mol)
    ligand_rw    = RWMol(ligand_mol)

    ####################
    ### Create dimer ###
    ####################
    catalyst1 = Chem.MolFromSmiles(catalyst_smiles)
    catalyst2 = Chem.MolFromSmiles(catalyst_smiles)
    B_pattern_dimer = Chem.MolFromSmarts("[B]c1ccccc1")
    B_match_dimer = catalyst1.GetSubstructMatches(B_pattern_dimer)
    B_idx_1 = B_match_dimer[0][0]
    catalyst1 = Chem.AddHs(catalyst1, onlyOnAtoms=[B_idx_1])
    catalyst2 = Chem.AddHs(catalyst2, onlyOnAtoms=[B_idx_1])
    catalyst1_RW = RWMol(catalyst1)
    catalyst2_RW = RWMol(catalyst2)
    dimer, offset = combine_rw_mols(catalyst1_RW, catalyst2_RW)
    BHH_pattern = Chem.MolFromSmarts("B([H])([H])c1ccccc1")
    BHH_match = catalyst1.GetSubstructMatches(BHH_pattern)
    cat1_H1_idx, cat1_H2_idx = BHH_match[0][1], BHH_match[0][2]
    cat2_H1_idx = cat1_H1_idx + offset
    cat2_H2_idx = cat1_H2_idx + offset
    cat1_B_idx  = BHH_match[0][0]
    cat2_B_idx  = cat1_B_idx + offset
    dimer.AddBond(cat1_H1_idx, cat2_B_idx, Chem.BondType.SINGLE)
    dimer.AddBond(cat2_H1_idx, cat1_B_idx, Chem.BondType.SINGLE)
    for idx, charge in [(cat1_B_idx, -1), (cat2_B_idx, -1),
                        (cat1_H1_idx, +1), (cat2_H1_idx, +1)]:
        atom = dimer.GetAtomWithIdx(idx)
        atom.SetFormalCharge(charge)
    dimer_mol = dimer.GetMol()
    Chem.SanitizeMol(dimer_mol)

    ####################
    ### Create HH  ###
    ####################
    HH_mol = Chem.MolFromSmiles("[H][H]")

    ######################
    ### Find unique cH ###
    ######################
    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = ligand_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]
    atom_rank = list(Chem.CanonicalRankAtoms(ligand_rw, breakTies=False))
    def find_unique_atoms(lst):
        seen = set(); out = []
        for i, x in enumerate(lst):
            if x not in seen:
                seen.add(x); out.append(i)
        return out
    unique_cH = set(find_unique_atoms(atom_rank)).intersection(cH_atoms)

    ############################################
    ### Create intermediate 2 and molecule 2 ###
    ############################################
    TMP       = Chem.MolFromSmarts('CC1(C)CCCC(C)(C)N1')
    mol2s = []; int2s = []
    for cH in unique_cH:
        combo_rw, offset = combine_rw_mols(catalyst_rw, ligand_rw)
        combo_rw.AddBond(catalyst_rw.GetAtomWithIdx(0).GetIdx(),
                         cH + offset, Chem.BondType.SINGLE)
        combo_rw.GetAtomWithIdx(0).SetFormalCharge(-1)
        nm = combo_rw.GetSubstructMatches(TMP)[0][9]
        combo_rw.GetAtomWithIdx(nm).SetFormalCharge(+1)
        mol2 = combo_rw.GetMol(); Chem.SanitizeMol(mol2)
        int2 = combo_rw.GetMol(); Chem.SanitizeMol(int2)
        mol2s.append((mol2, cH+offset))
        int2s.append((int2, cH+offset))

    ###########################
    ### Add HBpin to ligand ###
    ###########################
    HBpin_smile  = 'CC1(C)OB([H])OC1(C)C'
    HBpin_mol    = Chem.MolFromSmiles(HBpin_smile)
    HBpin_with_h = Chem.AddHs(HBpin_mol)
    HBpin_rw     = RWMol(HBpin_with_h)
    HBpin_b_idx  = HBpin_rw.GetSubstructMatches(Chem.MolFromSmarts("[B]"))[0][0]
    HBpin_ligands = []
    for cH in unique_cH:
        hrw, offset = combine_rw_mols(HBpin_rw, ligand_rw)
        hrw.AddBond(HBpin_b_idx, cH + offset, Chem.BondType.SINGLE)
        hb_lig = Chem.RemoveHs(hrw)
        HBpin_ligands.append((hb_lig, cH + offset))

    #######################
    ### Finalize output ###
    #######################
    names = ['dimer','HH','ligand','catalyst','int2','mol2','HBpin-ligand','HBpin-mol']
    if show_IUPAC:
        names[2] = get_molecule_name(ligand_smiles)
    mols = [
        dimer_mol,
        HH_mol,
        ligand_mol,
        catalyst_mol,
        int2s,
        mol2s,
        HBpin_ligands,
        HBpin_mol
    ]

    mols_dict: dict[str, Chem.Mol] = {}
    for name, mol in zip(names, mols):
        if only_generics:
            if name not in [names[2], 'int2', 'mol2', 'HBpin-ligand']:
                mols_dict[name] = mol
        else:
            if isinstance(mol, list):
                for m, i in mol:
                    mols_dict[f"{name}_rpos({i})"] = m
            elif not only_uniques:
                mols_dict[name] = mol

    iupac_ligand_name = names[2]

    # --- apply select filter if requested ---
    if select is not None:
        filtered: dict[str, Chem.Mol] = {}
        for choice in select:
            actual = choice
            if choice == "ligand" and show_IUPAC:
                actual = iupac_ligand_name
            for key, m in mols_dict.items():
                if key == actual or key.startswith(f"{actual}_rpos"):
                    filtered[key] = m
        mols_dict = filtered

    if key_prefix is None:
        key_prefix = ligand_smiles
    if key_prefix:
        mols_dict = {f"{key_prefix}_{k}": m for k, m in mols_dict.items()}

    return mols_dict



def transformer_ts2_old(
    ligand_smiles="CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct="../structures/ts2_guess_old.xyz",
    bonds_to_remove = [(59, 62), (52, 62), (52, 54)],
    constraint_atoms=[54, 52, 59, 62], # TS2: H, B1, B2, C ---> B1 = Catalyst, B2 = Pinacolborane
    H_idx=54,
    pre_name="TS2",
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
    from rdkit import Chem
    ts = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(ts, useVdw=True)
    ts_rw = RWMol(ts)
    
    bonds_to_remove = bonds_to_remove
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

    atom_rank = list(Chem.CanonicalRankAtoms(lig_mol, breakTies=False))

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

        atom_idx_to_remove = H_idx
        atom_to_remove = ts_rw.GetAtomWithIdx(atom_idx_to_remove)
        atom_symbol = atom_to_remove.GetSymbol()
        atom_coords = ts_rw.GetConformer().GetAtomPosition(atom_idx_to_remove)

        ts_rw.RemoveAtom(atom_idx_to_remove)
        rdDetermineBonds.DetermineBonds(ts_rw)

        frags = Chem.GetMolFrags(ts_rw, asMols=True)

        fixed_cat = fix_cat_frag(frags[0])
        fixed_pin = fix_pin_frag(frags[1])

        ts_rw = RWMol(Chem.CombineMols(fixed_cat, fixed_pin))

        #  --- Add reactive H back --- #
        new_atom_idx = ts_rw.AddAtom(Chem.Atom(atom_symbol))
        ts_rw.GetConformer().SetAtomPosition(new_atom_idx, atom_coords)          

        # --- Combine ligand and catalyst, add temporary bonds, and set temporary formal charges ---
        ts_combined = Chem.CombineMols(ts_rw, lig_mol)
        ts_rw_combined = RWMol(ts_combined)

        offset = ts_rw.GetNumAtoms()
        reactive_H = offset - 1 # the reactive H is the offset - 1, because it was the last atom added to the mol.
        reactive_C = rpos + offset

        cat_pat = Chem.MolFromSmarts('[B]-c1ccccc1-[N]')
        B_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][0]
        N_cat_idx = ts_rw_combined.GetSubstructMatches(cat_pat)[0][7]
        B_nbs = ts_rw_combined.GetAtomWithIdx(B_cat_idx).GetNeighbors()
        Hs_on_B = [nb.GetIdx() for nb in B_nbs if nb.GetAtomicNum() == 1]

        pin_pat = Chem.MolFromSmarts('[B]1OC(C(O1)(C)C)(C)C')
        B_pin_idx = ts_rw_combined.GetSubstructMatches(pin_pat)[0][0]

        atom_indices_to_keep = [B_cat_idx, N_cat_idx]
        atom_indices_to_keep.extend(Hs_on_B)
        atom_indices_to_keep.extend([B_pin_idx])
        atom_indices_to_keep.extend([reactive_H, reactive_C])

        if embed_ready:
            ts_rw_combined.AddBond(B_cat_idx, reactive_C, Chem.BondType.ZERO)
            ts_rw_combined.AddBond(B_cat_idx, reactive_H, Chem.BondType.SINGLE)
            ts_rw_combined.AddBond(B_cat_idx, B_pin_idx, Chem.BondType.SINGLE)
            b_atom = ts_rw_combined.GetAtomWithIdx(B_cat_idx)
            b_atom.SetFormalCharge(2)
            c_atom = ts_rw_combined.GetAtomWithIdx(reactive_C)
            c_atom.SetFormalCharge(-1)

        mol_name = get_molecule_name(ligand_smiles)
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep, ligand_smiles)

    return ts_mols