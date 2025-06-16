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
from .utils.mols import combine_rw_mols, remove_one_h, get_molecule_name

logger = logging.getLogger(__name__)


def transformer_ts1(
    ligand_smiles       = "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct     = "ts1_guess.xyz",
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
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep)
    
    return ts_mols


def transformer_ts2(
    ligand_smiles="CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
    ts_guess_struct="../structures/ts2_guess.xyz",
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

    BH_LEN = 1.18   # Å – average B–H bond length

    def _unit(vec):
        """return unit vector (or None if zero length)"""
        norm = math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
        if norm < 1e-6:
            return None
        return vec.__class__(vec.x / norm, vec.y / norm, vec.z / norm)


    def _fix_pin_frag(frag: Chem.Mol) -> Chem.Mol:
        """
        • converts B=O double bonds → single, neutralises B/O  
        • adds a B–H 1.18 Å away from B but does **not** move any other atom
        """
        rw   = Chem.RWMol(frag)
        conf = rw.GetConformer()

        # ---- locate boron ----
        b_idx = next(a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 5)
        boron = rw.GetAtomWithIdx(b_idx)

        # ---- make B-O single & neutral ----
        for nb in boron.GetNeighbors():
            bond = rw.GetBondBetweenAtoms(b_idx, nb.GetIdx())
            if bond.GetBondType() == rdchem.BondType.DOUBLE:
                bond.SetBondType(rdchem.BondType.SINGLE)
            if nb.GetAtomicNum() == 8:
                nb.SetFormalCharge(0)
        boron.SetFormalCharge(0)

        # ---- add the missing H (if needed) ----
        if boron.GetTotalDegree() < 3:
            # 1) choose a direction opposite to the average of B→heavy-neighbor vectors
            b_pos = conf.GetAtomPosition(b_idx)
            acc   = b_pos.__class__(0.0, 0.0, 0.0)
            heavy_cnt = 0
            for nb in boron.GetNeighbors():
                if nb.GetAtomicNum() > 1:        # O or C
                    n_pos = conf.GetAtomPosition(nb.GetIdx())
                    acc.x += n_pos.x - b_pos.x
                    acc.y += n_pos.y - b_pos.y
                    acc.z += n_pos.z - b_pos.z
                    heavy_cnt += 1
            # average & flip
            acc.x *= -1.0 / heavy_cnt
            acc.y *= -1.0 / heavy_cnt
            acc.z *= -1.0 / heavy_cnt
            direction = _unit(acc) or b_pos.__class__(1.0, 0.0, 0.0)  # fallback

            # 2) place H at BH_LEN along that direction
            h_pos = b_pos.__class__(
                b_pos.x + direction.x * BH_LEN,
                b_pos.y + direction.y * BH_LEN,
                b_pos.z + direction.z * BH_LEN,
            )

            # 3) add the atom & bond
            h_idx = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(b_idx, h_idx, rdchem.BondType.SINGLE)
            conf.SetAtomPosition(h_idx, h_pos)

        # ---- sanitize & return ----
        Chem.SanitizeMol(rw)
        rw.RemoveAtom(h_idx)

        return rw.GetMol()

    def _fix_cat_frag(mol: Chem.Mol, bh_len: float = 1.19) -> Chem.Mol:
        """
        • Finds the single B atom in *mol*
        • Adds one H at a standard B–H distance without moving any existing atoms
        • Neutralises B if it was −1
        • Rebuilds all connectivity from the final 3D coords (rings, valence, aromaticity)
        """
        rw   = Chem.RWMol(mol)
        conf = rw.GetConformer()

        # locate boron and neutralise if needed
        b_idx = next(a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 5)
        boron = rw.GetAtomWithIdx(b_idx)
        if boron.GetFormalCharge() == -1:
            boron.SetFormalCharge(0)

        # pick direction roughly opposite its neighbours
        bpos   = np.array(conf.GetAtomPosition(b_idx))
        neighs = [n.GetIdx() for n in boron.GetNeighbors()]
        if neighs:
            pts = np.array([conf.GetAtomPosition(i) for i in neighs])
            v   = bpos - pts.mean(axis=0)
        else:
            v   = np.array([1.0, 0.0, 0.0])
        v /= np.linalg.norm(v) or 1.0

        # add the H at ~bh_len Å from B
        h_pos = bpos + bh_len * v
        h_idx = rw.AddAtom(Chem.Atom(1))

        rw.AddBond(b_idx, h_idx, Chem.BondType.SINGLE)
        conf.SetAtomPosition(h_idx, Point3D(*h_pos))

        # rebuild connectivity purely from 3D coords
        xyz = Chem.MolToXYZBlock(rw)
        mol2 = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(mol2, useVdw=True)
        Chem.SanitizeMol(mol2)

        rw = RWMol(mol2)
        rw.RemoveAtom(h_idx)

        return rw.GetMol()
    
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

        fixed_cat = _fix_cat_frag(frags[0])
        fixed_pin = _fix_pin_frag(frags[1])

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
        ts_mols[f'{pre_name}({mol_name}_rpos({rpos}))'] = (ts_rw_combined, atom_indices_to_keep)

    return ts_mols


def transformer_mols(
    ligand_smiles = "CCCCCCN(CCCCCC)C(=O)c1ccccc1C(F)(F)F",
    catalyst_smiles = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
    only_uniques = False,
    only_generics = False,
    show_IUPAC=True,
):
    catalyst_mol = Chem.MolFromSmiles(catalyst_smiles)
    ligand_mol   = Chem.MolFromSmiles(ligand_smiles)

    catalyst_rw = RWMol(catalyst_mol)
    ligand_rw   = RWMol(ligand_mol)

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

    cat1_H1_idx = BHH_match[0][1]
    cat1_H2_idx = BHH_match[0][2]
    cat2_H1_idx = cat1_H1_idx + offset
    cat2_H2_idx = cat1_H2_idx + offset

    cat1_B_idx = BHH_match[0][0]
    cat2_B_idx = cat1_B_idx + offset

    dimer.AddBond(cat1_H1_idx, cat2_B_idx, Chem.BondType.SINGLE)
    dimer.AddBond(cat2_H1_idx, cat1_B_idx, Chem.BondType.SINGLE)

    cat1_B = dimer.GetAtomWithIdx(cat1_B_idx)
    cat2_B = dimer.GetAtomWithIdx(cat2_B_idx)
    cat1_B.SetFormalCharge(-1)
    cat2_B.SetFormalCharge(-1)

    cat1_H1 = dimer.GetAtomWithIdx(cat1_H1_idx)
    cat2_H1 = dimer.GetAtomWithIdx(cat2_H1_idx)
    cat1_H1.SetFormalCharge(+1)
    cat2_H1.SetFormalCharge(+1)

    dimer_mol = dimer.GetMol()
    Chem.SanitizeMol(dimer_mol)

    ######################
    ### Find unique cH ###
    ######################    
    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = ligand_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]

    atom_rank = list(Chem.CanonicalRankAtoms(ligand_rw,breakTies=False))

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

    ############################################
    ### Create intermediate 2 and molecule 2 ###
    ############################################
    b_pattern = Chem.MolFromSmarts("[B]")
    pyrrole_attach_pattern = Chem.MolFromSmarts("[n,s,o]1[cH][c][c][c]1")

    catalyst_matches = catalyst_mol.GetSubstructMatches(b_pattern)

    if not catalyst_matches:
        raise ValueError("No [B] atom found in the catalyst.")

    catalyst_b_idx = catalyst_matches[0][0]

    # print("Catalyst: B is at atom idx =", catalyst_b_idx)

    mol2s = []
    int2s = []
    for cH in unique_cH:

        combined_rw, offset = combine_rw_mols(catalyst_rw, ligand_rw)
        combined_mol = combined_rw.GetMol()
        Chem.SanitizeMol(combined_mol)
        
        b_idx_combined   = catalyst_b_idx
        ch_idx_combined  = cH + offset

        combined_rw.AddBond(b_idx_combined, ch_idx_combined, Chem.BondType.SINGLE)

        mol2 = combined_rw.GetMol()

        boron = combined_rw.GetAtomWithIdx(catalyst_b_idx)
        boron.SetFormalCharge(-1)

        TMP = Chem.MolFromSmarts('CC1(C)CCCC(C)(C)N1')
        TMP_match = combined_rw.GetSubstructMatches(TMP)
        nitrogen = combined_rw.GetAtomWithIdx(TMP_match[0][9])
        nitrogen.SetFormalCharge(+1)

        int2 = combined_rw.GetMol()
        Chem.SanitizeMol(mol2)
        Chem.SanitizeMol(int2)

        mol2s.append((mol2, ch_idx_combined))
        int2s.append((int2, ch_idx_combined))

    ###########################
    ### Add HBpin to ligand ###
    ###########################
    HBpin_smile = 'CC1(C)OB([H])OC1(C)C'
    HBpin_mol = Chem.MolFromSmiles(HBpin_smile)
    HBpin_with_hs = Chem.AddHs(HBpin_mol)
    HBpin_rw      = RWMol(HBpin_with_hs)
    HBpin_b_idx   = HBpin_rw.GetSubstructMatches(b_pattern)[0][0]

    remove_one_h(HBpin_rw, HBpin_b_idx)

    HBpin_ligands = []
    for cH in unique_cH:
        combined_HBpin, offset = combine_rw_mols(HBpin_rw, ligand_rw)

        b_idx_combined_HBpin = HBpin_b_idx
        ch_idx_combined = cH + offset

        combined_HBpin.AddBond(b_idx_combined_HBpin, ch_idx_combined, Chem.BondType.SINGLE)
        HBpin_ligand = Chem.RemoveHs(combined_HBpin)

        HBpin_ligands.append((HBpin_ligand, ch_idx_combined))


    #######################
    ### Finalize output ###
    #######################
    names = ['dimer', 'ligand', 'catalyst', 'int2', 'mol2', 'HBpin-ligand', 'HBpin-mol']
    if show_IUPAC: 
        name = get_molecule_name(ligand_smiles)
        names[1] = name

    mols = [dimer, ligand_mol, catalyst_mol, int2s, mol2s, HBpin_ligands, HBpin_mol]
    mols_dict = {}
    for name, mol in zip(names, mols):
        if only_generics:
            if name not in ['int2', 'mol2', 'HBpin-ligand']:
                mols_dict[name] = mol
        else:
            if name in ['int2', 'mol2', 'HBpin-ligand']:
                for m, i in mol:
                    n = f"{name}_rpos({i})"
                    mols_dict[n] = m
            elif not only_uniques:
                mols_dict[name] = mol

    return mols_dict