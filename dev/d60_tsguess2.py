from rdkit import Chem
from rdkit.Chem import RWMol


def combine_rw_mols(mol1: Chem.Mol, mol2: Chem.Mol) -> tuple[RWMol, int]:
    combined = Chem.CombineMols(mol1, mol2)
    offset = mol1.GetNumAtoms()

    return RWMol(combined), offset


def build_ts1_template_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
) -> dict[int, str]:
    catalyst_mol = Chem.MolFromSmiles(catalyst_smiles)
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)

    if catalyst_mol is None:
        raise ValueError(f"Could not parse catalyst SMILES: {catalyst_smiles}")

    if substrate_mol is None:
        raise ValueError(f"Could not parse substrate SMILES: {substrate_smiles}")

    substrate_rw = RWMol(substrate_mol)

    b_pattern = Chem.MolFromSmarts("[B]")
    catalyst_matches = catalyst_mol.GetSubstructMatches(b_pattern)

    if not catalyst_matches:
        raise ValueError("No [B] atom found in the catalyst.")

    catalyst_b_idx = catalyst_matches[0][0]

    amine_query = Chem.MolFromSmarts("[NX3]([CH3])([CH3])~[#6]")
    amine_matches = catalyst_mol.GetSubstructMatches(amine_query)

    if not amine_matches:
        raise ValueError("Could not find catalyst NMe2 nitrogen.")

    catalyst_n_idx = amine_matches[0][0]

    ######################
    ### Find unique cH ###
    ######################
    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = substrate_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]
    atom_rank = list(Chem.CanonicalRankAtoms(substrate_rw, breakTies=False))

    def find_unique_atoms(lst):
        seen = set(); out = []
        for i, x in enumerate(lst):
            if x not in seen:
                seen.add(x); out.append(i)
        return out

    unique_cH = set(find_unique_atoms(atom_rank)).intersection(cH_atoms)
    unique_cH = tuple(unique_cH)

    if not unique_cH:
        raise ValueError("No unique aromatic cH atoms found in substrate.")

    # If explicit CH positions are provided, validate and use them.
    if rpos_list is not None:
        invalid = set(rpos_list) - set(unique_cH)

        if invalid:
            raise ValueError(
                f"Invalid rpos values {sorted(invalid)} for SMILES "
                f"{substrate_smiles}. Valid unique cH positions: {unique_cH}"
            )

        unique_cH = tuple(rpos_list)

    ts_smiles = {}

    for cH in unique_cH:
        catalyst_with_h = Chem.AddHs(
            catalyst_mol,
            onlyOnAtoms=[catalyst_b_idx],
        )
        combined_rw, offset = combine_rw_mols(catalyst_with_h, substrate_rw)

        b_idx_combined = catalyst_b_idx
        n_idx_combined = catalyst_n_idx
        ch_idx_combined = cH + offset

        combined_rw.AddBond(
            b_idx_combined,
            ch_idx_combined,
            Chem.BondType.SINGLE,
        )

        boron = combined_rw.GetAtomWithIdx(b_idx_combined)
        boron.SetFormalCharge(-1)
        boron.SetNoImplicit(True)

        nitrogen = combined_rw.GetAtomWithIdx(n_idx_combined)
        nitrogen.SetFormalCharge(+1)
        nitrogen.SetNoImplicit(True)

        h_idx = combined_rw.AddAtom(Chem.Atom(1))
        combined_rw.AddBond(
            n_idx_combined,
            h_idx,
            Chem.BondType.SINGLE,
        )

        mol = combined_rw.GetMol()
        Chem.SanitizeMol(mol)

        ts_smiles[cH] = Chem.MolToSmiles(mol, canonical=True)

    return ts_smiles


def build_ts3_template_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
) -> dict[int, str]:
    catalyst_mol = Chem.MolFromSmiles(catalyst_smiles)
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)
    hbpin_mol = Chem.MolFromSmiles("CC1(C)OBOC1(C)C")

    if catalyst_mol is None:
        raise ValueError(f"Could not parse catalyst SMILES: {catalyst_smiles}")

    if substrate_mol is None:
        raise ValueError(f"Could not parse substrate SMILES: {substrate_smiles}")

    if hbpin_mol is None:
        raise ValueError("Could not parse HBpin template.")

    substrate_rw = RWMol(substrate_mol)

    b_pattern = Chem.MolFromSmarts("[B]")
    catalyst_matches = catalyst_mol.GetSubstructMatches(b_pattern)
    hbpin_matches = hbpin_mol.GetSubstructMatches(b_pattern)

    if not catalyst_matches:
        raise ValueError("No [B] atom found in the catalyst.")

    if not hbpin_matches:
        raise ValueError("No [B] atom found in HBpin.")

    catalyst_b_idx = catalyst_matches[0][0]
    hbpin_b_idx = hbpin_matches[0][0]

    ######################
    ### Find unique cH ###
    ######################
    cH_patt = Chem.MolFromSmarts('[cH]')
    matches = substrate_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]
    atom_rank = list(Chem.CanonicalRankAtoms(substrate_rw, breakTies=False))

    def find_unique_atoms(lst):
        seen = set(); out = []
        for i, x in enumerate(lst):
            if x not in seen:
                seen.add(x); out.append(i)
        return out

    unique_cH = set(find_unique_atoms(atom_rank)).intersection(cH_atoms)
    unique_cH = tuple(unique_cH)

    if not unique_cH:
        raise ValueError("No unique aromatic cH atoms found in substrate.")

    # If explicit CH positions are provided, validate and use them.
    if rpos_list is not None:
        invalid = set(rpos_list) - set(unique_cH)

        if invalid:
            raise ValueError(
                f"Invalid rpos values {sorted(invalid)} for SMILES "
                f"{substrate_smiles}. Valid unique cH positions: {unique_cH}"
            )

        unique_cH = tuple(rpos_list)

    ts_smiles = {}

    for cH in unique_cH:
        substrate_kekule = Chem.Mol(substrate_mol)
        Chem.Kekulize(substrate_kekule, clearAromaticFlags=True)

        charge_c_idx = None
        substrate_c = substrate_kekule.GetAtomWithIdx(cH)

        for bond in substrate_c.GetBonds():
            other_idx = bond.GetOtherAtomIdx(cH)
            other_atom = substrate_kekule.GetAtomWithIdx(other_idx)

            if (
                other_atom.GetAtomicNum() == 6
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                charge_c_idx = other_idx
                break

        if charge_c_idx is None:
            for atom in substrate_c.GetNeighbors():
                if atom.GetAtomicNum() == 6:
                    charge_c_idx = atom.GetIdx()
                    break

        if charge_c_idx is None:
            raise ValueError(
                f"Could not identify adjacent carbon for cation at rpos {cH}."
            )

        substrate_kekule_rw = RWMol(substrate_kekule)

        for bond in substrate_kekule_rw.GetAtomWithIdx(cH).GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            substrate_kekule_rw.GetBondBetweenAtoms(
                begin_idx,
                end_idx,
            ).SetBondType(Chem.BondType.SINGLE)

        substrate_kekule = substrate_kekule_rw.GetMol()

        combined_rw, substrate_offset = combine_rw_mols(
            catalyst_mol,
            substrate_kekule,
        )
        combined_rw, hbpin_offset = combine_rw_mols(
            combined_rw.GetMol(),
            hbpin_mol,
        )

        cat_b_idx_combined = catalyst_b_idx
        sub_c_idx_combined = cH + substrate_offset
        charge_c_idx_combined = charge_c_idx + substrate_offset
        hbpin_b_idx_combined = hbpin_b_idx + hbpin_offset

        combined_rw.AddBond(
            cat_b_idx_combined,
            sub_c_idx_combined,
            Chem.BondType.SINGLE,
        )
        combined_rw.AddBond(
            hbpin_b_idx_combined,
            sub_c_idx_combined,
            Chem.BondType.SINGLE,
        )

        bridge_h_idx = combined_rw.AddAtom(Chem.Atom(1))
        combined_rw.AddBond(
            cat_b_idx_combined,
            bridge_h_idx,
            Chem.BondType.SINGLE,
        )
        combined_rw.AddBond(
            hbpin_b_idx_combined,
            bridge_h_idx,
            Chem.BondType.SINGLE,
        )

        cat_h_idx = combined_rw.AddAtom(Chem.Atom(1))
        combined_rw.AddBond(
            cat_b_idx_combined,
            cat_h_idx,
            Chem.BondType.SINGLE,
        )

        for idx, charge in [
            (cat_b_idx_combined, -1),
            (hbpin_b_idx_combined, -1),
            (bridge_h_idx, +1),
            (charge_c_idx_combined, +1),
        ]:
            atom = combined_rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(charge)

        for idx in [
            cat_b_idx_combined,
            hbpin_b_idx_combined,
            sub_c_idx_combined,
        ]:
            atom = combined_rw.GetAtomWithIdx(idx)
            atom.SetNoImplicit(True)

        charge_c = combined_rw.GetAtomWithIdx(charge_c_idx_combined)
        charge_c.SetFormalCharge(+1)
        charge_c.SetNoImplicit(False)

        mol = combined_rw.GetMol()
        Chem.SanitizeMol(mol)

        ts_smiles[cH] = Chem.MolToSmiles(mol, canonical=True)

    return ts_smiles