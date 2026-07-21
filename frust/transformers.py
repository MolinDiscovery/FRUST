from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from .utils.mols import combine_rw_mols, get_molecule_name


def transformer_mols(
    ligand_smiles = "CCCCCCN(CCCCCC)C(=O)c1ccccc1C(F)(F)F",
    catalyst_smiles = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
    only_uniques = False,
    only_generics = False,
    show_IUPAC = True,
    select: str | list[str] | None = None,
    key_prefix: str | None = None,
    rpos_list: tuple[int, ...] | list[int] | None = None,
    return_metadata: bool = False,
):
    """Build the standard catalytic-cycle molecules.

    Parameters
    ----------
    ligand_smiles, catalyst_smiles : str, optional
        Substrate and catalyst SMILES used to construct the states.
    only_uniques, only_generics : bool, optional
        Legacy selection flags. Prefer ``select`` in new code.
    show_IUPAC : bool, optional
        Resolve an IUPAC substrate name for generated keys when ``True``.
    select : str or list of str or None, optional
        State or states to return. Accepted values are ``"dimer"``, ``"HH"``,
        ``"ligand"``, ``"catalyst"``, ``"int1"``, ``"int2"``,
        ``"HBpin-ligand"``, and ``"HBpin-mol"``. The former ``"int2"`` state
        is now ``"int1"``; the former ``"mol2"`` state is now ``"int2"``.
    key_prefix : str or None, optional
        Stable prefix for generated structure names.
    rpos_list : tuple of int, list of int, or None, optional
        Reactive positions to construct. If omitted, symmetry-unique aromatic
        C-H positions are detected.
    return_metadata : bool, optional
        Return each molecule together with its structured metadata.

    Returns
    -------
    dict
        Generated names mapped to RDKit molecules, or to ``(molecule,
        metadata)`` tuples when ``return_metadata=True``.
    """

    # --- normalize select to a list if given ---
    base_names = (
        "dimer",
        "HH",
        "ligand",
        "catalyst",
        "int1",
        "int2",
        "HBpin-ligand",
        "HBpin-mol",
    )
    if select is not None:
        if isinstance(select, str):
            select = [select]
        bad = set(select) - set(base_names)
        if bad:
            migration = (
                " 'mol2' was renamed to 'int2', and the former 'int2' was "
                "renamed to 'int1'."
                if "mol2" in bad
                else ""
            )
            raise ValueError(f"select must be from {base_names}, got {bad}.{migration}")

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
    unique_cH = tuple(unique_cH)

    # If explicit CH positions are provided, validate and use them.
    if rpos_list is not None:
        from frust.utils.mols import find_ch

        valid_positions = find_ch(ligand_smiles)
        invalid = set(rpos_list) - set(valid_positions)

        if invalid:
            raise ValueError(
                f"Invalid rpos values {sorted(invalid)} for SMILES {ligand_smiles}. "
                f"Valid cH positions: {valid_positions}"
            )

        unique_cH = tuple(rpos_list)

    ############################################
    ### Create intermediates 1 and 2 ###
    ############################################
    b_pattern = Chem.MolFromSmarts("[B]")

    catalyst_matches = catalyst_mol.GetSubstructMatches(b_pattern)

    if not catalyst_matches:
        raise ValueError("No [B] atom found in the catalyst.")

    catalyst_b_idx = catalyst_matches[0][0]
    from frust.tsguess.matching import match_catalyst_roles

    catalyst_roles = match_catalyst_roles(
        catalyst_mol,
        catalyst_name=catalyst_smiles,
    )
    catalyst_n_idx = catalyst_roles["cat_N"]

    int1s = []
    int2s = []
    for cH in unique_cH:

        combined_rw, offset = combine_rw_mols(catalyst_rw, ligand_rw)
        combined_mol = combined_rw.GetMol()
        Chem.SanitizeMol(combined_mol)
        
        b_idx_combined   = catalyst_b_idx
        ch_idx_combined  = cH + offset

        combined_rw.AddBond(b_idx_combined, ch_idx_combined, Chem.BondType.SINGLE)

        int2 = combined_rw.GetMol()

        boron = combined_rw.GetAtomWithIdx(catalyst_b_idx)
        boron.SetFormalCharge(-1)

        nitrogen = combined_rw.GetAtomWithIdx(catalyst_n_idx)
        nitrogen.SetFormalCharge(+1)

        int1 = combined_rw.GetMol()
        Chem.SanitizeMol(int2)
        Chem.SanitizeMol(int1)

        int1s.append((int1, ch_idx_combined))
        int2s.append((int2, ch_idx_combined))

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
    names = ['dimer','HH','ligand','catalyst','int1','int2','HBpin-ligand','HBpin-mol']
    if show_IUPAC:
        names[2] = get_molecule_name(ligand_smiles)
    mols = [
        dimer_mol,
        HH_mol,
        ligand_mol,
        catalyst_mol,
        int1s,
        int2s,
        HBpin_ligands,
        HBpin_mol
    ]

    mols_dict: dict[str, Chem.Mol] = {}
    metadata_dict: dict[str, dict] = {}
    iupac_substrate_name = names[2]
    unique_names = {names[2], "int1", "int2", "HBpin-ligand"}
    generic_names = {"dimer", "HH", "catalyst", "HBpin-mol"}

    def _add_entry(key: str, mol: Chem.Mol, role: str, rpos: int | None = None) -> None:
        mols_dict[key] = mol
        metadata_dict[key] = {
            "custom_name": key,
            "substrate_name": iupac_substrate_name,
            "input_smiles": ligand_smiles,
            "smiles": ligand_smiles,
            "structure_type": "MOL",
            "molecule_role": role,
            "rpos": rpos,
            "structure_id": (
                f"MOL:{iupac_substrate_name}:{role}:r{rpos}"
                if rpos is not None
                else f"MOL:{iupac_substrate_name}:{role}"
            ),
        }

    for name, mol in zip(names, mols):
        if only_uniques and name not in unique_names:
            continue
        if only_generics and name not in generic_names:
            continue
        if isinstance(mol, list):
            for m, i in mol:
                _add_entry(f"{name}_rpos({i})", m, name, i)
        else:
            role = "ligand" if name == names[2] else name
            _add_entry(name, mol, role)

    # --- apply select filter if requested ---
    if select is not None:
        filtered: dict[str, Chem.Mol] = {}
        for choice in select:
            actual = choice
            if choice == "ligand" and show_IUPAC:
                actual = iupac_substrate_name
            for key, m in mols_dict.items():
                if key == actual or key.startswith(f"{actual}_rpos"):
                    filtered[key] = m
        mols_dict = filtered

    if key_prefix is None:
        key_prefix = ligand_smiles
    if key_prefix:
        prefixed_mols: dict[str, Chem.Mol] = {}
        prefixed_metadata: dict[str, dict] = {}
        for key, mol in mols_dict.items():
            prefixed_key = f"{key_prefix}_{key}"
            prefixed_mols[prefixed_key] = mol
            meta = dict(metadata_dict[key])
            meta["custom_name"] = prefixed_key
            prefixed_metadata[prefixed_key] = meta
        mols_dict = prefixed_mols
        metadata_dict = prefixed_metadata

    if return_metadata:
        return {key: (mol, metadata_dict[key]) for key, mol in mols_dict.items()}

    return mols_dict
