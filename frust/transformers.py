from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from .tsguess.matching import match_catalyst_roles
from .utils.mols import combine_rw_mols, get_molecule_name


STANDARD_MOLECULE_STATES = (
    "dimer",
    "HH",
    "ligand",
    "catalyst",
    "int1",
    "int2",
    "HBpin-ligand",
    "HBpin-mol",
)
DIMER_VARIANT_STATES = (
    "dimer",
    "dimer_bh_bridged",
    "dimer_eight_membered",
)
OPTIONAL_DIMER_STATES = DIMER_VARIANT_STATES[1:]


def _prepare_aminoborane_monomer(
    catalyst_smiles: str,
) -> tuple[RWMol, dict[str, int | tuple[int, ...]]]:
    """Return one editable aminoborane with explicit B-H atoms."""
    mol = Chem.MolFromSmiles(catalyst_smiles)
    if mol is None:
        raise ValueError(f"Could not parse catalyst SMILES: {catalyst_smiles}")

    roles = match_catalyst_roles(mol, catalyst_name=catalyst_smiles)
    mol = Chem.AddHs(mol, onlyOnAtoms=[roles["cat_B"]])
    cat_hs = tuple(
        atom.GetIdx()
        for atom in mol.GetAtomWithIdx(roles["cat_B"]).GetNeighbors()
        if atom.GetAtomicNum() == 1
    )
    if len(cat_hs) not in {1, 2}:
        raise ValueError(
            f"Expected catalyst BH or BH2, found {len(cat_hs)} B-H hydrogens for "
            f"{catalyst_smiles}."
        )
    return RWMol(mol), {
        "cat_B": int(roles["cat_B"]),
        "cat_N": int(roles["cat_N"]),
        "cat_Hs": cat_hs,
    }


def _combine_aminoborane_monomers(
    catalyst_smiles: str,
) -> tuple[
    RWMol,
    dict[str, int | tuple[int, ...]],
    dict[str, int | tuple[int, ...]],
]:
    """Combine two prepared monomers and offset the second role map."""
    first, first_roles = _prepare_aminoborane_monomer(catalyst_smiles)
    second, second_roles = _prepare_aminoborane_monomer(catalyst_smiles)
    combined, offset = combine_rw_mols(first, second)
    second_roles = {
        "cat_B": int(second_roles["cat_B"]) + offset,
        "cat_N": int(second_roles["cat_N"]) + offset,
        "cat_Hs": tuple(
            int(atom_idx) + offset for atom_idx in second_roles["cat_Hs"]
        ),
    }
    return combined, first_roles, second_roles


def _set_formal_charges(editable: RWMol, charges: dict[int, int]) -> None:
    """Assign formal charges used to represent donor-acceptor single bonds."""
    for atom_idx, charge in charges.items():
        editable.GetAtomWithIdx(int(atom_idx)).SetFormalCharge(int(charge))


def _sanitize_dimer(editable: RWMol) -> Chem.Mol:
    """Finalize and sanitize an editable dimer graph."""
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _build_double_bh_bridged_dimer(catalyst_smiles: str) -> Chem.Mol:
    """Build the existing reciprocal B-H-B dimer graph."""
    editable, first, second = _combine_aminoborane_monomers(catalyst_smiles)
    first_h = int(first["cat_Hs"][0])
    second_h = int(second["cat_Hs"][0])
    first_b = int(first["cat_B"])
    second_b = int(second["cat_B"])

    editable.AddBond(first_h, second_b, Chem.BondType.SINGLE)
    editable.AddBond(second_h, first_b, Chem.BondType.SINGLE)
    _set_formal_charges(
        editable,
        {first_b: -1, second_b: -1, first_h: +1, second_h: +1},
    )
    return _sanitize_dimer(editable)


def _build_bh_bridged_dimer(catalyst_smiles: str) -> Chem.Mol:
    """Build the asymmetric six-membered B-H bridged dimer graph."""
    editable, first, second = _combine_aminoborane_monomers(catalyst_smiles)
    first_b = int(first["cat_B"])
    first_n = int(first["cat_N"])
    second_b = int(second["cat_B"])
    bridge_h = int(second["cat_Hs"][0])

    editable.AddBond(first_n, second_b, Chem.BondType.SINGLE)
    editable.AddBond(bridge_h, first_b, Chem.BondType.SINGLE)
    _set_formal_charges(
        editable,
        {first_n: +1, second_b: -1, bridge_h: +1, first_b: -1},
    )
    return _sanitize_dimer(editable)


def _build_eight_membered_dimer(catalyst_smiles: str) -> Chem.Mol:
    """Build the dimer graph with two reciprocal cross-monomer N-B bonds."""
    editable, first, second = _combine_aminoborane_monomers(catalyst_smiles)
    first_b = int(first["cat_B"])
    first_n = int(first["cat_N"])
    second_b = int(second["cat_B"])
    second_n = int(second["cat_N"])

    editable.AddBond(first_n, second_b, Chem.BondType.SINGLE)
    editable.AddBond(second_n, first_b, Chem.BondType.SINGLE)
    _set_formal_charges(
        editable,
        {first_n: +1, second_b: -1, second_n: +1, first_b: -1},
    )
    return _sanitize_dimer(editable)


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
        State or states to return. Accepted values are ``"dimer"``,
        ``"dimer_bh_bridged"``, ``"dimer_eight_membered"``, ``"HH"``,
        ``"ligand"``, ``"catalyst"``, ``"int1"``, ``"int2"``,
        ``"HBpin-ligand"``, and ``"HBpin-mol"``. ``"dimer"`` is the
        existing reciprocal B-H-B graph; ``"dimers"`` selects all three
        topologies. The former ``"int2"`` state is now ``"int1"``; the
        former ``"mol2"`` state is now ``"int2"``.
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
    base_names = (*STANDARD_MOLECULE_STATES, *OPTIONAL_DIMER_STATES)
    if select == "dimers":
        select = list(DIMER_VARIANT_STATES)
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

    #####################
    ### Create dimers ###
    #####################
    dimer_mol = _build_double_bh_bridged_dimer(catalyst_smiles)
    requested_optional_dimers = set(select or ()).intersection(OPTIONAL_DIMER_STATES)
    optional_dimers: dict[str, Chem.Mol] = {}
    if "dimer_bh_bridged" in requested_optional_dimers:
        optional_dimers["dimer_bh_bridged"] = _build_bh_bridged_dimer(
            catalyst_smiles
        )
    if "dimer_eight_membered" in requested_optional_dimers:
        optional_dimers["dimer_eight_membered"] = _build_eight_membered_dimer(
            catalyst_smiles
        )

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
    names.extend(optional_dimers)
    mols.extend(optional_dimers.values())

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
