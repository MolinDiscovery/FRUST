"""Connected-SMILES builders for the tsguess2 backend."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import RWMol

from frust.tsguess.matching import match_catalyst_roles, parse_rpos_value


BISARYL_ROLE_MAPS = {
    "cat_B": 1,
    "cat_N": 2,
    "N_transfer_H": 3,
    "B_transfer_H": 4,
    "bound_C": 5,
    "incoming_C": 6,
    "scaffold_B_C": 7,
    "scaffold_N_C": 8,
    "bound_frame": 9,
    "incoming_frame": 10,
}


BISARYL_EXCHANGE_ROLE_MAPS = {
    "cat_B": 1, "cat_N": 2, "exchanging_C": 5, "retained_C": 6,
    "scaffold_B_C": 7, "scaffold_N_C": 8, "exchanging_frame": 9,
    "retained_frame": 10, "pin_B": 11, "transfer_H": 12,
    "pin_O1": 13, "pin_O2": 14, "pin_frame": 15,
}


def build_bisaryl_adduct_connected_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
    *,
    bound_substrate_smiles: str | None = None,
    bound_rpos: int | None = None,
) -> dict[int, str]:
    """Build the bisaryl N-H/B-H adduct for second activation or H2 release.

    Parameters
    ----------
    catalyst_smiles : str
        Neutral BH2 catalyst containing the B-aryl-tertiary-N scaffold.
    substrate_smiles : str
        Incoming aromatic C-H substrate.
    rpos_list : tuple of int, list of int, or None, optional
        Incoming reactive heavy-atom indices. ``None`` selects symmetry-unique
        aromatic C-H positions.
    bound_substrate_smiles : str or None, optional
        First substrate-derived aryl identity. ``None`` uses a separate copy of
        ``substrate_smiles``; it never reuses the incoming molecule's atoms.
    bound_rpos : int or None, optional
        Reactive site of the first substrate. ``None`` uses the corresponding
        incoming index only when the two substrate SMILES are identical.

    Returns
    -------
    dict of int to str
        Hydrogen-explicit mapped adduct SMILES keyed by incoming site. Stable
        atom-map annotations distinguish catalyst, retained aryl, incoming
        aryl and the two eliminating H atoms through a SMILES round trip.

    Notes
    -----
    The first substrate activation's H2 is outside this local reacting complex.
    The second substrate's site H moves to N; the remaining catalyst hydride
    stays on B. The adduct has local N+/B- charges and zero net charge. This
    molecular graph is a construction scaffold for ``TS5`` (second C-H
    activation) or ``TS6`` (H2 elimination), not a validated saddle. Their
    separate geometry specifications define the changing distances.
    """
    catalyst = Chem.MolFromSmiles(catalyst_smiles)
    incoming = Chem.MolFromSmiles(substrate_smiles)
    bound_text = substrate_smiles if bound_substrate_smiles is None else bound_substrate_smiles
    bound = Chem.MolFromSmiles(bound_text)
    if catalyst is None or incoming is None or bound is None:
        raise ValueError("Invalid catalyst, bound substrate or incoming substrate SMILES")
    catalyst_roles = match_catalyst_roles(catalyst, catalyst_name=catalyst_smiles)
    b, n = catalyst_roles["cat_B"], catalyst_roles["cat_N"]
    if catalyst.GetAtomWithIdx(b).GetTotalNumHs(includeNeighbors=True) != 2:
        raise ValueError("Bisaryl H2 construction requires a BH2 parent catalyst")
    sites = parse_rpos_value(rpos_list, substrate_smiles)
    if bound_rpos is None and bound_text != substrate_smiles:
        raise ValueError("bound_rpos is required when bound and incoming substrates differ")
    scaffold = catalyst.GetSubstructMatch(Chem.MolFromSmarts("[#5]~c1ccccc1~[#7]"))
    results = {}
    for incoming_site in sites:
        bound_site = incoming_site if bound_rpos is None else int(bound_rpos)
        parse_rpos_value([bound_site], bound_text)
        cat_h = Chem.AddHs(catalyst)
        bound_h = Chem.AddHs(bound)
        incoming_h = Chem.AddHs(incoming)
        offset_bound = cat_h.GetNumAtoms()
        offset_incoming = offset_bound + bound_h.GetNumAtoms()
        rw = RWMol(Chem.CombineMols(Chem.CombineMols(cat_h, bound_h), incoming_h))
        for atom in rw.GetAtoms():
            atom.SetAtomMapNum(0)

        def hydrogen(mol, atom):
            return next(a.GetIdx() for a in mol.GetAtomWithIdx(atom).GetNeighbors()
                        if a.GetAtomicNum() == 1)

        def frame(mol, atom):
            ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
            return min((a.GetIdx() for a in mol.GetAtomWithIdx(atom).GetNeighbors()
                        if a.GetAtomicNum() > 1),
                       key=lambda i: (-mol.GetAtomWithIdx(i).GetAtomicNum(), ranks[i], i))

        hydrides = [a.GetIdx() for a in cat_h.GetAtomWithIdx(b).GetNeighbors() if a.GetAtomicNum() == 1]
        first_h = hydrogen(bound_h, bound_site) + offset_bound
        second_h = hydrogen(incoming_h, incoming_site) + offset_incoming
        first_c = bound_site + offset_bound
        second_c = incoming_site + offset_incoming
        roles = {
            "cat_B": b, "cat_N": n, "N_transfer_H": second_h,
            "B_transfer_H": hydrides[1], "bound_C": first_c,
            "incoming_C": second_c, "scaffold_B_C": scaffold[1],
            "scaffold_N_C": scaffold[-2],
            "bound_frame": frame(bound_h, bound_site) + offset_bound,
            "incoming_frame": frame(incoming_h, incoming_site) + offset_incoming,
        }
        for role, index in roles.items():
            rw.GetAtomWithIdx(index).SetAtomMapNum(BISARYL_ROLE_MAPS[role])
        rw.RemoveBond(first_c, first_h)
        rw.RemoveBond(second_c, second_h)
        rw.AddBond(b, first_c, Chem.BondType.SINGLE)
        rw.AddBond(b, second_c, Chem.BondType.SINGLE)
        rw.AddBond(n, second_h, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(b).SetFormalCharge(-1)
        rw.GetAtomWithIdx(n).SetFormalCharge(1)
        for index in [b, n, first_c, second_c]:
            rw.GetAtomWithIdx(index).SetNoImplicit(True)
        for index in sorted([first_h, hydrides[0]], reverse=True):
            rw.RemoveAtom(index)
        result = rw.GetMol()
        Chem.SanitizeMol(result)
        results[int(incoming_site)] = Chem.MolToSmiles(result, canonical=True)
    return results


build_bisaryl_h2_connected_smiles = build_bisaryl_adduct_connected_smiles


def build_bisaryl_diaryl_connected_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
    *,
    bound_substrate_smiles: str | None = None,
    bound_rpos: int | None = None,
) -> dict[int, str]:
    """Build the neutral diaryl catalyst after bisaryl H2 elimination.

    Parameters
    ----------
    catalyst_smiles : str
        Neutral parent BH2 catalyst with the B-aryl-tertiary-N scaffold.
    substrate_smiles : str
        Second substrate, whose reactive carbon becomes bonded to boron.
    rpos_list : tuple of int, list of int, or None, optional
        Second-substrate reactive heavy-atom indices. None selects unique sites.
    bound_substrate_smiles : str or None, optional
        First substrate identity. None uses the second substrate identity.
    bound_rpos : int or None, optional
        First substrate site. Required when its identity differs.

    Returns
    -------
    dict of int to str
        Connected mapped diaryl graphs keyed by second-substrate site. Both
        aryl groups remain on boron; the N-H and B-H pair is absent. Catalyst
        B and N are neutral, and there are no free H2 or HBpin fragments.
    """
    adducts = build_bisaryl_adduct_connected_smiles(
        catalyst_smiles, substrate_smiles, rpos_list,
        bound_substrate_smiles=bound_substrate_smiles, bound_rpos=bound_rpos,
    )
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    results = {}
    for site, text in adducts.items():
        rw = RWMol(Chem.MolFromSmiles(text, parser))
        remove = [atom.GetIdx() for atom in rw.GetAtoms()
                  if atom.GetAtomMapNum() in {BISARYL_ROLE_MAPS["N_transfer_H"],
                                             BISARYL_ROLE_MAPS["B_transfer_H"]}]
        for index in sorted(remove, reverse=True):
            rw.RemoveAtom(index)
        mapped = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw.GetAtoms()
                  if atom.GetAtomMapNum()}
        for role in ("cat_B", "cat_N"):
            rw.GetAtomWithIdx(mapped[BISARYL_ROLE_MAPS[role]]).SetFormalCharge(0)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        results[site] = Chem.MolToSmiles(mol, canonical=True)
    return results


def build_bisaryl_exchange_connected_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
    *,
    bound_substrate_smiles: str | None = None,
    bound_rpos: int | None = None,
) -> dict[int, str]:
    """Build the diaryl/HBpin bridge graph for exchange-path construction.

    Parameters
    ----------
    catalyst_smiles : str
        Neutral parent BH2 catalyst containing the B-aryl-tertiary-N scaffold.
    substrate_smiles : str
        Second substrate-derived aryl identity, retained on catalyst B.
    rpos_list : tuple of int, list of int, or None, optional
        Reactive heavy-atom sites of the second substrate. ``None`` selects
        symmetry-unique aromatic C-H sites.
    bound_substrate_smiles : str or None, optional
        First substrate identity, whose aryl group is assigned the exchanging
        role. ``None`` uses a separate copy of ``substrate_smiles``.
    bound_rpos : int or None, optional
        First substrate site. Required for a different first substrate identity.

    Returns
    -------
    dict of int to str
        Connected, explicit-H mapped graph keyed by second-substrate site.
        The inventory is diaryl catalyst plus one HBpin, with both previous
        activation H2 molecules outside the local complex.

    Notes
    -----
    Local B-/B-/H+/C+ charges permit a sanitized covalent representation of
    the bridge, following the TS3/TS4 construction convention. Net charge is
    zero. These display/embedding bonds and formal charges do not specify a
    validated multicenter electronic structure or optimizer constraints.
    Reaction-specific role coordinates and ``constraint_spec`` control the
    geometry; this graph alone cannot identify TS7, INT6 or TS8.
    """
    adducts = build_bisaryl_adduct_connected_smiles(
        catalyst_smiles, substrate_smiles, rpos_list,
        bound_substrate_smiles=bound_substrate_smiles, bound_rpos=bound_rpos,
    )
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    results = {}
    for site, text in adducts.items():
        adduct = Chem.MolFromSmiles(text, parser)
        rw = RWMol(adduct)
        remove = [a.GetIdx() for a in rw.GetAtoms()
                  if a.GetAtomMapNum() in {BISARYL_ROLE_MAPS["N_transfer_H"],
                                           BISARYL_ROLE_MAPS["B_transfer_H"]}]
        for index in sorted(remove, reverse=True):
            rw.RemoveAtom(index)
        mapped = {a.GetAtomMapNum(): a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum()}
        cat_b, cat_n = mapped[BISARYL_ROLE_MAPS["cat_B"]], mapped[BISARYL_ROLE_MAPS["cat_N"]]
        exchange_c = mapped[BISARYL_ROLE_MAPS["bound_C"]]
        rw.GetAtomWithIdx(cat_b).SetFormalCharge(0)
        rw.GetAtomWithIdx(cat_n).SetFormalCharge(0)
        diaryl = rw.GetMol()
        Chem.SanitizeMol(diaryl)
        Chem.Kekulize(diaryl, clearAromaticFlags=True)
        rw = RWMol(diaryl)
        carbon_neighbors = [a.GetIdx() for a in rw.GetAtomWithIdx(exchange_c).GetNeighbors()
                            if a.GetAtomicNum() == 6]
        if not carbon_neighbors:
            raise ValueError("Bisaryl exchange graph requires a neighboring carbon at the exchanging aryl site")
        charged_c = next((i for i in carbon_neighbors
                          if rw.GetBondBetweenAtoms(exchange_c, i).GetBondType() == Chem.BondType.DOUBLE),
                         carbon_neighbors[0])
        for bond in rw.GetAtomWithIdx(exchange_c).GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(charged_c).SetFormalCharge(1)
        rw.GetAtomWithIdx(charged_c).SetNoImplicit(True)

        hbpin = Chem.AddHs(Chem.MolFromSmiles("CC1(C)OBOC1(C)C"))
        for atom in hbpin.GetAtoms():
            atom.SetAtomMapNum(0)
        pin_b = next(a.GetIdx() for a in hbpin.GetAtoms() if a.GetAtomicNum() == 5)
        pin_h = next(a.GetIdx() for a in hbpin.GetAtomWithIdx(pin_b).GetNeighbors()
                     if a.GetAtomicNum() == 1)
        oxygens = sorted(a.GetIdx() for a in hbpin.GetAtomWithIdx(pin_b).GetNeighbors()
                         if a.GetAtomicNum() == 8)
        pin_frame = next(a.GetIdx() for a in hbpin.GetAtomWithIdx(oxygens[0]).GetNeighbors()
                         if a.GetAtomicNum() == 6)
        offset = rw.GetNumAtoms()
        rw = RWMol(Chem.CombineMols(rw.GetMol(), hbpin))
        pin_b, pin_h = pin_b + offset, pin_h + offset
        rw.AddBond(cat_b, pin_h, Chem.BondType.SINGLE)
        rw.AddBond(pin_b, exchange_c, Chem.BondType.SINGLE)
        for index, charge in [(cat_b, -1), (pin_b, -1), (pin_h, 1)]:
            rw.GetAtomWithIdx(index).SetFormalCharge(charge)
            rw.GetAtomWithIdx(index).SetNoImplicit(True)
        for role, index in {"pin_B": pin_b, "transfer_H": pin_h,
                            "pin_O1": oxygens[0] + offset, "pin_O2": oxygens[1] + offset,
                            "pin_frame": pin_frame + offset}.items():
            rw.GetAtomWithIdx(index).SetAtomMapNum(BISARYL_EXCHANGE_ROLE_MAPS[role])
        result = rw.GetMol()
        Chem.SanitizeMol(result)
        results[site] = Chem.MolToSmiles(result, canonical=True)
    return results


def combine_rw_mols(mol1: Chem.Mol, mol2: Chem.Mol) -> tuple[RWMol, int]:
    """Combine two molecules and return an editable molecule plus offset.

    Parameters
    ----------
    mol1, mol2 : rdkit.Chem.Mol
        Molecules to combine in order.

    Returns
    -------
    tuple
        ``(combined_rw, offset)`` where ``offset`` is the atom offset for
        ``mol2`` in the combined molecule.
    """
    combined = Chem.CombineMols(mol1, mol2)
    offset = mol1.GetNumAtoms()

    return RWMol(combined), offset


def build_ts1_ts2_connected_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
) -> dict[int, str]:
    """Build connected TS1/TS2 graph SMILES keyed by substrate ``rpos``.

    Parameters
    ----------
    catalyst_smiles : str
        Catalyst SMILES containing a boron atom and dimethyl amine motif.
    substrate_smiles : str
        Substrate SMILES containing aromatic C-H positions.
    rpos_list : tuple of int, list of int, or None, optional
        Explicit reactive-position atom indices. When omitted, symmetry-unique
        aromatic C-H positions are used.

    Returns
    -------
    dict
        Mapping from ``rpos`` to generated connected TS SMILES.
    """
    catalyst_mol = Chem.MolFromSmiles(catalyst_smiles)
    substrate_mol = Chem.MolFromSmiles(substrate_smiles)

    if catalyst_mol is None:
        raise ValueError(f"Could not parse catalyst SMILES: {catalyst_smiles}")

    if substrate_mol is None:
        raise ValueError(f"Could not parse substrate SMILES: {substrate_smiles}")

    substrate_rw = RWMol(substrate_mol)

    catalyst_roles = match_catalyst_roles(
        catalyst_mol,
        catalyst_name=catalyst_smiles,
    )
    catalyst_b_idx = catalyst_roles["cat_B"]
    catalyst_n_idx = catalyst_roles["cat_N"]

    cH_patt = Chem.MolFromSmarts("[cH]")
    matches = substrate_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]
    atom_rank = list(Chem.CanonicalRankAtoms(substrate_rw, breakTies=False))

    def find_unique_atoms(lst):
        seen = set()
        out = []
        for i, x in enumerate(lst):
            if x not in seen:
                seen.add(x)
                out.append(i)
        return out

    unique_cH = set(find_unique_atoms(atom_rank)).intersection(cH_atoms)
    unique_cH = tuple(unique_cH)

    if not unique_cH:
        raise ValueError("No unique aromatic cH atoms found in substrate.")

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


def build_ts3_ts4_connected_smiles(
    catalyst_smiles: str,
    substrate_smiles: str,
    rpos_list: tuple[int, ...] | list[int] | None = None,
) -> dict[int, str]:
    """Build connected TS3/TS4 HBpin graph SMILES keyed by ``rpos``.

    Parameters
    ----------
    catalyst_smiles : str
        Catalyst SMILES containing one boron atom.
    substrate_smiles : str
        Substrate SMILES containing aromatic C-H positions.
    rpos_list : tuple of int, list of int, or None, optional
        Explicit reactive-position atom indices. When omitted, symmetry-unique
        aromatic C-H positions are used.

    Returns
    -------
    dict
        Mapping from ``rpos`` to generated connected TS SMILES.
    """
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
    catalyst_b_h_count = int(
        catalyst_mol.GetAtomWithIdx(catalyst_b_idx).GetTotalNumHs()
    )
    if catalyst_b_h_count not in {1, 2}:
        raise ValueError(
            "TS3/TS4 construction requires a BH or BH2 catalyst; "
            f"found {catalyst_b_h_count} B-H hydrogens for {catalyst_smiles}."
        )

    cH_patt = Chem.MolFromSmarts("[cH]")
    matches = substrate_rw.GetSubstructMatches(cH_patt)
    cH_atoms = [ind[0] for ind in matches]
    atom_rank = list(Chem.CanonicalRankAtoms(substrate_rw, breakTies=False))

    def find_unique_atoms(lst):
        seen = set()
        out = []
        for i, x in enumerate(lst):
            if x not in seen:
                seen.add(x)
                out.append(i)
        return out

    unique_cH = set(find_unique_atoms(atom_rank)).intersection(cH_atoms)
    unique_cH = tuple(unique_cH)

    if not unique_cH:
        raise ValueError("No unique aromatic cH atoms found in substrate.")

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

        if catalyst_b_h_count == 2:
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
