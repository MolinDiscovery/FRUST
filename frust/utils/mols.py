# frust/utils/mols.py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem, rdDetermineBonds
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import Mol
from frust.utils.io import read_ts_type_from_xyz


def canonicalize_smiles(smiles: str) -> str:
    """Return a canonical isomeric SMILES string when RDKit can parse it.

    Parameters
    ----------
    smiles : str
        Input SMILES string.

    Returns
    -------
    str
        RDKit canonical isomeric SMILES, or the stripped input string if RDKit
        cannot parse the molecule.
    """
    value = str(smiles).strip()
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return value
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def sanitize_molecule_name(name: str) -> str:
    """Return a compact FRUST-safe molecule name.

    Parameters
    ----------
    name : str
        Human-readable molecule name.

    Returns
    -------
    str
        Name with whitespace and path separators replaced by underscores.
    """
    import re

    value = re.sub(r"\s+", "_", str(name).strip())
    value = value.replace("/", "_").replace("\\", "_")
    return value or "Unknown_Molecule"


def lookup_pubchem_name(
    smiles: str,
    *,
    max_retries: int = 5,
    delay: float = 10.0,
) -> dict[str, object]:
    """Look up PubChem IUPAC name metadata for one SMILES string.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    max_retries : int, optional
        Number of attempts for temporary PubChem server-busy errors.
    delay : float, optional
        Seconds to wait between server-busy retry attempts.

    Returns
    -------
    dict
        Lookup metadata with ``canonical_smiles``, ``pubchem_iupac``,
        ``pubchem_cid``, ``lookup_status``, and ``lookup_error`` keys.
    """
    import time

    import pubchempy as pcp

    canonical = canonicalize_smiles(smiles)
    result: dict[str, object] = {
        "input_smiles": str(smiles),
        "canonical_smiles": canonical,
        "pubchem_iupac": None,
        "pubchem_cid": None,
        "lookup_status": "not_queried",
        "lookup_error": None,
    }

    compounds = None
    for attempt in range(1, max_retries + 1):
        try:
            compounds = pcp.get_compounds(smiles, "smiles")
            break
        except pcp.PubChemHTTPError as exc:
            if "PUGREST.ServerBusy" in str(exc) and attempt < max_retries:
                time.sleep(delay)
                continue
            result["lookup_status"] = "error"
            result["lookup_error"] = str(exc)
            return result
        except Exception as exc:
            result["lookup_status"] = "error"
            result["lookup_error"] = str(exc)
            return result

    if not compounds:
        result["lookup_status"] = "not_found"
        return result

    compound = compounds[0]
    result["pubchem_cid"] = getattr(compound, "cid", None)
    iupac_name = getattr(compound, "iupac_name", None)
    if not iupac_name:
        result["lookup_status"] = "no_iupac"
        return result

    result["pubchem_iupac"] = str(iupac_name)
    result["lookup_status"] = "success"
    return result


def get_molecule_name(smiles: str):
    """Retrieve the IUPAC name for a molecule from its SMILES string.
    
    Queries the PubChem database to get the IUPAC name for a given molecule.
    Implements retry logic to handle server busy errors. Returns a sanitized
    name with spaces replaced by underscores for file naming compatibility.
    
    Args:
        smiles (str): The SMILES representation of the molecule.
        
    Returns:
        str: The IUPAC name with spaces replaced by underscores, or 
             "Unknown_Molecule" if retrieval fails or no name is found.
             
    Raises:
        pcp.PubChemHTTPError: When PubChem API encounters errors other than 
                             server busy status (which is handled with retries).
    """
    result = lookup_pubchem_name(smiles)
    name = result.get("pubchem_iupac")
    if name:
        return sanitize_molecule_name(str(name))

    print("Warning: Failed to retrieve compound name. Using fallback name.")
    return "Unknown_Molecule"
    

def generate_id(id_name: str, job_id:int = None) -> str:
    """Generate a unique identifier string with timestamp and hash.
    
    Creates a unique ID by combining a base name, optional job ID, current 
    timestamp, and a short UUID hash. The format ensures chronological 
    ordering while maintaining uniqueness.
    
    Args:
        id_name (str): Base name for the identifier.
        job_id (int, optional): Optional job identifier to include in the ID.
                               If provided, will be formatted as "job{job_id}".
                               Defaults to None.
        
    Returns:
        str: Formatted unique identifier string. Format is either:
             "{id_name}-job{job_id}-{YYMMDD-HHMMSS}-{hash}" if job_id provided,
             or "{id_name}-{YYMMDD-HHMMSS}-{hash}" if job_id is None.
             
    Example:
        >>> generate_id("analysis", 123)
        'analysis-job123-250527-142530-a4'
        >>> generate_id("calculation")
        'calculation-250527-142530-b7'
    """
    import datetime
    import uuid
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%y%m%d-%H%M%S')  # YYMMDD-HHMMSS
    # Create a UUID and take the first 2 characters for a shorter hash
    uuid_hash = str(uuid.uuid4())[:2]
    if job_id:
        id = f"{id_name}-job{job_id}-{formatted_time}-{uuid_hash}"
    else:
        id = f"{id_name}-{formatted_time}-{uuid_hash}"
    return id


def get_rpos_from_name(ts_name):
    import re
    match = re.search(r"rpos\((\d+)\)", ts_name)
    if match:
        return int(match.group(1))
    else:
        print("No rpos found in name!")


def rotated_maps(lig, old, reactive_old_idx, new_targets):
    """Return a list of rotated atom‑maps."""
    from collections import deque
    lig = list(lig)
    old = list(old)
    pos_old = old.index(reactive_old_idx)          # where 41 sits in old list

    maps = []
    for new_atom in new_targets:
        pos_new = lig.index(new_atom)              # where 4 / 5 / 6 sits now
        shift   = pos_old - pos_new                # how far to rotate right
        d = deque(lig)
        d.rotate(shift)                            # positive = rotate right
        maps.append(list(zip(d, old)))
    return maps


def combine_rw_mols(rw1, rw2):
    """Merge two RWMols, returning (combined_rwmol, offset_for_rw2)."""
    from rdkit.Chem.rdchem import RWMol
    combined = RWMol(rw1)
    old_to_new = {}
    offset = combined.GetNumAtoms()
    
    # Add atoms from rw2
    for a_idx in range(rw2.GetNumAtoms()):
        new_idx = combined.AddAtom(rw2.GetAtomWithIdx(a_idx))
        old_to_new[a_idx] = new_idx
    
    # Add bonds from rw2
    for b_idx in range(rw2.GetNumBonds()):
        bond = rw2.GetBondWithIdx(b_idx)
        a1 = old_to_new[bond.GetBeginAtomIdx()]
        a2 = old_to_new[bond.GetEndAtomIdx()]
        combined.AddBond(a1, a2, bond.GetBondType())
    
    return combined, offset


def remove_one_h(rwmol, atom_idx):
    """Removes a single hydrogen from the specified atom_idx if present."""
    atom = rwmol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:  # hydrogen
            rwmol.RemoveBond(atom.GetIdx(), nbr.GetIdx())
            rwmol.RemoveAtom(nbr.GetIdx())
            # print(f"Removed an H from atom {atom_idx}")
            return
    print(f"No hydrogen found to remove on atom {atom_idx}")


def fix_pin_frag(frag: Chem.Mol) -> Chem.Mol:
    """
    • converts B=O double bonds → single, neutralises B/O  
    • adds a B–H 1.18 Å away from B but does **not** move any other atom
    """
    BH_LEN = 1.18   # Å – average B–H bond length

    def _unit(vec):
        """return unit vector (or None if zero length)"""
        import math
        norm = math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
        if norm < 1e-6:
            return None
        return vec.__class__(vec.x / norm, vec.y / norm, vec.z / norm)

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

def fix_cat_frag(mol: Chem.Mol, bh_len: float = 1.19) -> Chem.Mol:
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


def create_ts_per_rpos_old(
    ligand_smiles_list: list[str],
    ts_guess_xyz: str,
    ) -> list[dict[str, Mol]]:
    """
    Generate transition state (TS) structures for each ligand SMILES using a TS guess XYZ template.

    Args:
        ligand_smiles_list (List[str]): List of ligand SMILES strings for which to create TS structures.
        ts_guess_xyz (str): Path to an XYZ file containing the TS guess geometry. The TS type
            (e.g., 'TS1', 'TS2', 'TS3', 'TS4') is inferred from the comment line.

    Returns:
        List[Dict[str, rdkit.Chem.Mol]]: A list of dictionaries, each mapping a TS identifier
            (e.g., reaction position key) to an RDKit Mol object representing the generated TS.
    """

    ts_type = read_ts_type_from_xyz(ts_guess_xyz)

    if ts_type == 'TS1':
        from frust.transformers import transformer_ts1
        transformer_ts = transformer_ts1
    elif ts_type == 'TS2':
        from frust.transformers import transformer_ts2
        transformer_ts = transformer_ts2
    elif ts_type == 'TS3':
        from frust.transformers import transformer_ts3
        transformer_ts = transformer_ts3
    elif ts_type == 'TS4':
        from frust.transformers import transformer_ts4
        transformer_ts = transformer_ts4
    elif ts_type == 'INT3':
        from frust.transformers import transformer_int3
        transformer_ts = transformer_int3        
    else:
        raise ValueError(f"Unrecognized TS type: {ts_type}")

    ts_structs = {}
    for smi in ligand_smiles_list:
        ts_mols = transformer_ts(smi, ts_guess_xyz)
        ts_structs.update(ts_mols)

    ts_structs_list = []
    for k, i in ts_structs.items():
        ts_structs_list.append({k:i})   

    return ts_structs_list


def _extract_rpos_from_df(df):
    rpos_list = []
    for _, row in df.iterrows():

        rpos_in  = str(row["rpos"])
        smi      = row["smiles"]
        rpos_out = None

        if rpos_in is None:
            rpos_out = find_unique_ch(smi)
        if pd.isna(rpos_in):
            rpos_out = find_unique_ch(smi)
        elif ";" in rpos_in:
            l_str = rpos_in.split(";")
            try:
                rpos_out = tuple(int(i) for i in l_str)
            except ValueError as e:
                raise ValueError("rpos must be given in the format of 2;3") from e
        else:
            try:
                rpos_out = (int(rpos_in), )
            except (ValueError, TypeError) as e:
                raise ValueError("rpos error: rpos format must be either a single integer or a series of \n" \
                "integers seperated by ;") from e

        for ch in rpos_out:
            valid_ch = find_ch(smi)

            if ch not in valid_ch:
                try:
                    from IPython.display import display
                    from frust.vis import DrawUniqueChGrid
                    display(DrawUniqueChGrid(smi))
                except ImportError:
                    pass
                raise ValueError(
                    f"Invalid CH index {rpos_out} for SMILES: {smi}\n"
                    f"Valid aromatic CH positions: {valid_ch}"
                )
            
        rpos_list.append(rpos_out)
            
    return rpos_list


def create_ts_per_rpos(
    ligand_smiles_df: pd.DataFrame,
    ts_guess_xyz: str,
    return_format: str = "list",
    ) -> list[dict[str, Mol]]:
    """Generate TS structures from a dataframe of SMILES and optional reactive positions.

    Parameters
    ----------
    ligand_smiles_df : pandas.DataFrame
        Dataframe containing a ``smiles`` column. If present, the ``rpos``
        column specifies one or more aromatic C-H positions per row. Multiple
        positions may be given as a semicolon-separated string such as
        ``"2;3"``.
    ts_guess_xyz : str
        Path to a TS-guess XYZ file. The TS type is inferred from the comment
        line in the file.
    return_format : str, optional
        Output format. Use ``"list"`` to return a list of single-item
        dictionaries, or ``"dict"`` to return one merged dictionary.

    Returns
    -------
    list[dict[str, rdkit.Chem.Mol]] or dict[str, rdkit.Chem.Mol]
        Generated TS structures keyed by their TS identifiers. Duplicate
        SMILES entries in ``ligand_smiles_df`` are ignored after the first
        occurrence.

    Notes
    -----
    The ``rpos`` values are validated against the aromatic C-H positions for
    each SMILES string before the corresponding transformer is called.
    """

    ligand_smiles_list = list(dict.fromkeys(ligand_smiles_df["smiles"]))

    rpos_list = None
    if "rpos" in ligand_smiles_df.columns:
        rpos_list = _extract_rpos_from_df(ligand_smiles_df)

    ts_type = read_ts_type_from_xyz(ts_guess_xyz)

    if ts_type == 'TS1':
        from frust.transformers import transformer_ts1
        transformer_ts = transformer_ts1
    elif ts_type == 'TS2':
        from frust.transformers import transformer_ts2
        transformer_ts = transformer_ts2
    elif ts_type == 'TS3':
        from frust.transformers import transformer_ts3
        transformer_ts = transformer_ts3
    elif ts_type == 'TS4':
        from frust.transformers import transformer_ts4
        transformer_ts = transformer_ts4
    elif ts_type == 'INT3':
        from frust.transformers import transformer_int3
        transformer_ts = transformer_int3        
    else:
        raise ValueError(f"Unrecognized TS type: {ts_type}")

    ts_structs = {}
    if rpos_list is None:
        for smi in ligand_smiles_list:
            ts_mols = transformer_ts(smi, ts_guess_xyz)
            ts_structs.update(ts_mols)
    else:
        for smi, rpos in zip(ligand_smiles_list, rpos_list):
            ts_mols = transformer_ts(smi, ts_guess_xyz, rpos_list=rpos)
            ts_structs.update(ts_mols)

    if return_format == "dict":
        return ts_structs

    if return_format == "list":
        ts_structs_list = []
        for k, i in ts_structs.items():
            ts_structs_list.append({k:i})   
        return ts_structs_list


def create_mol_per_rpos(
    ligand_smiles_df: pd.DataFrame,
    return_format: str = "dict",
    select_mols: str | list[str] = "all",
) -> list[dict[str, Mol | tuple[Mol, dict]]] | dict[str, Mol | tuple[Mol, dict]]:
    """Generate catalytic-cycle molecules for each unique ligand SMILES.

    Parameters
    ----------
    ligand_smiles_df : pandas.DataFrame
        Input table containing a ``smiles`` column with ligand SMILES strings.
        Duplicate SMILES entries are ignored after the first occurrence.
    return_format : str, optional
        Output format. Use ``"dict"`` to return a merged dictionary of
        molecule names to RDKit molecules, or ``"list"`` to return a list of
        single-item dictionaries.
    select_mols : str or list[str], optional
        Molecule selection passed through to :func:`frust.transformers.transformer_mols`.
        Supported string values are ``"all"``, ``"uniques"``, and ``"generics"``.

    Returns
    -------
    dict[str, tuple[rdkit.Chem.Mol, dict]] or list[dict[str, tuple[rdkit.Chem.Mol, dict]]]
        Catalytic-cycle molecule structures plus dataframe metadata keyed by
        their generated names.

    Raises
    ------
    ValueError
        If ``ligand_smiles_df`` does not contain a ``smiles`` column or if
        ``return_format`` is unsupported.
    """
    if "smiles" not in ligand_smiles_df.columns:
        raise ValueError("ligand_smiles_df must contain a 'smiles' column")

    smiles_series = ligand_smiles_df["smiles"]
    if smiles_series.isna().any():
        raise ValueError("ligand_smiles_df['smiles'] contains missing values")

    from frust.transformers import transformer_mols

    ligand_smiles_list = list(dict.fromkeys(smiles_series.tolist()))
    rpos_map: dict[str, tuple[int, ...]] = {}

    if "rpos" in ligand_smiles_df.columns:
        extracted_rpos = _extract_rpos_from_df(ligand_smiles_df)
        grouped_rpos: dict[str, list[int]] = {}

        for smi, rpos_tuple in zip(smiles_series.tolist(), extracted_rpos):
            grouped_rpos.setdefault(smi, [])
            for rpos in rpos_tuple:
                if rpos not in grouped_rpos[smi]:
                    grouped_rpos[smi].append(rpos)

        rpos_map = {smi: tuple(rpos_list) for smi, rpos_list in grouped_rpos.items()}

    mols: dict[str, Mol] = {}
    for smi in ligand_smiles_list:
        transformer_kwargs = {"ligand_smiles": smi}
        if smi in rpos_map:
            transformer_kwargs["rpos_list"] = rpos_map[smi]

        if select_mols == "all":
            tmp = transformer_mols(**transformer_kwargs, return_metadata=True)
        elif select_mols == "uniques":
            tmp = transformer_mols(**transformer_kwargs, only_uniques=True, return_metadata=True)
        elif select_mols == "generics":
            tmp = transformer_mols(**transformer_kwargs, only_generics=True, return_metadata=True)
        else:
            tmp = transformer_mols(**transformer_kwargs, select=select_mols, return_metadata=True)

        mols.update(tmp)

    if return_format == "dict":
        return mols

    if return_format == "list":
        mols_list = []
        for k, i in mols.items():
            mols_list.append({k: i})
        return mols_list

    raise ValueError(f"Unrecognized return_format: {return_format!r}")


def find_unique_ch(smi: str):
    # --- Find unique positions and check that they are valid cH --- #
    lig_mol = Chem.MolFromSmiles(smi)
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
    return unique_cH


def find_ch(smi: str):
    lig_mol = Chem.MolFromSmiles(smi)
    lig_mol = Chem.AddHs(lig_mol)

    cH_patt = Chem.MolFromSmarts("[cH]")
    matches = lig_mol.GetSubstructMatches(cH_patt)

    cH_atoms = tuple(ind[0] for ind in matches)
    return cH_atoms
