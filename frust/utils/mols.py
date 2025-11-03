# frust/utils/mols.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdDetermineBonds
from rdkit.Geometry.rdGeometry import Point3D
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import Mol
from frust.utils.io import read_ts_type_from_xyz

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
    import pubchempy as pcp
    import time
    
    max_retries = 5      # Maximum number of retry attempts
    delay = 10           # Seconds to wait between attempts

    compounds = None
    for attempt in range(1, max_retries + 1):
        try:
            compounds = pcp.get_compounds(smiles, 'smiles')
            break
        except pcp.PubChemHTTPError as e:
            if 'PUGREST.ServerBusy' in str(e):
                print(f"Attempt {attempt}/{max_retries}: PubChem is busy. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"Attempt {attempt}/{max_retries}: Encountered error: {e}")
                break

    if compounds and compounds[0].iupac_name:
        name = compounds[0].iupac_name
        return name.replace(" ", "_")
    else:
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


def create_ts_per_rpos(
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