from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def _find_unique_atoms_from_ranks(ranks: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for i, rank in enumerate(ranks):
        if rank not in seen:
            out.append(i)
            seen.add(rank)
    return out

def _unique_aromatic_ch_positions(mol: Chem.Mol) -> List[int]:
    c_h_pattern = Chem.MolFromSmarts("[cH]")
    matches = mol.GetSubstructMatches(c_h_pattern)
    ch_atoms = [match[0] for match in matches]

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    unique_atoms = _find_unique_atoms_from_ranks(ranks)

    return sorted(set(unique_atoms).intersection(ch_atoms))

def _normalize_smiles_input(
    data: Union[pd.DataFrame, Sequence[str], str],
    smiles_col: str,
) -> List[str]:
    if isinstance(data, str):
        return [data]

    if isinstance(data, pd.DataFrame):
        if smiles_col not in data.columns:
            raise ValueError(f"Column '{smiles_col}' not found in DataFrame.")

        smiles_list: List[str] = []
        for value in data[smiles_col].tolist():
            if pd.isna(value):
                smiles_list.append("")
            else:
                smiles_list.append(str(value))
        return smiles_list

    if isinstance(data, Sequence):
        return [str(smi) for smi in data]

    raise TypeError(
        "data must be a pandas DataFrame, a SMILES string, or a sequence "
        "of SMILES strings."
    )

def DrawUniqueChGrid(
    data: Union[pd.DataFrame, Sequence[str], str],
    smiles_col: str = "smiles",
    mols_per_row: int = 4,
    sub_img_size: Tuple[int, int] = (250, 350),
    add_atom_indices: bool = True,
    kekulize: bool = True,
):
    smiles_list = _normalize_smiles_input(data, smiles_col)

    mols: List[Optional[Chem.Mol]] = []
    legends: List[str] = []
    highlight_lists: List[List[int]] = []

    for smi in smiles_list:
        smi = smi.strip()

        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            mols.append(None)
            legends.append("INVALID SMILES" + (f"\n{smi}" if smi else ""))
            highlight_lists.append([])
            continue

        unique_ch = _unique_aromatic_ch_positions(mol)
        draw_mol = Draw.rdMolDraw2D.PrepareMolForDrawing(
            mol,
            kekulize=kekulize,
        )

        mols.append(draw_mol)
        highlight_lists.append(unique_ch)
        legends.append(
            "unique cH: "
            + (", ".join(map(str, unique_ch)) if unique_ch else "none")
            + "\n"
            + smi
        )

    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.addAtomIndices = add_atom_indices
    opts.annotationFontScale = 0.8

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        highlightAtomLists=highlight_lists,
        useSVG=False,
        drawOptions=opts,
    )
