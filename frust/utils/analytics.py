import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

def summarize_ts_vibrations(
    df: pd.DataFrame,
    col: str = "DFT-wB97X-D3-6-31G**-OptTS-vibs",
    max_rows: int = 5
):
    true_ts_count = 0
    non_ts_count = 0
    rows = []

    for idx, row in df.iterrows():
        ligand = row.get("ligand_name", "")
        rpos   = row.get("rpos", "")
        vibs   = row[col]

        freqs = [entry.get('frequency') for entry in vibs]
        neg_freqs = [f for f in freqs if f < 0]
        pos_freqs = [f for f in freqs if f >= 0]

        is_true_ts = len(neg_freqs) == 1
        status = "✅ True TS" if is_true_ts else f"❌ Not TS ({len(neg_freqs)} neg)"

        if is_true_ts:
            true_ts_count += 1
        else:
            non_ts_count += 1

        if neg_freqs:
            neg_str = ", ".join(f"{f:.2f}" for f in neg_freqs[:3])
            if len(neg_freqs) > 3:
                neg_str += " ..."
        else:
            neg_str = "No negatives"
        neg_str += " |"

        if pos_freqs:
            pos_str = ", ".join(f"{f:.1f}" for f in pos_freqs[:3])
            if len(pos_freqs) > 3:
                pos_str += " ..."
        else:
            pos_str = "No positives"
        pos_str += " |"

        rows.append({
            "Structure": idx,
            "Ligand": ligand,
            "RPOS": rpos,
            "Status": status,
            "Neg. freqs": neg_str,
            "Pos. freqs": pos_str
        })

    result_df = pd.DataFrame(rows)

    print(result_df.head(max_rows).to_string(index=False))
    if len(result_df) > max_rows:
        print(f"\n... and {len(result_df) - max_rows} more rows.")

    print("\nSummary:")
    print(f"  ✅ True TSs : {true_ts_count}")
    print(f"  ❌ Non-TSs  : {non_ts_count}")


def _svg_annotated_smi(
        smi, pos_list, dE_list,
        size=(250, 250), highlight_color=(1, 0, 0)):
    """Return an SVG string of the molecule with per-atom ΔE labels."""

    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)

    for p, e in zip(pos_list, dE_list):
        mol.GetAtomWithIdx(int(p)).SetProp("atomNote", f"{e:.2f}")

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.drawAtomNotes       = True
    opts.annotationFontScale = 0.9 

    drawer.DrawMolecule(
        mol,
        #highlightAtoms      =[int(p) for p in pos_list],
        #highlightAtomColors ={int(p): highlight_color for p in pos_list},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def build_annotated_frame(df,
                          ligand_col="ligand_name",
                          smi_col="smiles",
                          pos_col="rpos",
                          energy_col="dE"):
    """One row per ligand + an SVG column with all ΔE annotations."""
    rows = []
    for lig, grp in df.groupby(ligand_col):
        smi  = grp[smi_col].iloc[0]
        pos  = grp[pos_col].astype(int).tolist()
        dE   = grp[energy_col].tolist()
        svg  = _svg_annotated_smi(smi, pos, dE)
        rows.append({ligand_col: lig, smi_col: smi, "annotated_svg": svg})
    
    df = pd.DataFrame(rows)
    html = df.to_html(
    escape=False,
    formatters={"annotated_svg": lambda x: x},
    index=False
    )
    return df, html