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


import re
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

def _svg_annotated_smi(
        smi, pos_list, dE_list,
        size=(250, 250), highlight_color=(1, 0, 0),
        show_rpos: bool = False, step_list: list[str] | None = None,
        fixed_bond_px: float | None = 25.0,
        note_font_px: float | None = None,
        annotation_scale: float = 1):
    """Return an SVG string of the molecule with per-atom ΔE labels.

    Args:
        smi: SMILES string.
        pos_list: Atom indices to annotate.
        dE_list: Values to display next to each atom.
        size: (width, height) of the SVG in px.
        highlight_color: Unused unless highlight code is enabled.
        show_rpos: If True, append ' (rX)' after the value.
        step_list: Optional per-position step tags (e.g., 'ts1').
            When provided, labels become like '20.10(ts1)'.
        fixed_bond_px: If set, draw with a fixed px/bond to keep drawings
            on a common scale (RDKit may still shrink to fit if canvas is
            too small).
        note_font_px: If set, enforce an absolute pixel size for annotation
            text (via fixedFontSize * annotationFontScale).
        annotation_scale: Multiplier for annotation size relative to the
            base label font.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    rdDepictor.Compute2DCoords(mol)

    for i, (p, e) in enumerate(zip(pos_list, dE_list)):
        try:
            p_int = int(p)
        except (TypeError, ValueError):
            continue

        try:
            val = float(e)
            note = f"{val:.2f}"
        except (TypeError, ValueError):
            note = f"{e}"
    
        if step_list is not None and i < len(step_list):
            note += f" [{step_list[i]}]"
        if show_rpos:
            note += f" (r{p_int})"

        mol.GetAtomWithIdx(p_int).SetProp("atomNote", note)

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.drawAtomNotes = True

    # Keep all molecules at the same scale (px per bond).
    if fixed_bond_px is not None:
        opts.fixedBondLength = float(fixed_bond_px)

    # Control annotation size deterministically.
    # If note_font_px is given, make the final annotation size = note_font_px.
    scale = float(annotation_scale) if annotation_scale else 0.4
    opts.annotationFontScale = scale
    use_svg_patch = False
    if note_font_px is not None:
        # final px = annotationFontScale * fixedFontSize
        base_font_px = max(1, int(round(float(note_font_px) / max(scale, 1e-6))))
        if hasattr(opts, "fixedFontSize"):
            opts.fixedFontSize = base_font_px  # <- must be INT
            # (optional clamps; harmless when fixedFontSize is honored)
            if hasattr(opts, "minFontSize"):
                opts.minFontSize = base_font_px
            if hasattr(opts, "maxFontSize"):
                opts.maxFontSize = base_font_px
        else:
            # very old RDKit fallback: patch SVG after drawing
            use_svg_patch = True

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if note_font_px is not None and use_svg_patch:
        px = float(note_font_px)
        # Patch both style: and attribute-based font declarations.
        svg = re.sub(r'(font-size\s*:\s*)[\d.]+px', rf'\1{px}px', svg)
        svg = re.sub(r'(font-size\s*=\s*["\'])[\d.]+(px)?(["\'])',
                     rf'\1{px}\3', svg)

    return svg

import pandas as pd

def build_annotated_frame(
    df: pd.DataFrame,
    ligand_col: str = "ligand_name",
    smi_col: str = "smiles",
    pos_col: str = "rpos",
    energy_col: str = "dE",
    output_path: str | None = None,
    step_col: str | None = None,
    show_rpos: bool = False,
    energy_cols: list[str] | None = None,
    size: tuple[int, int] = (250, 250),
    fixed_bond_px: float | None = 25.0,   # NEW
    note_font_px: float | None = 14.0,    # NEW
    annotation_scale: float = 1,   # NEW
) -> tuple[pd.DataFrame, str]:
    """One row per ligand + an SVG column with all ΔE annotations.
    If output_path is set, writes a standalone HTML file.

    Args:
        df (pd.DataFrame): Input data.
        ligand_col (str): Column name for ligand grouping.
        smi_col (str): Column name for SMILES.
        pos_col (str): Column name for annotation positions.
        energy_col (str): Column name for ΔE values. If this column is not
            present, you can provide `energy_cols` and the function will use
            the per-row maximum across those columns.
        output_path (str | None): File path to write HTML. No file written
            if None.
        step_col (str | None): Optional column with per-row step labels.
        show_rpos (bool): If True, append ' (rX)' to each label.
        energy_cols (list[str] | None): Columns to compute per-row maxima
            when `energy_col` is not available.
        fixed_bond_px (float | None): Passed to `_svg_annotated_smi`.
        note_font_px (float | None): Passed to `_svg_annotated_smi`.

    Returns:
        Tuple[pd.DataFrame, str]:
            - DataFrame with one row per ligand and an "annotated_svg" column.
            - HTML table fragment as a string.
    """
    for col in (ligand_col, smi_col, pos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    work = df.copy()

    energy_col_local = energy_col
    step_col_local = step_col

    if energy_col_local not in work.columns:
        if not energy_cols:
            raise ValueError(
                "Energy column not found and no energy_cols provided. "
                f"Missing: {energy_col!r}"
            )
        missing = [c for c in energy_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing energy columns: {missing}")

        vals = work[energy_cols].apply(pd.to_numeric, errors="coerce")
        work["_energy_max_tmp"] = vals.max(axis=1, skipna=True)
        try:
            step_idx = vals.idxmax(axis=1, skipna=True)
        except Exception:
            step_idx = vals.apply(
                lambda r: (r.idxmax(skipna=True)
                           if r.notna().any() else None),
                axis=1
            )
        work["_step_tmp"] = step_idx
        energy_col_local = "_energy_max_tmp"
        if step_col_local is None:
            step_col_local = "_step_tmp"

    if step_col_local is not None and step_col_local not in work.columns:
        raise ValueError(f"Column not found: {step_col_local}")

    def _norm_step(s: object) -> str:
        if not isinstance(s, str):
            return str(s)
        t = s.strip()
        t = t.replace("dG_", "").replace("dE_", "").lower()
        if t.startswith("ts"):
            t = t[2:]  # remove the 'ts' prefix
        return t

    rows = []
    for lig, grp in work.groupby(ligand_col):
        smi = grp[smi_col].iloc[0]
        pos = grp[pos_col].astype(int).tolist()
        dE_vals = grp[energy_col_local].tolist()

        steps = None
        if step_col_local is not None:
            steps = [_norm_step(x) for x in grp[step_col_local].tolist()]

        svg = _svg_annotated_smi(
            smi, pos, dE_vals,
            show_rpos=show_rpos,
            step_list=steps,
            fixed_bond_px=fixed_bond_px,
            note_font_px=note_font_px,
            annotation_scale=annotation_scale,  # pass through
            size=size,
        )
        rows.append({ligand_col: lig, smi_col: smi, "annotated_svg": svg})

    result_df = pd.DataFrame(rows)
    html_table = result_df.to_html(
        escape=False,
        formatters={"annotated_svg": lambda x: x},
        index=False
    )

    if output_path:
        title = output_path.split(".html")[0]
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
<p>{title}</p>
{html_table}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

    return result_df, html_table


import argparse
from pathlib import Path
from typing import Sequence, Union

def merge_parquet_dir(
    input_dir: Union[str, Path],
    output: Union[str, Path] = "merged.parquet",
) -> Path:
    """Merge multiple Parquet files with identical schemas into one file.

    Args:
        input_dir: Directory containing .parquet files to merge.
        output: Output Parquet file path.

    Returns:
        Path to the merged Parquet file.

    Raises:
        FileNotFoundError: If the input directory does not exist or contains
            no .parquet files.
        ValueError: If the merged DataFrame is empty.
    """
    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input directory '{in_path}' not found.")
    files = sorted(in_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in '{in_path}'.")
    dfs = [pd.read_parquet(str(fp)) for fp in files]
    merged = pd.concat(dfs, ignore_index=True)
    if merged.empty:
        raise ValueError("Merged DataFrame is empty.")
    out_path = Path(output)
    merged.to_parquet(out_path)
    return out_path


def main_merge_parquet(argv: Sequence[str] | None = None) -> int:
    """CLI entry for merging Parquet files.

    Args:
        argv: Optional list of CLI args for testing or entry points.

    Returns:
        Process exit code (0 on success, nonzero on error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Parquet files with the same schema into one file."
        )
    )
    default_in = str(Path(__file__).resolve().parent.parent / "results")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=default_in,
        help="Directory containing .parquet files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged.parquet",
        help="Output Parquet file path.",
    )
    args = parser.parse_args(argv)
    try:
        out = merge_parquet_dir(args.input_dir, args.output)
    except Exception as e:
        print(str(e))
        return 1
    print(f"Merged files into '{out}'.")
    return 0