from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.interpolate import PchipInterpolator
from scipy.stats import linregress, spearmanr
from tooltoad.chemutils import ac2mol
from tooltoad.vis import DrawMolSvg, MolTo3DGrid, RxnTo3DGrid
from frust.utils.RMSD import compare_xyz_rmsd

darkmode: bool = False

def set_theme(dark: bool = True) -> None:
    """Set module-wide theme."""
    global darkmode
    darkmode = dark


@contextmanager
def use_darkmode(on: bool = True):
    """Temporarily enable dark mode within a with-block."""
    global darkmode
    prev = darkmode
    darkmode = on
    try:
        yield
    finally:
        darkmode = prev

def plot_mols(
    df: pd.DataFrame,
    row_indices: Optional[List[int]] = None,
    ligand_filter: Optional[List[str]] = None,
    rpos_filter: Optional[List[Union[str, int]]] = None,
    exclude_coords: Optional[List[str]] = None,
    include_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    dark: bool = False,
    **molto3d_kwargs: Any
) -> None:
    """
    Display molecules from a dataframe with filtering capabilities.

    Args:
        df: DataFrame with molecular data
        row_indices: List of row indices to display (if None, displays all
            rows)
        ligand_filter: List of ligand names to include (if None, includes all)
        rpos_filter: List of rpos values to include (if None, includes all)
        exclude_coords: List of coordinate column patterns to exclude
        include_coords: List of coordinate column patterns to include
            (overrides exclude)
        coord_indices: List of indices or a slice for coordinate columns
            (overrides include/exclude).
        dark: If True, use a dark background by default. Ignored if
            'background_color' is explicitly provided in molto3d_kwargs.
        **molto3d_kwargs: Additional arguments to pass to MolTo3DGrid

    Returns:
        None
    """
    filtered_df = df.copy()

    if row_indices is not None:
        filtered_df = filtered_df.iloc[row_indices]

    if ligand_filter is not None:
        filtered_df = filtered_df[
            filtered_df['ligand_name'].isin(ligand_filter)
        ]

    if rpos_filter is not None:
        filtered_df = filtered_df[filtered_df['rpos'].isin(rpos_filter)]

    if filtered_df.empty:
        print("No molecules match the specified filters.")
        return

    coord_columns = [c for c in df.columns if "coords" in c]

    if coord_indices is not None:
        if isinstance(coord_indices, slice):
            coord_columns = coord_columns[coord_indices]
        else:
            coord_columns = [
                coord_columns[i] for i in coord_indices
                if 0 <= i < len(coord_columns)
            ]
    elif include_coords is not None:
        coord_columns = [
            c for c in coord_columns
            if any(pattern in c for pattern in include_coords)
        ]
    elif exclude_coords is not None:
        coord_columns = [
            c for c in coord_columns
            if not any(pattern in c for pattern in exclude_coords)
        ]

    if not coord_columns:
        print("No coordinate columns found after filtering.")
        return

    print(f"Found {len(coord_columns)} coordinate columns: {coord_columns}")
    print(f"Processing {len(filtered_df)} rows")

    all_mols = []
    all_legends = []

    for idx, row in filtered_df.iterrows():
        atoms = row["atoms"]
        ligand_name = row["ligand_name"]
        rpos = row["rpos"]

        for coord_col in coord_columns:
            coords = row[coord_col]

            if coords is not None:
                if isinstance(coords, np.ndarray):
                    is_valid = coords.size > 0 and not pd.isna(coords).all()
                else:
                    is_valid = (not pd.isna(coords)
                                if not isinstance(coords, list)
                                else len(coords) > 0)

                if is_valid:
                    mol = ac2mol(atoms, coords)
                    if mol is not None:
                        all_mols.append(mol)

                        coord_type = (coord_col.replace("coords_", "")
                                      .replace("_coords", ""))
                        if rpos is None:
                            legend = f"{ligand_name}\n{coord_type}"
                        else:
                            legend = f"{ligand_name} r{rpos}\n{coord_type}"
                        all_legends.append(legend)

    if not all_mols:
        print("No valid molecules could be generated.")
        return

    print(f"Generated {len(all_mols)} molecules for display")

    # Defaults (can be overridden by molto3d_kwargs)
    molto3d_args = {
        'legends': all_legends,
        'show_labels': False,
        'show_confs': True,
        #'background_color': 'black' if darkmode else 'white',
        'cell_size': (400, 400),
        'columns': len(coord_columns) if coord_indices is None else 4,
        'linked': False,
        'kekulize': True,
        'show_charges': True,
    }

    molto3d_args.update(molto3d_kwargs)

    MolTo3DGrid(all_mols, **molto3d_args)

def plot_row(
    df: pd.DataFrame,
    row_index: int = 0,
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display all coordinate types for a single row."""
    plot_mols(
        df,
        row_indices=[row_index],
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )


def plot_lig(
    df: pd.DataFrame,
    ligand_names: Union[str, List[str]],
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display molecules filtered by ligand name(s)."""
    if isinstance(ligand_names, str):
        ligand_names = [ligand_names]

    plot_mols(
        df,
        ligand_filter=ligand_names,
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )


def plot_rpos(
    df: pd.DataFrame,
    rpos_values: Union[str, int, List[Union[str, int]]],
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display molecules filtered by rpos value(s)."""
    if isinstance(rpos_values, (str, int)):
        rpos_values = [rpos_values]

    plot_mols(
        df,
        rpos_filter=rpos_values,
        exclude_coords=exclude_coords,
        coord_indices=coord_indices,
        **kwargs
    )


def plot_vibs(
    df,
    row_index=0,
    vId: int = 0,
    width: float = 600,
    height: float = 400,
    numFrames: int = 20,
    amplitude: float = 1,
    transparent: bool = True,
    fps: float | None = None,
    reps: int = 50,
    custom_coords_col_name: str | None = None,
    row_indices: list[int] | None = None,
    viewergrid: tuple[int, int] | None = None,
    linked: bool = True,
    freq_label: bool = True,
    legends: list[str] | None = None,
    legend_screen_offset: dict | None = None,  # e.g. {'x': 10, 'y': 6}
    export_HTML: str = ""
):
    import math

    import py3Dmol
    from tooltoad.vis import ac2xyz, show_vibs

    vibs_col = [c for c in df.columns if "vibs" in c][-1]
    print(f"vibs col {vibs_col}")
    vibs_col_pre = vibs_col.split("vibs")[0]
    coords_col = vibs_col_pre + "opt_coords"
    if custom_coords_col_name:
        coords_col = custom_coords_col_name

    # ---- Single-row (unchanged rendering, improved label) ----
    if row_indices is None or (isinstance(row_indices, list)
                               and len(row_indices) == 1):
        if isinstance(row_indices, list) and len(row_indices) == 1:
            row_index = row_indices[0]

        atoms = df["atoms"].iloc[row_index]
        vibs = df[vibs_col].iloc[row_index]
        coords = df[coords_col].iloc[row_index]

        view_dict = {"atoms": atoms, "opt_coords": coords, "vibs": vibs}
        bg = "0x000000" if darkmode else "0xeeeeee"

        view = show_vibs(
            view_dict,
            vId,
            width,
            height,
            numFrames,
            amplitude,
            transparent,
            fps,
            reps,
            background_color=bg,
            export_HTML=export_HTML,
        )

        if freq_label:
            vib = vibs[vId]
            freq = vib["frequency"]
            label_text = (
                legends[0] if legends and len(legends) >= 1
                else f"{freq:.1f} cm^-1"
            )
            # Screen-anchored like MolTo3DGrid
            font_color = "white" if darkmode else "black"
            back_color = "black" if darkmode else "white"
            offs = legend_screen_offset or {"x": 10, "y": 6}
            view.addLabel(
                label_text,
                {
                    "useScreen": True,
                    "inFront": True,
                    "fontSize": 14,
                    "fontColor": font_color,
                    "backgroundColor": back_color,
                    "borderColor": font_color,
                    "borderWidth": 1,
                    "screenOffset": offs,  # top-left corner offset
                },
            )
        return view

    # ---- Grid across multiple rows ----
    idxs = list(row_indices)
    if legends is not None and len(legends) != len(idxs):
        raise ValueError("Length of legends must match row_indices.")

    interval_ms = 50 if fps is None else max(1, int(1000.0 / fps))

    n = len(idxs)
    if viewergrid is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        rows, cols = viewergrid

    bg = "0x000000" if darkmode else "0xeeeeee"
    alpha = 0 if transparent else 1

    p = py3Dmol.view(width=width, height=height,
                     viewergrid=(rows, cols), linked=linked)

    for i, ri in enumerate(idxs):
        atoms = df["atoms"].iloc[ri]
        vibs = df[vibs_col].iloc[ri]
        coords = df[coords_col].iloc[ri]

        vib = vibs[vId]
        mode = vib["mode"]
        freq = vib["frequency"]

        xyz = ac2xyz(atoms, coords)

        r, c = divmod(i, cols)
        p.addModel(xyz, "xyz", viewer=(r, c))

        propmap = [
            {"index": j, "props": {"dx": m[0], "dy": m[1], "dz": m[2]}}
            for j, m in enumerate(mode)
        ]
        p.mapAtomProperties(propmap, viewer=(r, c))

        p.vibrate(numFrames, amplitude, True, viewer=(r, c))
        p.animate(
            {"loop": "backAndForth", "interval": interval_ms, "reps": reps},
            viewer=(r, c),
        )

        p.setStyle({"sphere": {"radius": 0.4}, "stick": {}}, viewer=(r, c))
        p.setBackgroundColor(bg, alpha, viewer=(r, c))
        p.zoomTo(viewer=(r, c))

        if freq_label:
            label_text = (
                legends[i] if legends else f"{freq:.1f} cm^-1"
            )
            font_color = "white" if darkmode else "black"
            back_color = "black" if darkmode else "white"
            offs = legend_screen_offset or {"x": 10, "y": 6}
            p.addLabel(
                label_text,
                {
                    "useScreen": True,
                    "inFront": True,
                    "fontSize": 14,
                    "fontColor": font_color,
                    "backgroundColor": back_color,
                    "borderColor": font_color,
                    "borderWidth": 1,
                    "screenOffset": offs,  # per-cell top-left
                },
                viewer=(r, c),
            )

    return p


def plot_regression_outliers(
    df: pd.DataFrame,
    x_col: str = "dE",
    y_col: str = "dG",
    xlabel: str = "dE, kcal/mol",
    ylabel: str = "dG, kcal/mol",
    font_size: int = 14,
    label_col: str = "ligand_name",
    rpos_col: str = "rpos",
    method: str = "spearman",
    num_outliers: int = 2,
    size: tuple = (8, 6),
    plot_1x: bool = False,
    equal_axis: bool = False,
) -> pd.DataFrame:
    """Plot x vs y with linear fit, score outliers, and annotate top points.

    Args:
        df (pd.DataFrame): Input data.
        x_col (str): Name of the column to use for x values. Defaults to "dG".
        y_col (str): Name of the column to use for y values. Defaults to "dE".
        label_col (str): Column used for point labels. Defaults to
            "ligand_name".
        rpos_col (str): Column used for position annotations. Defaults to
            "rpos".
        method (str, optional): Scoring method, "pearson" or "spearman".
            Defaults to "spearman".
        num_outliers (int, optional): Number of top outliers to annotate.
            Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame of the top outliers sorted by score.
    """
    for col in (x_col, y_col, label_col, rpos_col):
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Invalid method: {method}")

    data = df.copy()
    x = data[x_col]
    y = data[y_col]

    lr = linregress(x, y)
    y_fit = lr.slope * x + lr.intercept
    rho, _ = spearmanr(x, y)
    
    c = float(np.mean(y - x))
    y_hat = x + c

    # Metrics
    y_arr = np.asarray(y, dtype=float)
    yfit_arr = np.asarray(y_fit, dtype=float)
    yhat_arr = np.asarray(y_hat, dtype=float)

    rmsd_fit = float(np.sqrt(np.mean((y_arr - yfit_arr) ** 2)))
    rmsd_hat = float(np.sqrt(np.mean((y_arr - yhat_arr) ** 2)))

    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    sse_hat = float(np.sum((y_arr - yhat_arr) ** 2))
    r2_hat = 1.0 - (sse_hat / sst) if sst > 0 else np.nan

    rho_hat, _ = spearmanr(y_hat, y)

    # Print equations to stdout (not on the plot)
    eq_label = (f"y = {lr.slope:.2f}x "
                f"{'+' if lr.intercept >= 0 else '-'} "
                f"{abs(lr.intercept):.2f}")
    print("[INFO]: Linear relation:", eq_label)
    eq2_label = (f"y = 1x "
                 f"{'+' if c >= 0 else '-'} "
                 f"{abs(c):.2f}")
    print("[INFO]: Error relationship: ", eq2_label)

    if method == "pearson":
        data["score"] = (y - y_fit).abs()
    else:
        data["rank_x"] = x.rank()
        data["rank_y"] = y.rank()
        data["score"] = (data["rank_y"] - data["rank_x"]).abs()

    outliers = data.nlargest(num_outliers, "score")

    style_ctx = (plt.style.context('dark_background')
                 if darkmode else nullcontext())
    with style_ctx:
        plt.figure(figsize=size)
        plt.scatter(x, y, alpha=0.7)
        plt.plot(
            x, y_fit, color="red", marker="",
            label=(f"$R^2$={lr.rvalue**2:.3f}, "
                   f"spearman={rho:.3f}, "
                   f"RMSD={rmsd_fit:.3f}")
        )
        if plot_1x:
            plt.plot(
                x, y_hat, marker="",
                label=(f"$R^2$={r2_hat:.3f}, "
                       f"spearman={rho_hat:.3f}, "
                       f"RMSD={rmsd_hat:.3f}")
            )
            
        for _, row in outliers.iterrows():
            label = f"{row[label_col]}-r{int(row[rpos_col])}"
            plt.annotate(
                label,
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.5)
            )
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid(True)
        if equal_axis:
            xmin = min(x.min(), y.min())
            xmax = max(x.max(), y.max())

            plt.xlim(xmin, xmax)
            plt.ylim(xmin, xmax)
            plt.gca().set_aspect("equal", adjustable="box")        

        plt.tight_layout()
        plt.show()
        
    return None


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


def plot_energy_profile(
    states,
    ylabel: str = "ΔG (kcal/mol)",
    n_points: int = 500,
    figsize=(8, 3.5),
    annotate_energies: bool = True,
    decimals: int = 1,
    int_prefix: str = "int",
    label_offset_up: int = 8,
    label_offset_down: int = 12,
    hide_y_ticks: bool = True,
    hide_x_ticks: bool = True,
    hide_spines: bool = True,
    grid: bool = False,
    ax=None,
    dummy_substr: str = "dummy",
    dummy_alpha: float = 0.5,
    side_token: str = "side-rxn",
    show_main_to_product: bool = True,
    main_to_product_alpha: float = 1,
    main_to_product_linestyle: str = ":",
    main_to_product_lw: float = 3.0,
    main_to_product_bow: float = 0.5,
    main_to_product_drop_frac: float = 0.65,
    main_to_product_drop_points: int | None = None,
    main_to_product_flat_points: int | None = None,
    product_x_offset: float = 0.18,
    # --- multi-molecule overlays ---
    overlay: str = "auto",  # "auto" | "off" | "on"
    overlay_annotate: str = "energy",  # "none" | "energy" | "full"
    overlay_alpha: float = 0.35,
    overlay_lw_scale: float = 1.0,
    marker: str = "o",
    overlay_markers=None,
    show_legend: bool = True,
    profile_label: str | None = None,
    overlay_colors=None,
    same_energy_tol: float = 1e-3,
    same_energy_mode: str = "hide",  # "hide" | "tag"
    same_energy_tag: str = "≡",
    # --- NEW: bottom state labels (recommended for overlays) ---
    show_state_labels: bool | None = None,
    state_label_rotation: float = 0.0,
    font_size: float | None = None,
    state_label_fontsize: float | None = None,
    energy_fontsize: float | None = None,
    axis_label_fontsize: float | None = None,
    tick_label_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    same_energy_tag_fontsize: float | None = None,
    state_label_pad: float = 6.0,
):
    base_fontsize = 12.0 if font_size is None else float(font_size)
    state_label_fontsize = (
        base_fontsize
        if state_label_fontsize is None
        else float(state_label_fontsize)
    )
    energy_fontsize = (
        base_fontsize
        if energy_fontsize is None
        else float(energy_fontsize)
    )
    axis_label_fontsize = (
        base_fontsize
        if axis_label_fontsize is None
        else float(axis_label_fontsize)
    )
    tick_label_fontsize = (
        base_fontsize
        if tick_label_fontsize is None
        else float(tick_label_fontsize)
    )
    legend_fontsize = (
        base_fontsize
        if legend_fontsize is None
        else float(legend_fontsize)
    )
    same_energy_tag_fontsize = (
        energy_fontsize
        if same_energy_tag_fontsize is None
        else float(same_energy_tag_fontsize)
    )

    def _parse_placement(value):
        if value is None:
            return None

        s = str(value).lower().strip()
        if not s:
            return None

        aliases = {
            "t": "top",
            "top": "top",
            "b": "bottom",
            "bottom": "bottom",
            "l": "left",
            "left": "left",
            "r": "right",
            "right": "right",
        }

        parts = [p for p in s.replace("_", "-").replace(" ", "-").split("-") if p]

        expanded = []
        for p in parts:
            p = p.strip().lower()
            if p and all(ch in {"t", "b", "l", "r"} for ch in p) and p not in aliases:
                expanded.extend(list(p))
            else:
                expanded.append(p)

        parts = [aliases.get(p, p) for p in expanded]

        allowed = {"top", "bottom", "left", "right"}
        if any(p not in allowed for p in parts):
            return None

        counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for p in parts:
            counts[p] += 1

        if counts["top"] and counts["bottom"]:
            return None
        if counts["left"] and counts["right"]:
            return None

        if sum(counts.values()) == 0:
            return None

        return counts

    def _dedup_for_interp(xv, yv):
        seen = set()
        x_out = []
        y_out = []
        for xx, yy in zip(xv, yv):
            key = float(xx)
            if key in seen:
                continue
            seen.add(key)
            x_out.append(float(xx))
            y_out.append(float(yy))
        return np.array(x_out, dtype=float), np.array(y_out, dtype=float)

    def _norm_label(label):
        return str(label).strip().lower()

    def _is_product(label):
        return _norm_label(label).startswith("product")

    def _parse_entries(profile_states):
        entries = []
        seg_ids = []
        seg = 0
        token = str(side_token).lower().strip()

        side_anchor_label = None
        side_connector_rise_frac = None
        embedded_side_label = None

        for item in profile_states:
            if isinstance(item, str):
                side_spec, legend_spec = (
                    item.split("#", 1)
                    if "#" in item
                    else (item, None)
                )
                parsed_legend = (
                    legend_spec.strip()
                    if legend_spec is not None and legend_spec.strip()
                    else None
                )
                s = side_spec.lower().strip()

                if s == token:
                    embedded_side_label = parsed_legend
                    seg = 1
                    continue

                if s.startswith(token + "@") or s.startswith(token + ":"):
                    rest = (
                        side_spec.split("@", 1)[1]
                        if "@" in side_spec
                        else side_spec.split(":", 1)[1]
                    )
                    parts = [p.strip() for p in str(rest).split("@") if p.strip()]

                    side_anchor_label = parts[0] if len(parts) >= 1 else None
                    side_connector_rise_frac = None
                    if len(parts) >= 2:
                        side_connector_rise_frac = float(parts[1])

                    embedded_side_label = parsed_legend
                    seg = 1
                    continue

                raise ValueError(
                    f"Unknown string entry in states: {item!r}. "
                    f"Only {side_token!r} (optionally with @{'<label>'} and "
                    "a #legend suffix) is supported."
                )

            label = item[0]
            energy = item[1]
            placement = item[2] if len(item) >= 3 else None

            entries.append((label, energy, placement))
            seg_ids.append(seg)

        return (
            entries,
            seg_ids,
            side_anchor_label,
            side_connector_rise_frac,
            embedded_side_label,
        )

    def _compute_x_single(entries):
        names = [e[0] for e in entries]

        x_list = []
        product_indices = []
        for i, label in enumerate(names):
            if _is_product(label):
                product_indices.append(i)

        n_prod = len(product_indices)
        base_x = float(product_indices[0]) if n_prod else None

        product_rank = {idx: k for k, idx in enumerate(product_indices)}

        for i, label in enumerate(names):
            if _is_product(label) and n_prod:
                k = product_rank[i]
                shift = (k - (n_prod - 1) / 2.0) * float(product_x_offset)
                x_list.append(base_x + shift)
            else:
                x_list.append(float(i))

        return np.array(x_list, dtype=float)

    def _compute_x_from_reference(entries, ref_x_map, ref_prod_xs):
        names = [e[0] for e in entries]
        x_list = []

        prod_labels = [lab for lab in names if _is_product(lab)]
        prod_rank = {lab: k for k, lab in enumerate(prod_labels)}
        n_prod = len(prod_labels)

        if ref_prod_xs:
            base_prod_x = float(np.mean(ref_prod_xs))
        else:
            base_prod_x = float(max(ref_x_map.values())) if ref_x_map else 0.0

        for lab in names:
            key = _norm_label(lab)
            if key in ref_x_map:
                x_list.append(float(ref_x_map[key]))
                continue

            if _is_product(lab) and n_prod:
                k = prod_rank[lab]
                shift = (k - (n_prod - 1) / 2.0) * float(product_x_offset)
                x_list.append(base_prod_x + shift)
                continue

            raise ValueError(
                "Overlay mode: label not found in reference profile: "
                f"{lab!r}. Either add it to the reference profile, or "
                "turn overlay='off'."
            )

        return np.array(x_list, dtype=float)

    def _next_color(ax_):
        return ax_._get_lines.get_next_color()

    def _resolve_colors(profile_name, is_reference, overlay_idx, needs_side: bool):
        if overlay_colors is not None and not is_reference:
            if isinstance(overlay_colors, dict):
                spec = overlay_colors.get(profile_name)
                if spec is not None:
                    if isinstance(spec, (tuple, list)) and len(spec) == 2:
                        return spec[0], spec[1]
                    return spec, spec
    
            elif isinstance(overlay_colors, (list, tuple)):
                if overlay_idx < len(overlay_colors):
                    spec = overlay_colors[overlay_idx]
                    if isinstance(spec, (tuple, list)) and len(spec) == 2:
                        return spec[0], spec[1]
                    return spec, spec
    
        if is_reference:
            return "C0", "C1"
    
        base_idx = 2 * overlay_idx + 1
        main_color = f"C{base_idx % 10}"
        side_color = f"C{(base_idx + 1) % 10}" if needs_side else main_color
        return main_color, side_color

    def _build_energy_map(entries):
        out = {}
        for lab, en, _ in entries:
            out[_norm_label(lab)] = float(en)
        return out

    def _style_meta_for_side_label(label, default_meta, side_metas):
        label_norm = _norm_label(label)
        matches = []
        for meta in side_metas:
            profile_name = meta.get("profile_name")
            if profile_name is None:
                continue
            profile_norm = _norm_label(profile_name)
            if profile_norm and profile_norm in label_norm:
                matches.append((len(profile_norm), meta))

        if not matches:
            return default_meta

        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def _annotate_energy_only(
        ax_,
        xi,
        Ei,
        alpha,
        color,
        placement_counts,
        is_dummy,
    ):
        top_n = placement_counts["top"]
        bottom_n = placement_counts["bottom"]
        left_n = placement_counts["left"]
        right_n = placement_counts["right"]

        dx = 0
        dy = 0
        ha = "center"
        va = "center"

        if left_n:
            dx = -12 * left_n
            ha = "right"
        elif right_n:
            dx = 12 * right_n
            ha = "left"

        if top_n:
            dy = abs(label_offset_up) * top_n
            va = "bottom"
        elif bottom_n:
            dy = -abs(label_offset_down) * bottom_n
            va = "top"

        add_arrow = max(top_n, bottom_n, left_n, right_n) > 1

        text = f"{Ei:.{decimals}f}"

        a = (dummy_alpha if is_dummy else 1.0) * alpha

        arrowprops = None
        if add_arrow:
            arrowprops = {
                "arrowstyle": "->",
                "lw": 0.8,
                "alpha": a * 0.8,
                "shrinkA": 0,
                "shrinkB": 6,
                "mutation_scale": 8,
            }

        ax_.annotate(
            text,
            (xi, Ei),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            alpha=a,
            arrowprops=arrowprops,
            color=color,
            fontsize=energy_fontsize,
        )

    def _plot_one(
        profile_name,
        profile_states,
        ax_,
        is_reference,
        ref_x_map,
        ref_prod_xs,
        ref_energy_map,
        overlay_idx,
    ):
        (
            entries,
            seg_ids,
            side_anchor_label,
            side_connector_rise_frac,
            side_legend_label,
        ) = _parse_entries(
            profile_states
        )

        names = [e[0] for e in entries]
        E = np.array([e[1] for e in entries], dtype=float)

        if is_reference or not ref_x_map:
            x = _compute_x_single(entries)
        else:
            x = _compute_x_from_reference(entries, ref_x_map, ref_prod_xs)

        profile_energy_map = _build_energy_map(entries)

        product_indices = [i for i, lab in enumerate(names) if _is_product(lab)]
        main_product_idx = product_indices[0] if product_indices else (len(entries) - 1)
        side_product_idx = (
            product_indices[1] if len(product_indices) >= 2 else main_product_idx
        )

        side_start_idx = None
        for i, sid in enumerate(seg_ids):
            if sid == 1:
                side_start_idx = i
                break

        main_color, side_color = _resolve_colors(
            profile_name,
            is_reference,
            overlay_idx,
            side_start_idx is not None,
        )        
        a = 1.0 if is_reference else float(overlay_alpha)
        lw = (1.5 * float(overlay_lw_scale)) if not is_reference else 1.5
        z_line = 5 if is_reference else 3
        z_scatter = 6 if is_reference else 4
        z_conn = 2.5 if is_reference else 2.0
        legend_marker = marker if is_reference else (
            overlay_markers.get(profile_name, marker)
            if isinstance(overlay_markers, dict)
            else marker
        )
        side_legend_meta = (
            {
                "profile_name": None if profile_name is None else str(profile_name),
                "label": str(side_legend_label) if side_legend_label is not None else None,
                "color": side_color,
                "alpha": a,
                "marker": legend_marker,
            }
            if side_start_idx is not None
            else None
        )

        if side_start_idx is None:
            x_i, E_i = _dedup_for_interp(x, E)
            xs = np.linspace(x_i.min(), x_i.max(), int(n_points))
            interp = PchipInterpolator(x_i, E_i)
            Es = interp(xs)

            ax_.plot(
                xs,
                Es,
                marker="",
                alpha=a,
                linewidth=lw,
                color=main_color,
                zorder=z_line
            )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )
            ax_.scatter(
                x,
                E,
                zorder=z_scatter,
                color=main_color,
                alpha=a,
                marker=m,
                s=30,
            )
        else:
            if side_start_idx == 0:
                raise ValueError(f"{side_token!r} cannot be the first entry.")

            main_end = side_start_idx - 1

            x_main = x[: main_end + 1]
            E_main = E[: main_end + 1]
            x_main_i, E_main_i = _dedup_for_interp(x_main, E_main)

            xs_main = np.linspace(
                x_main_i.min(), x_main_i.max(), max(2, int(n_points * 0.6))
            )
            interp_main = PchipInterpolator(x_main_i, E_main_i)
            Es_main = interp_main(xs_main)

            ax_.plot(
                xs_main,
                Es_main,
                marker="",
                alpha=a,
                linewidth=lw,
                color=main_color,
                zorder=z_line
            )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )            
            ax_.scatter(
                x_main,
                E_main,
                zorder=z_scatter,
                color=main_color,
                alpha=a,
                marker=m,
                s=30,
            )

            side_anchor_idx = main_end
            if side_anchor_label is not None:
                target = side_anchor_label.lower().strip()
                for j, (lab, _, _) in enumerate(entries):
                    if _norm_label(lab) == target:
                        side_anchor_idx = j
                        break
                else:
                    raise ValueError(
                        f"side-rxn anchor {side_anchor_label!r} not found among labels."
                    )

            if side_anchor_idx >= side_start_idx:
                raise ValueError(
                    f"side-rxn anchor {side_anchor_label!r} must be before side segment."
                )

            side_idxs = [i for i in range(side_start_idx, len(entries))]
            if main_product_idx in side_idxs and main_product_idx != side_product_idx:
                side_idxs = [i for i in side_idxs if i != main_product_idx]

            if side_product_idx not in side_idxs and side_product_idx >= side_start_idx:
                side_idxs.append(side_product_idx)
                side_idxs = sorted(set(side_idxs))

            x_side_main = x[side_idxs]
            E_side_main = E[side_idxs]
            x_side_i, E_side_i = _dedup_for_interp(x_side_main, E_side_main)

            xs_side = np.linspace(
                float(x_side_i.min()),
                float(x_side_i.max()),
                max(2, int(n_points * 0.6)),
            )
            interp_side = PchipInterpolator(x_side_i, E_side_i)
            Es_side = interp_side(xs_side)

            ax_.plot(
                xs_side,
                Es_side,
                marker="",
                alpha=a,
                linewidth=lw,
                color=side_color,
                zorder=z_line
            )
            m = marker if is_reference else (
                overlay_markers.get(profile_name, marker)
                if isinstance(overlay_markers, dict)
                else marker
            )            
            ax_.scatter(
                x_side_main,
                E_side_main,
                zorder=z_scatter,
                color=side_color,
                alpha=a,
                marker=m,
                s=30,
            )

            x0 = float(x[side_anchor_idx])
            y0 = float(E[side_anchor_idx])
            x1c = float(x[side_start_idx])
            y1c = float(E[side_start_idx])

            frac = (
                0.0
                if side_connector_rise_frac is None
                else float(side_connector_rise_frac)
            )
            frac = min(max(frac, 0.0), 1.0)

            x_rise = x0 + frac * (x1c - x0)

            xs_flat = np.linspace(x0, x_rise, 60, endpoint=False)
            ys_flat = np.full_like(xs_flat, y0, dtype=float)

            xs_rise = np.linspace(x_rise, x1c, 120)
            denom = (x1c - x_rise)
            if denom == 0:
                ys_rise = np.full_like(xs_rise, y1c, dtype=float)
            else:
                t = (xs_rise - x_rise) / denom
                t = np.clip(t, 0.0, 1.0)
                s = t * t * (3.0 - 2.0 * t)
                ys_rise = y0 + (y1c - y0) * s

            xs_conn = np.concatenate([xs_flat, xs_rise])
            ys_conn = np.concatenate([ys_flat, ys_rise])

            ax_.plot(
                xs_conn,
                ys_conn,
                linestyle=":",
                linewidth=3.0,
                alpha=a,
                marker="",
                color=side_color,
                zorder=z_conn
            )

            if show_main_to_product and len(x) >= 2:
                x0u = float(x[main_end])
                y0u = float(E[main_end])
                x1u = float(x[main_product_idx])
                y1u = float(E[main_product_idx])

                frac = min(max(float(main_to_product_drop_frac), 0.0), 1.0)
                x_drop = x0u + frac * (x1u - x0u)

                n_flat = (
                    int(main_to_product_flat_points)
                    if main_to_product_flat_points is not None
                    else max(20, int(n_points * 0.15))
                )
                n_drop = (
                    int(main_to_product_drop_points)
                    if main_to_product_drop_points is not None
                    else max(80, int(n_points * 0.35))
                )

                xs_flat = np.linspace(x0u, x_drop, max(2, n_flat), endpoint=False)
                ys_flat = np.full_like(xs_flat, y0u, dtype=float)

                xs_drop = np.linspace(x_drop, x1u, max(2, n_drop))
                denom = (x1u - x_drop)
                if denom == 0:
                    ys_drop = np.full_like(xs_drop, y1u, dtype=float)
                else:
                    t = (xs_drop - x_drop) / denom
                    t = np.clip(t, 0.0, 1.0)
                    s = t * t * (3.0 - 2.0 * t)
                    ys_drop = y0u + (y1u - y0u) * s

                xs_usual = np.concatenate([xs_flat, xs_drop])
                ys_usual = np.concatenate([ys_flat, ys_drop])

                mp_color = "C0" if is_reference else main_color

                ax_.plot(
                    xs_usual,
                    ys_usual,
                    linestyle=main_to_product_linestyle,
                    linewidth=main_to_product_lw,
                    alpha=main_to_product_alpha * a,
                    marker="",
                    color=mp_color,
                    zorder=z_conn
                )
                m = marker if is_reference else (
                    overlay_markers.get(profile_name, marker)
                    if isinstance(overlay_markers, dict)
                    else marker
                )                
                ax_.scatter(
                    [x1u],
                    [y1u],
                    zorder=z_scatter,
                    color=mp_color,
                    alpha=a,
                    marker=m,
                    s=30,
                )

        # --- Energy annotations (labels are handled on x-axis if enabled) ---
        # --- Annotations ---
        if is_reference:
            do_annotate = bool(annotate_energies)
        else:
            do_annotate = overlay_annotate in {"energy", "full"}

        if do_annotate:
            for i, (xi, Ei, label) in enumerate(zip(x, E, names), start=1):
                key = _norm_label(label)
                is_dummy = dummy_substr.lower() in key

                # keep your "same energy" suppression for overlays
                if not is_reference and ref_energy_map is not None:
                    ref_e = ref_energy_map.get(key)
                    if ref_e is not None and abs(float(Ei) - float(ref_e)) <= float(
                        same_energy_tol
                    ):
                        continue

                placement_counts = _parse_placement(entries[i - 1][2])
                if placement_counts is None:
                    is_int = key.startswith(int_prefix.lower())
                    if i == 1:
                        placement_counts = {"left": 1, "right": 0, "top": 0, "bottom": 0}
                    elif i == len(entries):
                        placement_counts = {"right": 1, "left": 0, "top": 0, "bottom": 0}
                    elif is_int:
                        placement_counts = {"bottom": 1, "top": 0, "left": 0, "right": 0}
                    else:
                        placement_counts = {"top": 1, "bottom": 0, "left": 0, "right": 0}

                in_side = seg_ids[i - 1] == 1
                txt_color = side_color if in_side else main_color
                alpha = 1.0 if is_reference else float(overlay_alpha)

                if not multi:
                    # SINGLE-MOLECULE: restore original label+energy annotations
                    top_n = placement_counts["top"]
                    bottom_n = placement_counts["bottom"]
                    left_n = placement_counts["left"]
                    right_n = placement_counts["right"]

                    dx = 0
                    dy = 0
                    ha = "center"
                    va = "center"

                    if left_n:
                        dx = -12 * left_n
                        ha = "right"
                    elif right_n:
                        dx = 12 * right_n
                        ha = "left"

                    if top_n:
                        dy = abs(label_offset_up) * top_n
                        va = "bottom"
                    elif bottom_n:
                        dy = -abs(label_offset_down) * bottom_n
                        va = "top"

                    add_arrow = max(top_n, bottom_n, left_n, right_n) > 1

                    if annotate_energies:
                        text = f"{label}\n{Ei:.{decimals}f}"
                    else:
                        text = f"{label}"

                    a = (dummy_alpha if is_dummy else 1.0) * alpha

                    arrowprops = None
                    if add_arrow:
                        arrowprops = {
                            "arrowstyle": "->",
                            "lw": 0.8,
                            "alpha": a * 0.8,
                            "shrinkA": 0,
                            "shrinkB": 6,
                            "mutation_scale": 8,
                        }

                    ax_.annotate(
                        text,
                        (xi, Ei),
                        textcoords="offset points",
                        xytext=(dx, dy),
                        ha=ha,
                        va=va,
                        alpha=a,
                        arrowprops=arrowprops,
                        color=txt_color,
                        fontsize=energy_fontsize,
                    )
                else:
                    # MULTI-MOLECULE: energy-only (state names are on x-axis)
                    _annotate_energy_only(
                        ax_=ax_,
                        xi=float(xi),
                        Ei=float(Ei),
                        alpha=alpha,
                        color=txt_color,
                        placement_counts=placement_counts,
                        is_dummy=is_dummy,
                    )

        if is_reference:
            x_map = {}
            prod_xs = []
            ordered = []
            for xi, lab in zip(x, names):
                k = _norm_label(lab)
                x_map[k] = float(xi)
                ordered.append((float(xi), str(lab)))
                if _is_product(lab):
                    prod_xs.append(float(xi))
            return x_map, prod_xs, profile_energy_map, ordered, side_legend_meta

        return None, None, profile_energy_map, None, side_legend_meta

    # ---- Detect multi-profile input (no breaking of current list input) ----
    multi = False
    profiles = None

    if isinstance(states, dict):
        profiles = list(states.items())
        multi = True
    elif isinstance(states, (list, tuple)) and states:
        first = states[0]
        if (
            isinstance(first, (list, tuple))
            and len(first) == 2
            and isinstance(first[0], str)
            and isinstance(first[1], (list, tuple))
        ):
            profiles = list(states)
            multi = True

    if overlay == "off":
        multi = False
        profiles = None
    elif overlay == "on":
        if not multi:
            raise ValueError("overlay='on' requires dict or list-of-(name, states).")

    if show_state_labels is None:
        show_state_labels = bool(multi)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ref_x_map = {}
    ref_prod_xs = []
    ref_energy_map = None
    ref_ordered = None
    side_legend_metas = []

    if not multi:
        _, _, _, _, side_meta = _plot_one(
            profile_name=None,
            profile_states=states,
            ax_=ax,
            is_reference=True,
            ref_x_map=ref_x_map,
            ref_prod_xs=ref_prod_xs,
            ref_energy_map=None,
            overlay_idx=0,
        )
        if side_meta is not None:
            side_legend_metas.append(side_meta)
    else:
        ref_name, ref_states = profiles[0]
        ref_x_map, ref_prod_xs, ref_energy_map, ref_ordered, side_meta = _plot_one(
            profile_name=ref_name,
            profile_states=ref_states,
            ax_=ax,
            is_reference=True,
            ref_x_map=ref_x_map,
            ref_prod_xs=ref_prod_xs,
            ref_energy_map=None,
            overlay_idx=0,
        )
        if side_meta is not None:
            side_legend_metas.append(side_meta)

        overlay_energy_maps = []
        for k, (name, st) in enumerate(profiles[1:], start=0):
            _, _, e_map, _, side_meta = _plot_one(
                profile_name=name,
                profile_states=st,
                ax_=ax,
                is_reference=False,
                ref_x_map=ref_x_map,
                ref_prod_xs=ref_prod_xs,
                ref_energy_map=ref_energy_map,
                overlay_idx=k,
            )
            overlay_energy_maps.append(e_map)
            if side_meta is not None:
                side_legend_metas.append(side_meta)

        if same_energy_mode == "tag" and annotate_energies and ref_energy_map is not None:
            for key, ref_e in ref_energy_map.items():
                matched = False
                for om in overlay_energy_maps:
                    oe = om.get(key) if om is not None else None
                    if oe is None:
                        continue
                    if abs(float(oe) - float(ref_e)) <= float(same_energy_tol):
                        matched = True
                        break
                if not matched:
                    continue

                xi = float(ref_x_map[key])
                yi = float(ref_e)
                ax.annotate(
                    same_energy_tag,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(8, 0),
                    ha="left",
                    va="center",
                    alpha=1.0,
                    color="C0",
                    fontsize=same_energy_tag_fontsize,
                )

        if show_legend:
            handles = []
            labels = []
            for i, (name, _) in enumerate(profiles):
                if name is None:
                    continue
                if i == 0:
                    color = "C0"
                    a = 1.0
                else:
                    if isinstance(overlay_colors, dict) and name in overlay_colors:
                        spec = overlay_colors[name]
                        color = spec[0] if isinstance(spec, (tuple, list)) else spec
                    else:
                        color = f"C{i}"
                    a = overlay_alpha
                if i == 0:
                    m = marker
                else:
                    if isinstance(overlay_markers, dict):
                        m = overlay_markers.get(name, marker)
                    else:
                        m = marker

                h = plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    alpha=a,
                    marker=m,
                    linestyle="-",
                )
                handles.append(h)
                labels.append(str(name))
            for meta in side_legend_metas:
                label = meta["label"]
                if label is None:
                    continue

                style_meta = _style_meta_for_side_label(
                    label,
                    meta,
                    side_legend_metas,
                )
                h = plt.Line2D(
                    [0],
                    [0],
                    color=style_meta["color"],
                    alpha=style_meta["alpha"],
                    marker=style_meta["marker"],
                    linestyle="-",
                )
                handles.append(h)
                labels.append(label)
            if handles:
                ax.legend(handles, labels, frameon=False, fontsize=legend_fontsize)

    if not multi and show_legend:
        handles = []
        labels = []

        if profile_label is not None:
            h = plt.Line2D(
                [0],
                [0],
                color="C0",
                alpha=1.0,
                marker=marker,
                linestyle="-",
            )
            handles.append(h)
            labels.append(str(profile_label))

        for meta in side_legend_metas:
            label = meta["label"]
            if label is None:
                continue

            h = plt.Line2D(
                [0],
                [0],
                color=meta["color"],
                alpha=meta["alpha"],
                marker=meta["marker"],
                linestyle="-",
            )
            handles.append(h)
            labels.append(label)

        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=legend_fontsize)

    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # --- Bottom labels (states) ---
    if show_state_labels:
        # Use reference ordering if available (multi); otherwise derive from single.
        if ref_ordered is None:
            entries, _, _, _, _ = _parse_entries(states)
            x_single = _compute_x_single(entries)
            ref_ordered = [(float(xi), str(lab)) for xi, (lab, _, _) in zip(x_single, entries)]

        # If there are duplicated x (multiple products), matplotlib will still
        # accept them; labels may overlap, but the offsets typically separate them.
        xs = [p[0] for p in ref_ordered]
        labs = [p[1] for p in ref_ordered]

        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=state_label_rotation,
                           fontsize=state_label_fontsize)
        ax.tick_params(axis="x", pad=state_label_pad)

        hide_x_ticks = False

    # --- Limits ---
    if ref_x_map:
        xmin = min(ref_x_map.values())
        xmax = max(ref_x_map.values())
    else:
        xmin, xmax = ax.get_xlim()

    left_pad = 1.05
    right_pad = 0.8
    ax.set_xlim(xmin - left_pad, xmax + right_pad)

    ax.grid(bool(grid))
    ax.set_facecolor("white")

    if hide_x_ticks:
        ax.set_xticks([])
    elif not show_state_labels:
        ax.tick_params(axis="x", labelsize=tick_label_fontsize)

    if hide_y_ticks:
        ax.set_yticks([])
    else:
        ax.tick_params(axis="y", labelsize=tick_label_fontsize)

    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if created_fig:
        fig.tight_layout()

    return fig, ax
