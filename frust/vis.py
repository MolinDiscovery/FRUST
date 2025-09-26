from typing import Optional, List, Union, Any
from tooltoad.vis import MolTo3DGrid
from tooltoad.chemutils import ac2mol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr
from contextlib import contextmanager, nullcontext

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
    coord_indices: Optional[Union[List[int], slice]] = None,
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
        'background_color': 'black' if darkmode else 'white',
        'cell_size': (400, 400),
        'columns': len(coord_columns),
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
    coord_indices: Optional[Union[List[int], slice]] = None,
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
    coord_indices: Optional[Union[List[int], slice]] = None,
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
    coord_indices: Optional[Union[List[int], slice]] = None,
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
):
    from tooltoad.vis import show_vibs, ac2xyz
    import math
    import py3Dmol

    vibs_col = [c for c in df.columns if "vibs" in c][-1]
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
    x_col: str = "dG",
    y_col: str = "dE",
    xlabel: str = "dG, kcal/mol",
    ylabel: str = "dE, kcal/mol",
    label_col: str = "ligand_name",
    rpos_col: str = "rpos",
    method: str = "spearman",
    num_outliers: int = 2,
    size: tuple = (8, 6)
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

    if method == "pearson":
        data["score"] = (y - y_fit).abs()
    else:
        data["rank_x"] = x.rank()
        data["rank_y"] = y.rank()
        data["score"] = (data["rank_y"] - data["rank_x"]).abs()

    outliers = data.nlargest(num_outliers, "score")

    style_ctx = plt.style.context('dark_background') if darkmode else nullcontext()
    with style_ctx:
        plt.figure(figsize=size)
        plt.scatter(x, y, alpha=0.7)
        plt.plot(
            x, y_fit, color="red",
            label=f"$R^2$={lr.rvalue**2:.3f}, spearman={rho:.3f}"
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
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return None