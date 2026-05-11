from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from tooltoad.chemutils import ac2mol
from tooltoad.vis import DrawMolSvg, MolTo3DGrid, RxnTo3DGrid

from frust.schema import normalize_dataframe


def plot_mols(
    df: pd.DataFrame,
    row_indices: Optional[List[int]] = None,
    substrate_filter: Optional[List[str]] = None,
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
        substrate_filter: List of substrate names to include (if None, includes all)
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
    filtered_df = normalize_dataframe(df)

    if row_indices is not None:
        filtered_df = filtered_df.iloc[row_indices]

    if substrate_filter is not None:
        filtered_df = filtered_df[
            filtered_df['substrate_name'].isin(substrate_filter)
        ]

    if rpos_filter is not None:
        filtered_df = filtered_df[filtered_df['rpos'].isin(rpos_filter)]

    if filtered_df.empty:
        print("No molecules match the specified filters.")
        return

    coord_columns = [
        c for c in filtered_df.columns
        if "coords" in c or str(c).endswith("-oc") or str(c).endswith("-opt_coords")
    ]

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
        substrate_name = row["substrate_name"]
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
                        if rpos is None or pd.isna(rpos):
                            legend = f"{substrate_name}\n{coord_type}"
                        else:
                            legend = f"{substrate_name} r{rpos}\n{coord_type}"
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
    substrate_names: Union[str, List[str]],
    exclude_coords: Optional[List[str]] = None,
    coord_indices: Optional[Union[List[int], slice]] = slice(-1, None),
    **kwargs: Any
) -> None:
    """Display molecules filtered by substrate name(s)."""
    if isinstance(substrate_names, str):
        substrate_names = [substrate_names]

    plot_mols(
        df,
        substrate_filter=substrate_names,
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
