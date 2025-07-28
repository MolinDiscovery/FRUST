from typing import Optional, List, Union, Any
from tooltoad.vis import MolTo3DGrid
from tooltoad.chemutils import ac2mol
import pandas as pd
import numpy as np

def plot_mols(
    df: pd.DataFrame, 
    row_indices: Optional[List[int]] = None,
    ligand_filter: Optional[List[str]] = None,
    rpos_filter: Optional[List[Union[str, int]]] = None,
    exclude_coords: Optional[List[str]] = None,
    include_coords: Optional[List[str]] = None,
    **molto3d_kwargs: Any
) -> None:
    """
    Display molecules from a dataframe with filtering capabilities.
    
    Args:
        df: DataFrame with molecular data
        row_indices: List of row indices to display (if None, displays all rows)
        ligand_filter: List of ligand names to include (if None, includes all)
        rpos_filter: List of rpos values to include (if None, includes all)
        exclude_coords: List of coordinate column patterns to exclude
        include_coords: List of coordinate column patterns to include (overrides exclude)
        **molto3d_kwargs: Additional arguments to pass to MolTo3DGrid
    
    Returns:
        None
    """

    filtered_df = df.copy()
    
    if row_indices is not None:
        filtered_df = filtered_df.iloc[row_indices]
    
    if ligand_filter is not None:
        filtered_df = filtered_df[filtered_df['ligand_name'].isin(ligand_filter)]
    
    if rpos_filter is not None:
        filtered_df = filtered_df[filtered_df['rpos'].isin(rpos_filter)]
    
    if filtered_df.empty:
        print("No molecules match the specified filters.")
        return

    coord_columns = [c for c in df.columns if "coords" in c]
    
    if include_coords is not None:
        coord_columns = [c for c in coord_columns if any(pattern in c for pattern in include_coords)]
    elif exclude_coords is not None:
        coord_columns = [c for c in coord_columns if not any(pattern in c for pattern in exclude_coords)]
    
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
        
            if coords is not None and not (isinstance(coords, np.ndarray) and coords.size == 0) and not pd.isna(coords).all() if isinstance(coords, np.ndarray) else not pd.isna(coords):
                mol = ac2mol(atoms, coords)
                if mol is not None:
                    all_mols.append(mol)
                
                    coord_type = coord_col.replace("coords_", "").replace("_coords", "")
                    if rpos == None:
                        legend = f"{ligand_name}\n{coord_type}"
                    else:
                        legend = f"{ligand_name} r{rpos}\n{coord_type}"
                    all_legends.append(legend)
    
    if not all_mols:
        print("No valid molecules could be generated.")
        return
    
    print(f"Generated {len(all_mols)} molecules for display")
    

    molto3d_args = {
        'legends': all_legends,
        'show_labels': False,
        'show_confs': True,
        'background_color': 'white',
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
    **kwargs: Any
) -> None:
    """Display all coordinate types for a single row."""
    plot_mols(
        df, 
        row_indices=[row_index], 
        exclude_coords=exclude_coords,
        **kwargs
    )

def plot_lig(
    df: pd.DataFrame, 
    ligand_names: Union[str, List[str]], 
    exclude_coords: Optional[List[str]] = None, 
    **kwargs: Any
) -> None:
    """Display molecules filtered by ligand name(s)."""
    if isinstance(ligand_names, str):
        ligand_names = [ligand_names]
    
    plot_mols(
        df, 
        ligand_filter=ligand_names, 
        exclude_coords=exclude_coords,
        **kwargs
    )

def plot_rpos(
    df: pd.DataFrame, 
    rpos_values: Union[str, int, List[Union[str, int]]], 
    exclude_coords: Optional[List[str]] = None, 
    **kwargs: Any
) -> None:
    """Display molecules filtered by rpos value(s)."""
    if isinstance(rpos_values, (str, int)):
        rpos_values = [rpos_values]
    
    plot_mols(
        df, 
        rpos_filter=rpos_values, 
        exclude_coords=exclude_coords,
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
):
    from tooltoad.vis import show_vibs

    vibs_col = [c for c in df.columns if "vibs" in c][-1]
    vibs_col_pre = vibs_col.split("vibs")[0]

    coords_col = vibs_col_pre + "opt_coords"

    atoms = df["atoms"].iloc[row_index]
    vibs = df[vibs_col].iloc[row_index]
    coords = df[coords_col].iloc[row_index]

    view_dict = {
        "atoms": atoms,
        "opt_coords": coords,
        "vibs": vibs,
    }

    view = show_vibs(view_dict, vId, width, height, numFrames, amplitude, transparent, fps, reps)

    return view