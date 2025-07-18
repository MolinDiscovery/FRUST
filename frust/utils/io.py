import pandas as pd
from pathlib import Path
import re

def dump_df(df: pd.DataFrame, step: str, base_dir: Path) -> Path:
    """
    If dump_each_step is True, writes DataFrame to `base_dir/{step}.csv`.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{step}.csv"
    df.to_csv(path, index=False)
    return path


def read_ts_type_from_xyz(xyz_file: str):
    """
    Reads the transition state (TS) type from the comment line of an XYZ file.

    Args:
        xyz_file (str): Path to the XYZ file containing the transition state structure.
            The TS type must be specified in the second line as 'TS' followed by a number
            (e.g., 'TS1 guess', 'TS2').

    Returns:
        str: The transition state type in uppercase (e.g., 'TS1', 'TS2').
    """    
    try:
        with open(xyz_file, 'r') as file:
            file.readline()  # Skip first line
            comment = file.readline()  # Read second line
            
            if not comment: 
                raise ValueError("XYZ file must have at least 2 lines with a comment on the second line")
                
    except FileNotFoundError:
        print(f"Error: Transition state structure file not found: {xyz_file}")
        raise
    except PermissionError:
        print(f"Error: Permission denied when accessing file: {xyz_file}")
        raise
    except IOError as e:
        print(f"Error: Failed to read transition state structure file {xyz_file}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading transition state structure from {xyz_file}: {e}")
        raise

    match = re.search(r'\b(?:TS|INT)\d+\b', comment, re.IGNORECASE)
    if match:
        return match.group().upper()
    else:
        raise ValueError(
            "XYZ file must specify a structure type in the comments. "
            "Please include TSX or INTX in the comment (e.g. TS1, INT3)."
        )