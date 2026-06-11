import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tooltoad.chemutils import ac2xyz

import frust.vis as _frust_vis


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


def write_xyz_structures(
    df: pd.DataFrame,
    path: Path | str,
    coord_options: Mapping[str, str],
    name_col: str = "custom_name",
    atoms_col: str = "atoms",
    show_mols: bool = False,
    **molto3d_kwargs,
) -> None:
    """Write XYZ structure files from coordinate columns in a dataframe.

    .. deprecated::
        Use :func:`write_xyz` instead.

    For each ``coord_options`` entry, the key is used as both the output
    subdirectory name and filename suffix. For example,
    ``{"dft": "dft_coords", "xtb": "xtb_coords"}`` writes files such as
    ``path / "dft" / "{name}_dft.xyz"`` and
    ``path / "xtb" / "{name}_xtb.xyz"``.

    Parameters
    ----------
    df
        Dataframe containing one structure per row. It must include
        ``name_col``, ``atoms_col``, and every coordinate column named in
        ``coord_options``.
    path
        Base directory where output subdirectories are created.
    coord_options
        Mapping from output subdirectory/suffix to coordinate column name.
    name_col
        Column containing the base structure name used in output filenames.
    atoms_col
        Column containing atomic symbols compatible with
        :func:`tooltoad.chemutils.ac2xyz`.
    show_mols
        If ``True``, display all written structures with
        :func:`frust.vis.MolTo3DGrid` after writing. Connectivity is inferred
        with RDKit for structures that can be parsed from the generated XYZ.
    **molto3d_kwargs
        Additional keyword arguments passed to :func:`frust.vis.MolTo3DGrid`
        when ``show_mols=True``.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a required dataframe column is missing.
    OSError
        If an output directory or XYZ file cannot be created.
    """
    warnings.warn(
        "`write_xyz_structures` is deprecated; use "
        "`write_xyz(..., coords_col=...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    write_xyz(
        df,
        path,
        coords_col=coord_options,
        name_col=name_col,
        atoms_col=atoms_col,
        show_mols=show_mols,
        **molto3d_kwargs,
    )


def write_xyz(
    df: pd.DataFrame,
    path: Path | str,
    coords_col: str | Mapping[str, str] | None = None,
    name_col: str | None = None,
    atoms_col: str = "atoms",
    show_mols: bool = False,
    overwrite: bool = True,
    **molto3d_kwargs,
) -> list[Path]:
    """Write XYZ structure files from a dataframe.

    Parameters
    ----------
    df
        Dataframe containing one structure per row.
    path
        Output directory. For single-column export, XYZ files are written
        directly into this directory. For mapping export, one subdirectory is
        created per mapping key.
    coords_col
        Coordinate column to export. If ``None``, the latest coordinate-like
        column is selected from the dataframe. A mapping such as
        ``{"DFT": "dft_coords", "xTB": "xtb_coords"}`` exports multiple
        coordinate columns into labelled subdirectories.
    name_col
        Column containing output filename stems. If ``None``, the first
        available column from ``custom_name``, ``ligand_name``, and
        ``substrate_name`` is used. If none are present, dataframe indices are
        used.
    atoms_col
        Column containing atomic symbols compatible with
        :func:`tooltoad.chemutils.ac2xyz`.
    show_mols
        If ``True``, display written structures with
        :func:`frust.vis.MolTo3DGrid`.
    overwrite
        If ``False``, raise :class:`FileExistsError` when an output path
        already exists.
    **molto3d_kwargs
        Additional keyword arguments passed to :func:`frust.vis.MolTo3DGrid`
        when ``show_mols=True``.

    Returns
    -------
    list[pathlib.Path]
        Paths to all written XYZ files.

    Raises
    ------
    KeyError
        If a required dataframe column is missing.
    ValueError
        If no coordinate column can be inferred, a requested coordinate column
        is missing, or a row contains missing/empty atoms or coordinates.
    FileExistsError
        If ``overwrite=False`` and an output file already exists.

    Examples
    --------
    Export the latest optimized geometry from each row:

    >>> paths = write_xyz(df, "structures/products")

    Export a specific coordinate column:

    >>> paths = write_xyz(df, "structures/products", coords_col="orca-opt_coords")

    Export multiple coordinate columns into labelled subdirectories:

    >>> paths = write_xyz(
    ...     df,
    ...     "structures/products",
    ...     coords_col={"DFT": "dft-opt_coords", "xTB": "xtb-opt_coords"},
    ... )
    """
    path = Path(path)
    _validate_columns(df, [atoms_col])
    resolved_name_col = _resolve_name_col(df, name_col)
    coord_map = _resolve_coord_map(df, coords_col)
    mapping_export = isinstance(coords_col, Mapping)

    output_dirs = (
        {label: path / _safe_filename(label) for label in coord_map}
        if mapping_export
        else {label: path for label in coord_map}
    )
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    mols = []
    legends = []
    written_paths: list[Path] = []
    seen_stems: defaultdict[tuple[Path, str], int] = defaultdict(int)

    for row_index, row in df.iterrows():
        raw_name = _row_name(row, row_index, resolved_name_col)
        base_name = _safe_filename(raw_name)
        atoms = row[atoms_col]
        _validate_structure_value(atoms, "atoms", row_index, raw_name)

        for label, coord_col in coord_map.items():
            coords = row[coord_col]
            _validate_structure_value(coords, coord_col, row_index, raw_name)
            xyz_str = ac2xyz(atoms, coords)

            output_dir = output_dirs[label]
            label_safe = _safe_filename(label)
            stem = (
                _deduplicate_stem(
                    f"{base_name}_{label_safe}",
                    output_dir,
                    seen_stems,
                )
                if mapping_export
                else _deduplicate_stem(base_name, output_dir, seen_stems)
            )
            xyz_path = output_dir / f"{stem}.xyz"

            if xyz_path.exists() and not overwrite:
                raise FileExistsError(f"XYZ file already exists: {xyz_path}")

            with open(xyz_path, "w") as f:
                f.write(xyz_str)
            written_paths.append(xyz_path)

            if show_mols:
                mol = Chem.MolFromXYZBlock(xyz_str)
                if mol is not None:
                    rdDetermineBonds.DetermineConnectivity(mol)
                    mols.append(mol)
                    legends.append(stem)

    if show_mols and mols:
        _frust_vis.MolTo3DGrid(mols, legends=legends, **molto3d_kwargs)

    return written_paths


def _coordinate_columns(df: pd.DataFrame) -> list[str]:
    """Return coordinate-like columns in dataframe order."""
    return [
        col for col in df.columns
        if (
            "coords" in str(col)
            or str(col).endswith("-oc")
            or str(col).endswith("-opt_coords")
        )
    ]


def _resolve_coord_map(
    df: pd.DataFrame,
    coords_col: str | Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve coordinate export labels to dataframe columns."""
    if coords_col is None:
        coord_cols = _coordinate_columns(df)
        if not coord_cols:
            raise ValueError(
                "No coordinate columns found. Expected a column containing "
                "'coords' or ending in '-oc'/'-opt_coords'."
            )
        col = coord_cols[-1]
        return {"": col}

    if isinstance(coords_col, Mapping):
        coord_map = {str(label): str(col) for label, col in coords_col.items()}
        if not coord_map:
            raise ValueError("coords_col mapping must contain at least one entry")
        _validate_columns(df, coord_map.values())
        return coord_map

    col = str(coords_col)
    _validate_columns(df, [col])
    return {"": col}


def _resolve_name_col(df: pd.DataFrame, name_col: str | None) -> str | None:
    """Resolve the dataframe column used for XYZ filename stems."""
    if name_col is not None:
        _validate_columns(df, [name_col])
        return name_col

    for candidate in ("custom_name", "ligand_name", "substrate_name"):
        if candidate in df.columns:
            return candidate
    return None


def _validate_columns(df: pd.DataFrame, columns) -> None:
    """Raise if any required dataframe column is missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Missing required dataframe column(s): {missing}. "
            f"Available columns: [{available}]"
        )


def _row_name(row: pd.Series, row_index, name_col: str | None) -> str:
    """Return the unsanitized filename stem for a dataframe row."""
    if name_col is None:
        return str(row_index)
    name = row[name_col]
    if _is_missing_or_empty(name):
        return str(row_index)
    return str(name)


def _safe_filename(value: object) -> str:
    """Return a filesystem-safe filename component."""
    text = str(value).strip()
    text = re.sub(r"[\\/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._() -]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    return text or "structure"


def _deduplicate_stem(
    stem: str,
    output_dir: Path,
    seen_stems: defaultdict[tuple[Path, str], int],
) -> str:
    """Append a stable suffix for duplicate stems in one output directory."""
    key = (output_dir, stem)
    seen_stems[key] += 1
    count = seen_stems[key]
    if count == 1:
        return stem
    return f"{stem}_{count}"


def _validate_structure_value(
    value,
    column: str,
    row_index,
    row_name: str,
) -> None:
    """Validate atoms or coordinate values before XYZ conversion."""
    if _is_missing_or_empty(value):
        raise ValueError(
            f"Missing or empty value in column {column!r} for row "
            f"{row_index!r} ({row_name!r})"
        )


def _is_missing_or_empty(value) -> bool:
    """Return True for None, NaN-like, or empty array/list values."""
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0 or bool(pd.isna(value).all())
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False
