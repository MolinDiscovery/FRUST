"""FRUST dataframe adapters for Tooltoad 3D scenes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from tooltoad.scene3d import (
    AtomHighlight,
    AtomLabel,
    DistanceOverlay,
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    SceneCell,
    VibrationAnimation,
)

from frust.schema import latest_opt_coords_column, normalize_dataframe


def show_scene(scene: GridScene):
    """Render a Tooltoad 3D scene.

    Parameters
    ----------
    scene
        Scene to render.

    Returns
    -------
    py3Dmol.view
        Rendered viewer.
    """

    return Py3DmolGridRenderer(scene).show()


def molecule_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int] | None = None,
    substrate_filter: Sequence[str] | None = None,
    rpos_filter: Sequence[str | int] | None = None,
    exclude_coords: Sequence[str] | None = None,
    include_coords: Sequence[str] | None = None,
    coord_indices: Sequence[int] | slice | None = slice(-1, None),
    columns: int | None = None,
    cell_size: tuple[int, int] = (400, 400),
    linked: bool = False,
    background_color: str | tuple[str, float] = ("blue", 0.1),
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    style: dict[str, Any] | None = None,
) -> GridScene:
    """Create a molecule scene from a FRUST dataframe.

    Parameters
    ----------
    df
        Dataframe with ``atoms`` and coordinate columns.
    row_indices
        Optional positional rows to include.
    substrate_filter
        Optional substrate names to include.
    rpos_filter
        Optional reactive positions to include.
    exclude_coords
        Coordinate column name fragments to exclude.
    include_coords
        Coordinate column name fragments to include.
    coord_indices
        Positional coordinate-column selection.
    columns
        Grid columns. When omitted, preserve ``plot_mols`` defaults.
    cell_size
        Grid cell size in pixels.
    linked
        Link viewer motion across cells.
    background_color
        py3Dmol background color or ``(color, opacity)``.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw formal charge labels where possible.
    kekulize
        Kekulize generated mol blocks when explicit connectivity is present.
    style
        py3Dmol model style.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene ready for rendering.
    """

    filtered_df = _filter_dataframe(
        df,
        row_indices=row_indices,
        substrate_filter=substrate_filter,
        rpos_filter=rpos_filter,
    )
    coord_columns = _select_coordinate_columns(
        filtered_df,
        exclude_coords=exclude_coords,
        include_coords=include_coords,
        coord_indices=coord_indices,
    )

    cells: list[SceneCell] = []
    for _, row in filtered_df.iterrows():
        atoms = row["atoms"]
        bonds = _optional_bonds(row)
        for coord_col in coord_columns:
            coords = row[coord_col]
            if not _valid_coords(coords):
                continue
            cells.append(
                SceneCell(
                    title=_row_coord_title(row, coord_col),
                    models=[
                        MoleculeModel(
                            atoms=atoms,
                            coords=coords,
                            bonds=bonds,
                            show_atom_labels=show_labels,
                            show_charges=show_charges,
                            kekulize=kekulize,
                            style=style,
                        )
                    ],
                )
            )

    if not cells:
        raise ValueError("No valid molecules could be generated for display.")

    if columns is None:
        columns = len(coord_columns) if coord_indices is None else 4

    return GridScene(
        cells=cells,
        columns=max(1, int(columns)),
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
    )


def vibration_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    row_index: int = 0,
    row_indices: Sequence[int] | str | None = None,
    max_rows: int | None = None,
    vId: int = 0,
    custom_coords_col_name: str | None = None,
    columns: int | None = None,
    viewergrid: tuple[int, int] | None = None,
    width: float = 600,
    height: float = 400,
    cell_size: tuple[int, int] | None = None,
    numFrames: int = 20,
    amplitude: float = 1.0,
    transparent: bool = True,
    fps: float | None = None,
    reps: int = 50,
    linked: bool = True,
    freq_label: bool = True,
    legends: Sequence[str] | None = None,
    background_color: str | None = None,
) -> GridScene:
    """Create an animated vibration scene from a FRUST dataframe."""

    vibs_col = select_vibration_column(df)
    coords_col = select_vibration_coords_column(
        df,
        vibs_col,
        custom_coords_col_name=custom_coords_col_name,
    )
    indices = _vibration_row_indices(
        df,
        row_index=row_index,
        row_indices=row_indices,
        max_rows=max_rows,
    )
    if legends is not None and len(legends) != len(indices):
        raise ValueError("Length of legends must match selected vibration rows.")

    cells: list[SceneCell] = []
    for display_idx, row_pos in enumerate(indices):
        row = df.iloc[row_pos]
        vibs = row[vibs_col]
        if missing_vibrations(vibs):
            raise ValueError(f"Vibration column {vibs_col!r} is missing for row {row_pos}")
        vib = vibs[vId]
        frequency = float(vib["frequency"])
        title = None
        if freq_label:
            title = legends[display_idx] if legends else _vibration_title(row, frequency)
        cells.append(
            SceneCell(
                title=title,
                models=[
                    MoleculeModel(
                        atoms=row["atoms"],
                        coords=row[coords_col],
                        bonds=_optional_bonds(row),
                        style={"sphere": {"radius": 0.4}, "stick": {}},
                    )
                ],
                animations=[
                    VibrationAnimation(
                        mode=vib["mode"],
                        frequency=frequency,
                        num_frames=numFrames,
                        amplitude=amplitude,
                        fps=fps,
                        reps=reps,
                    )
                ],
            )
        )

    if viewergrid is not None:
        rows, cols = viewergrid
        scene_columns = cols
        scene_cell_size = (
            max(1, int(width / cols)),
            max(1, int(height / rows)),
        )
    elif columns is not None:
        scene_columns = int(columns)
        scene_cell_size = cell_size or (400, 400)
    else:
        scene_columns = 1 if len(cells) == 1 else math.ceil(math.sqrt(len(cells)))
        rows = math.ceil(len(cells) / scene_columns)
        scene_cell_size = cell_size or (
            max(1, int(width / scene_columns)),
            max(1, int(height / rows)),
        )

    bg = background_color or "0xeeeeee"
    return GridScene(
        cells=cells,
        columns=max(1, scene_columns),
        cell_size=scene_cell_size,
        linked=linked,
        background_color=bg,
        transparent=transparent,
    )


def ts_guess_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int] | None = None,
    coords_col: str = "coords_embedded",
    show_roles: bool = True,
    show_constraint_distances: bool = False,
    columns: int = 2,
    cell_size: tuple[int, int] = (400, 400),
    linked: bool = False,
) -> GridScene:
    """Create a TS-guess scene with optional role and constraint overlays."""

    rows = df.iloc[list(row_indices)] if row_indices is not None else df
    cells: list[SceneCell] = []
    for _, row in rows.iterrows():
        overlays = []
        roles = row.get("constraint_roles", {})
        if show_roles and isinstance(roles, Mapping):
            for role, atom_idx in roles.items():
                overlays.extend(
                    [
                        AtomHighlight(atom=int(atom_idx), color="cyan", radius=0.5, alpha=0.35),
                        AtomLabel(atom=int(atom_idx), text=str(role)),
                    ]
                )
        if show_constraint_distances and isinstance(roles, Mapping):
            overlays.extend(_constraint_distance_overlays(row, roles))

        cells.append(
            SceneCell(
                title=_row_coord_title(row, coords_col),
                models=[
                    MoleculeModel(
                        atoms=row["atoms"],
                        coords=row[coords_col],
                        bonds=_optional_bonds(row),
                    )
                ],
                overlays=overlays,
            )
        )

    return GridScene(
        cells=cells,
        columns=columns,
        cell_size=cell_size,
        linked=linked,
    )


def select_vibration_column(df: pd.DataFrame) -> str:
    """Choose the latest non-missing vibration column in dataframe order."""

    vibs_cols = vibration_columns(df)
    if not vibs_cols:
        raise ValueError("No vibration columns found. Expected a column ending in '-vibs'.")
    non_missing_cols = [
        col
        for col in vibs_cols
        if df[col].map(lambda value: not missing_vibrations(value)).any()
    ]
    return (non_missing_cols or vibs_cols)[-1]


def select_vibration_coords_column(
    df: pd.DataFrame,
    vibs_col: str,
    *,
    custom_coords_col_name: str | None = None,
) -> str:
    """Choose the coordinate column that best matches a vibration column."""

    if custom_coords_col_name:
        if custom_coords_col_name not in df.columns:
            available = ", ".join(coordinate_columns(df))
            raise KeyError(
                f"Coordinate column {custom_coords_col_name!r} is not present. "
                f"Available coordinate columns: [{available}]"
            )
        return custom_coords_col_name

    vibs_prefix = str(vibs_col).removesuffix("vibs")
    matching_col = latest_opt_coords_column(vibs_prefix, df)
    if matching_col is not None:
        return matching_col

    coord_cols = coordinate_columns(df)
    if not coord_cols:
        raise ValueError("No coordinate columns found for vibration display.")

    column_positions = {str(col): idx for idx, col in enumerate(df.columns)}
    vibs_position = column_positions[str(vibs_col)]
    preceding = [col for col in coord_cols if column_positions[str(col)] < vibs_position]
    return (preceding or coord_cols)[-1]


def vibration_columns(df: pd.DataFrame) -> list[str]:
    """Return vibration-like columns in dataframe order."""

    return [
        str(col)
        for col in df.columns
        if str(col).lower() == "vibs" or str(col).lower().endswith("-vibs")
    ]


def coordinate_columns(df: pd.DataFrame) -> list[str]:
    """Return coordinate-like columns in dataframe order."""

    return [
        str(col)
        for col in df.columns
        if (
            "coords" in str(col)
            or str(col).endswith("-oc")
            or str(col).endswith("-opt_coords")
        )
    ]


def missing_vibrations(vibs: Any) -> bool:
    """Return whether a vibrations cell is missing."""

    if vibs is None:
        return True
    try:
        missing = pd.isna(vibs)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _filter_dataframe(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int] | None,
    substrate_filter: Sequence[str] | None,
    rpos_filter: Sequence[str | int] | None,
) -> pd.DataFrame:
    filtered = normalize_dataframe(df)
    if row_indices is not None:
        filtered = filtered.iloc[list(row_indices)]
    if substrate_filter is not None:
        filtered = filtered[filtered["substrate_name"].isin(substrate_filter)]
    if rpos_filter is not None:
        filtered = filtered[filtered["rpos"].isin(rpos_filter)]
    if filtered.empty:
        raise ValueError("No molecules match the specified filters.")
    return filtered


def _select_coordinate_columns(
    df: pd.DataFrame,
    *,
    exclude_coords: Sequence[str] | None,
    include_coords: Sequence[str] | None,
    coord_indices: Sequence[int] | slice | None,
) -> list[str]:
    coord_cols = coordinate_columns(df)
    if coord_indices is not None:
        if isinstance(coord_indices, slice):
            coord_cols = coord_cols[coord_indices]
        else:
            coord_cols = [coord_cols[i] for i in coord_indices if 0 <= i < len(coord_cols)]
    elif include_coords is not None:
        coord_cols = [
            col for col in coord_cols if any(pattern in col for pattern in include_coords)
        ]
    elif exclude_coords is not None:
        coord_cols = [
            col for col in coord_cols if not any(pattern in col for pattern in exclude_coords)
        ]
    if not coord_cols:
        raise ValueError("No coordinate columns found after filtering.")
    return coord_cols


def _valid_coords(coords: Any) -> bool:
    if coords is None:
        return False
    if isinstance(coords, np.ndarray):
        return coords.size > 0 and not pd.isna(coords).all()
    if isinstance(coords, list):
        return len(coords) > 0
    try:
        return not bool(pd.isna(coords))
    except (TypeError, ValueError):
        return True


def _optional_bonds(row: pd.Series) -> list[tuple[int, int]] | None:
    bonds = row.get("connectivity_bonds")
    if isinstance(bonds, list) and bonds:
        return [(int(begin), int(end)) for begin, end in bonds]
    return None


def _row_coord_title(row: pd.Series, coord_col: str) -> str:
    substrate_name = row.get("substrate_name", row.get("custom_name", "molecule"))
    rpos = row.get("rpos")
    coord_type = str(coord_col).replace("coords_", "").replace("_coords", "")
    if rpos is None or pd.isna(rpos):
        return f"{substrate_name}\n{coord_type}"
    return f"{substrate_name} r{rpos}\n{coord_type}"


def _vibration_title(row: pd.Series, frequency: float) -> str:
    substrate_name = row.get("substrate_name", row.get("custom_name", "molecule"))
    rpos = row.get("rpos")
    if rpos is None or pd.isna(rpos):
        return f"{substrate_name}\n{frequency:.1f} cm^-1"
    return f"{substrate_name} r{rpos}\n{frequency:.1f} cm^-1"


def _vibration_row_indices(
    df: pd.DataFrame,
    *,
    row_index: int,
    row_indices: Sequence[int] | str | None,
    max_rows: int | None,
) -> list[int]:
    if row_indices is None:
        indices = [int(row_index)]
    elif row_indices == "all":
        indices = list(range(len(df)))
    elif isinstance(row_indices, Iterable) and not isinstance(row_indices, (str, bytes)):
        indices = [int(idx) for idx in row_indices]
    else:
        raise ValueError("row_indices must be None, 'all', or a sequence of row indices.")

    if max_rows is not None:
        indices = indices[: int(max_rows)]
    return indices


def _constraint_distance_overlays(
    row: pd.Series,
    roles: Mapping[str, Any],
) -> list[DistanceOverlay]:
    spec = row.get("constraint_spec", [])
    if isinstance(spec, Mapping):
        spec = [spec]
    overlays = []
    for entry in spec:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("kind", "")).lower() != "distance":
            continue
        entry_roles = entry.get("roles", [])
        if not isinstance(entry_roles, Sequence) or len(entry_roles) != 2:
            continue
        role1, role2 = str(entry_roles[0]), str(entry_roles[1])
        if role1 not in roles or role2 not in roles:
            continue
        overlays.append(
            DistanceOverlay(
                atom1=int(roles[role1]),
                atom2=int(roles[role2]),
                label=f"{role1}-{role2}",
            )
        )
    return overlays
