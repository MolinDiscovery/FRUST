"""Conformer ensemble visualization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tooltoad.scene3d import GridScene, MoleculeModel, Py3DmolGridRenderer, SceneCell

from frust.schema import energy_columns, infer_group_columns, normalize_dataframe
from frust.vis.scenes import (
    DEFAULT_SCENE_BACKGROUND,
    DEFAULT_CELL_SIZE,
    coordinate_columns,
)

HARTREE_TO_KCAL_MOL = 627.5094740631
VALID_CONFORMER_MODES = {"single", "cloud", "representatives+cloud", "cluster"}
VALID_COLOR_MODES = {"energy", "cluster", "uniform"}

CORE_STYLE: dict[str, Any] = {
    "stick": {"radius": 0.18, "color": "black"},
    "sphere": {"radius": 0.32, "color": "black"},
}
CLOUD_STYLE: dict[str, Any] = {
    "stick": {"radius": 0.075, "color": "#4f6f7a", "opacity": 0.55},
    "sphere": {"radius": 0.165, "color": "#4f6f7a", "opacity": 0.55},
}
REPRESENTATIVE_STYLE: dict[str, Any] = {
    "stick": {"radius": 0.11, "color": "#d55e00", "opacity": 0.92},
    "sphere": {"radius": 0.20, "color": "#d55e00", "opacity": 0.92},
}
ENERGY_LOW_COLOR = (0x1F, 0x77, 0xB4)
ENERGY_HIGH_COLOR = (0xD6, 0x27, 0x28)
CLUSTER_COLORS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#9467bd",
    "#8c564b",
)


@dataclass(frozen=True)
class _ConformerRecord:
    row_key: Any
    display_order: int
    row: pd.Series
    coords: np.ndarray
    aligned_coords: np.ndarray
    relative_energy_kcal: float | None
    cluster: int | None = None


def conformer_ensemble_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    row_index: int = 0,
    coords_col: str | None = None,
    energy_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    core_atoms: Sequence[int] | str = "auto",
    mode: str = "representatives+cloud",
    color_by: str | None = None,
    top_n: int | None = 25,
    energy_window_kcal: float | None = None,
    energy_scale_to_kcal: float | None = None,
    conformer_index: int = 0,
    cluster_threshold: float = 0.75,
    n_clusters: int | None = None,
    cell_size: tuple[int, int] = DEFAULT_CELL_SIZE,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    linked: bool = False,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    cloud_opacity: float = 0.55,
    cloud_radius: float = 0.075,
    cloud_color: str = "#4f6f7a",
) -> GridScene:
    """Create a core-aligned scene for one conformer family.

    The selected ``row_index`` identifies a conformer family. FRUST gathers the
    matching rows, aligns every conformer on a fixed atom core, draws that core
    once in black, and overlays the remaining mobile atoms as a conformer cloud.
    Bonds that cross from the fixed core to mobile atoms are drawn as connector
    sticks so the molecule still reads as chemically connected.

    Examples
    --------
    Show the lowest-energy conformer clearly and keep the other selected
    conformers as a background cloud:

    >>> import frust as ft
    >>> scene = ft.vis.conformer_ensemble_scene_from_dataframe(
    ...     ts_guesses["TS1"],
    ...     row_index=0,
    ...     mode="representatives+cloud",
    ...     top_n=25,
    ...     energy_window_kcal=5.0,
    ...     background_color="white",
    ... )
    >>> ft.vis.show_conformer_ensemble_scene(scene)

    Inspect a specific conformer while using explicit core atoms:

    >>> scene = ft.vis.conformer_ensemble_scene_from_dataframe(
    ...     df,
    ...     core_atoms=[0, 1, 2, 7],
    ...     mode="single",
    ...     conformer_index=3,
    ... )

    Use cluster mode when the ensemble contains several distinct mobile
    geometries:

    >>> scene = ft.vis.conformer_ensemble_scene_from_dataframe(
    ...     df,
    ...     mode="cluster",
    ...     n_clusters=4,
    ...     color_by="cluster",
    ... )

    Parameters
    ----------
    df
        FRUST dataframe containing one row per conformer. Required columns are
        ``atoms`` and at least one coordinate column such as
        ``coords_embedded`` or an optimized coordinate column. Optional columns
        used when present are ``connectivity_bonds``, ``constraint_roles``,
        ``constraint_atoms``, and energy columns.
    row_index
        Positional row used to choose the conformer family. This is not a
        pandas index label. The selected row is matched against ``group_cols``
        to find sibling conformers.
    coords_col
        Coordinate column to visualize. If omitted, FRUST uses the latest
        coordinate column in dataframe order.
    energy_col
        Energy column used for sorting, relative-energy coloring, and
        ``energy_window_kcal`` filtering. If omitted, FRUST uses the latest
        known energy column, falling back to ``energy_uff``. If no energy column
        exists, conformers stay in dataframe order and energy coloring falls
        back to ``cloud_color``.
    group_cols
        Columns defining one conformer family. If omitted, FRUST infers stable
        structure identity columns such as system, substrate, catalyst,
        structure type, and reactive position. Pass this explicitly when your
        dataframe uses custom identity columns. Do not include a conformer id
        column such as ``cid`` if each conformer should remain in the same
        family.
    core_atoms
        Atoms used for rigid alignment and drawn once as the fixed black core.
        ``"auto"`` uses atom indices from ``constraint_roles`` first, then
        legacy ``constraint_atoms``. Pass an explicit sequence when the
        automatic core is too small, too large, or chemically ambiguous.
    mode
        Display mode:

        ``"single"``
            Show one selected conformer after filtering and sorting.
        ``"cloud"``
            Show only the mobile-atom cloud plus the fixed core.
        ``"representatives+cloud"``
            Show the mobile-atom cloud and one clear representative, normally
            the lowest-energy conformer.
        ``"cluster"``
            Cluster conformers by mobile-atom RMSD and show one clear
            representative per cluster.
    color_by
        Coloring rule:

        ``"energy"``
            Color conformers from low to high relative energy when an energy
            column is available.
        ``"cluster"``
            Color by cluster label. This is the default for ``mode="cluster"``.
        ``"uniform"``
            Use one cloud color for all non-core conformers. This is useful
            with no energies or when color should not encode data.

        If omitted, cluster mode uses ``"cluster"`` and all other modes use
        ``"energy"``.
    top_n
        Maximum number of conformers to include after sorting. ``None`` keeps
        all conformers passing other filters.
    energy_window_kcal
        Keep only conformers within this relative energy of the family minimum.
        Ignored when no energy column is available.
    energy_scale_to_kcal
        Optional multiplier converting ``energy_col`` values to kcal/mol.
    conformer_index
        Selected conformer position in ``mode="single"`` after filtering and
        sorting.
    cluster_threshold
        RMSD threshold in Angstrom used for hierarchical clustering.
    n_clusters
        Optional maximum cluster count. If supplied, this takes precedence over
        ``cluster_threshold``.
    cell_size
        Width and height of the py3Dmol scene cell in pixels.
    background_color
        py3Dmol background color. Pass a color string such as ``"white"`` or a
        ``(color, opacity)`` tuple such as ``("white", 1.0)``.
    linked
        Link viewer motion if future scene variants contain multiple cells.
    show_labels
        Draw atom labels for displayed models.
    show_charges
        Draw formal charge labels where possible.
    kekulize
        Kekulize generated mol blocks when explicit connectivity is present.
    cloud_opacity
        Opacity for background conformer sticks and spheres. Use a smaller
        value such as ``0.35`` if the cloud is too dense.
    cloud_radius
        Stick radius for background conformers. Sphere radii are scaled from
        this value.
    cloud_color
        Color for ``color_by="uniform"`` and for fallback coloring when no
        energy values are available.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene ready for rendering with
        :func:`show_conformer_ensemble_scene`. The generic ``show_scene``
        helper does not reapply per-model styles for overlaid conformers.

    Notes
    -----
    The cloud intentionally excludes the fixed core atoms. This avoids drawing
    several overlapping reactive cores, which can make constrained structures
    look chemically broken. The connector sticks preserve the visual bond
    relationship between the fixed core and each mobile conformer.
    """

    mode = _validate_mode(mode)
    color_by = _resolve_color_by(color_by, mode)
    if top_n is not None and int(top_n) < 1:
        raise ValueError("top_n must be >= 1 or None.")
    if n_clusters is not None and int(n_clusters) < 1:
        raise ValueError("n_clusters must be >= 1 or None.")
    cloud_opacity = _validate_opacity(cloud_opacity, name="cloud_opacity")
    cloud_radius = _validate_positive_float(cloud_radius, name="cloud_radius")

    normalized = normalize_dataframe(df)
    if normalized.empty:
        raise ValueError("Cannot display conformers from an empty dataframe.")
    row_index = int(row_index)
    if row_index < 0 or row_index >= len(normalized):
        raise IndexError(f"row_index {row_index} is outside dataframe bounds.")

    coords_col = _resolve_coords_col(normalized, coords_col)
    energy_col = _resolve_energy_col(normalized, energy_col)
    group = _select_conformer_group(normalized, row_index=row_index, group_cols=group_cols)
    group = _rows_with_valid_coords(group, coords_col)
    if group.empty:
        raise ValueError(f"No valid coordinates found in {coords_col!r} for the selected group.")

    atoms = list(group.iloc[0]["atoms"])
    bonds = _normalize_group_bonds(group.iloc[0])
    core = _resolve_core_atoms(group.iloc[0], core_atoms=core_atoms, atom_count=len(atoms))
    mobile = [idx for idx in range(len(atoms)) if idx not in core]
    if not mobile:
        raise ValueError("The selected core contains every atom; no mobile atoms remain.")

    records = _build_conformer_records(
        group,
        atoms=atoms,
        coords_col=coords_col,
        energy_col=energy_col,
        core_atoms=core,
        energy_scale_to_kcal=energy_scale_to_kcal,
    )
    records = _filter_conformer_records(
        records,
        top_n=top_n,
        energy_window_kcal=energy_window_kcal,
    )
    if not records:
        raise ValueError("No conformers remain after applying filters.")

    if mode == "cluster":
        records = _with_cluster_labels(
            records,
            atoms=atoms,
            mobile_atoms=mobile,
            threshold=cluster_threshold,
            n_clusters=n_clusters,
        )

    models = _ensemble_models(
        records,
        atoms=atoms,
        bonds=bonds,
        core_atoms=core,
        mobile_atoms=mobile,
        mode=mode,
        color_by=color_by,
        conformer_index=conformer_index,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        cloud_opacity=cloud_opacity,
        cloud_radius=cloud_radius,
        cloud_color=cloud_color,
    )

    cell = SceneCell(
        title=_ensemble_title(
            group.iloc[0],
            coords_col=coords_col,
            energy_col=energy_col,
            mode=mode,
            n_conformers=len(records),
        ),
        models=models,
    )
    return GridScene(
        cells=[cell],
        columns=1,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
    )


def conformer_ensemble_grid_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int] | str | None = None,
    coords_col: str | None = None,
    energy_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    core_atoms: Sequence[int] | str = "auto",
    mode: str = "representatives+cloud",
    color_by: str | None = None,
    top_n: int | None = 25,
    energy_window_kcal: float | None = None,
    energy_scale_to_kcal: float | None = None,
    conformer_index: int = 0,
    cluster_threshold: float = 0.75,
    n_clusters: int | None = None,
    cell_size: tuple[int, int] = DEFAULT_CELL_SIZE,
    columns: int | None = None,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    linked: bool = False,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    cloud_opacity: float = 0.55,
    cloud_radius: float = 0.075,
    cloud_color: str = "#4f6f7a",
) -> GridScene:
    """Create a grid scene with one cell per selected conformer family.

    By default, every inferred conformer family is shown once. For screen
    dataframes this normally means one cell per reactive position, because
    ``rpos`` is part of the standard FRUST identity columns.

    Examples
    --------
    Show all reactive positions from a TS-guess dataframe:

    >>> import frust as ft
    >>> scene = ft.vis.conformer_ensemble_grid_scene_from_dataframe(
    ...     ts_guesses["TS3"],
    ...     mode="representatives+cloud",
    ... )

    Show only the families containing positional rows ``0`` and ``20``:

    >>> scene = ft.vis.conformer_ensemble_grid_scene_from_dataframe(
    ...     df,
    ...     row_indices=[0, 20],
    ... )

    Parameters
    ----------
    df
        FRUST dataframe containing one row per conformer.
    row_indices
        ``None`` or ``"all"`` to show every inferred conformer family. Pass one
        or more positional row numbers to show only the families containing
        those rows.
    coords_col, energy_col, group_cols, core_atoms, mode, color_by, top_n,
    energy_window_kcal, energy_scale_to_kcal, conformer_index,
    cluster_threshold, n_clusters, cell_size, background_color, linked,
    show_labels, show_charges, kekulize, cloud_opacity, cloud_radius,
    cloud_color
        Forwarded to :func:`conformer_ensemble_scene_from_dataframe` for each
        selected conformer family.
    columns
        Number of grid columns. If omitted, FRUST uses one column for a single
        family, two columns for two families, and at most three columns for
        larger grids.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene containing one cell per selected conformer family.
    """

    normalized = normalize_dataframe(df)
    if normalized.empty:
        raise ValueError("Cannot display conformers from an empty dataframe.")

    family_row_indices = _selected_family_row_indices(
        normalized,
        row_indices=row_indices,
        group_cols=group_cols,
    )
    cells: list[SceneCell] = []
    for row_index in family_row_indices:
        scene = conformer_ensemble_scene_from_dataframe(
            normalized,
            row_index=row_index,
            coords_col=coords_col,
            energy_col=energy_col,
            group_cols=group_cols,
            core_atoms=core_atoms,
            mode=mode,
            color_by=color_by,
            top_n=top_n,
            energy_window_kcal=energy_window_kcal,
            energy_scale_to_kcal=energy_scale_to_kcal,
            conformer_index=conformer_index,
            cluster_threshold=cluster_threshold,
            n_clusters=n_clusters,
            cell_size=cell_size,
            background_color=background_color,
            linked=linked,
            show_labels=show_labels,
            show_charges=show_charges,
            kekulize=kekulize,
            cloud_opacity=cloud_opacity,
            cloud_radius=cloud_radius,
            cloud_color=cloud_color,
        )
        cells.extend(scene.cells)

    return GridScene(
        cells=cells,
        columns=_resolve_grid_columns(columns, n_cells=len(cells)),
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
    )


def plot_conformers(
    df: pd.DataFrame,
    *,
    row_index: int | None = None,
    row_indices: Sequence[int] | str | None = None,
    coords_col: str | None = None,
    energy_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    core_atoms: Sequence[int] | str = "auto",
    mode: str = "representatives+cloud",
    color_by: str | None = None,
    top_n: int | None = 25,
    energy_window_kcal: float | None = None,
    energy_scale_to_kcal: float | None = None,
    conformer_index: int = 0,
    cluster_threshold: float = 0.75,
    n_clusters: int | None = None,
    cell_size: tuple[int, int] = (520, 480),
    columns: int | None = None,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    linked: bool = False,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    cloud_opacity: float = 0.55,
    cloud_radius: float = 0.075,
    cloud_color: str = "#4f6f7a",
    export_HTML: str = "",
):
    """Render an interactive py3Dmol conformer ensemble viewer.

    This is the notebook-facing helper for conformer display. By default it
    builds one core-aligned scene cell per inferred conformer family, renders
    the grid with py3Dmol, and optionally writes the same viewer to an HTML
    file. For screen dataframes, this normally means one cell per reactive
    position.

    The fixed core is drawn once in black. Conformer variation is shown by
    overlaying only the mobile atoms and their connector bonds. This makes the
    ensemble easier to read for transition-state guesses and constrained
    structures where duplicating the core can create misleading visuals.

    Examples
    --------
    Display every reactive position in a TS-guess dataframe:

    >>> import frust as ft
    >>> ft.plot_conformers(ts_guesses["TS1"])

    Display only the conformer family containing positional row ``0``:

    >>> ft.plot_conformers(ts_guesses["TS1"], row_index=0)

    Show the lowest-energy representative and a visible cloud on a white
    notebook background:

    >>> ft.plot_conformers(
    ...     ts_guesses["TS1"],
    ...     background_color="white",
    ...     mode="representatives+cloud",
    ...     color_by="uniform",
    ...     top_n=25,
    ...     energy_window_kcal=5.0,
    ...     cloud_opacity=0.55,
    ...     cloud_radius=0.075,
    ... )

    Inspect a single conformer from the selected family:

    >>> ft.plot_conformers(
    ...     df,
    ...     row_index=0,
    ...     mode="single",
    ...     conformer_index=2,
    ... )

    Export an HTML viewer for sharing or documentation:

    >>> ft.plot_conformers(
    ...     df,
    ...     mode="cluster",
    ...     n_clusters=4,
    ...     export_HTML="conformer-ensemble.html",
    ... )

    Parameters
    ----------
    df
        FRUST dataframe containing one row per conformer. Required columns are
        ``atoms`` and at least one coordinate column. ``connectivity_bonds``,
        ``constraint_roles``, ``constraint_atoms``, and energy columns are used
        when available.
    row_index
        Positional row used to choose one conformer family. If omitted, all
        inferred conformer families are shown.
    row_indices
        Positional rows used to choose multiple conformer families. ``None`` or
        ``"all"`` shows every inferred family. Do not pass both ``row_index``
        and ``row_indices``.
    coords_col
        Coordinate column to visualize. If omitted, FRUST uses the latest
        coordinate column in dataframe order.
    energy_col
        Energy column used for sorting, relative-energy coloring, and
        ``energy_window_kcal`` filtering. If omitted, FRUST chooses the latest
        known energy column, falling back to ``energy_uff``.
    group_cols
        Columns defining one conformer family. Leave as ``None`` for the
        standard FRUST identity columns, or pass custom columns for custom
        dataframes.
    core_atoms
        ``"auto"`` to use ``constraint_roles`` then ``constraint_atoms`` for
        the fixed alignment core, or an explicit sequence of atom indices.
    mode
        Display mode. ``"single"`` shows one conformer, ``"cloud"`` shows the
        mobile-atom cloud only, ``"representatives+cloud"`` adds the
        lowest-energy representative, and ``"cluster"`` adds one representative
        per mobile-atom RMSD cluster.
    color_by
        ``"energy"``, ``"cluster"``, or ``"uniform"``. If omitted, cluster mode
        uses ``"cluster"`` and other modes use ``"energy"``.
    top_n
        Maximum number of conformers to display after energy sorting.
        ``None`` keeps all conformers that pass other filters.
    energy_window_kcal
        Keep only conformers within this relative energy of the family minimum.
        Ignored when no energy column is available.
    energy_scale_to_kcal
        Optional multiplier converting ``energy_col`` values to kcal/mol.
    conformer_index
        Selected conformer position for ``mode="single"`` after filtering and
        sorting.
    cluster_threshold
        Mobile-atom RMSD threshold in Angstrom for ``mode="cluster"``.
    n_clusters
        Optional maximum cluster count. If supplied, this overrides
        ``cluster_threshold``.
    cell_size
        Width and height of the py3Dmol scene cell in pixels.
    columns
        Number of grid columns. If omitted, FRUST uses one column for a single
        family, two columns for two families, and at most three columns for
        larger grids.
    background_color
        py3Dmol background color, for example ``"white"`` or
        ``("white", 1.0)``.
    linked
        Link viewer motion if the scene contains multiple cells.
    show_labels
        Draw atom labels for displayed models.
    show_charges
        Draw formal charge labels where possible.
    kekulize
        Kekulize generated mol blocks when explicit connectivity is present.
    cloud_opacity
        Opacity for background conformer sticks and spheres. Increase it when
        the cloud is too faint; decrease it when the ensemble is too dense.
    cloud_radius
        Stick radius for background conformers. Increase it when the cloud is
        visually too thin on a white background.
    cloud_color
        Color for ``color_by="uniform"`` and for fallback coloring when no
        energy values are available.
    export_HTML
        Optional HTML export path. When supplied, the rendered py3Dmol viewer
        is written to this file.

    Returns
    -------
    py3Dmol.view
        Rendered viewer.

    See Also
    --------
    conformer_ensemble_scene_from_dataframe
        Build a single-family scene object without immediately rendering it.
    conformer_ensemble_grid_scene_from_dataframe
        Build an all-families grid scene object without immediately rendering
        it.
    show_conformer_ensemble_scene
        Render a prebuilt conformer scene while preserving per-model styles.
    """

    if row_index is not None and row_indices is not None:
        raise ValueError("Pass either row_index or row_indices, not both.")
    selected_row_indices: Sequence[int] | str | None
    selected_row_indices = [int(row_index)] if row_index is not None else row_indices

    scene = conformer_ensemble_grid_scene_from_dataframe(
        df,
        row_indices=selected_row_indices,
        coords_col=coords_col,
        energy_col=energy_col,
        group_cols=group_cols,
        core_atoms=core_atoms,
        mode=mode,
        color_by=color_by,
        top_n=top_n,
        energy_window_kcal=energy_window_kcal,
        energy_scale_to_kcal=energy_scale_to_kcal,
        conformer_index=conformer_index,
        cluster_threshold=cluster_threshold,
        n_clusters=n_clusters,
        cell_size=cell_size,
        columns=columns,
        background_color=background_color,
        linked=linked,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        cloud_opacity=cloud_opacity,
        cloud_radius=cloud_radius,
        cloud_color=cloud_color,
    )
    return show_conformer_ensemble_scene(scene, export_HTML=export_HTML)


def show_conformer_ensemble_scene(scene: GridScene, *, export_HTML: str = ""):
    """Render a conformer ensemble scene with per-model py3Dmol styles.

    Use this helper for scenes created by
    :func:`conformer_ensemble_scene_from_dataframe`. It reapplies styles after
    py3Dmol rendering so transparent cloud models, representatives, connectors,
    and the fixed black core keep their individual colors and opacities.

    Parameters
    ----------
    scene
        Scene produced by :func:`conformer_ensemble_scene_from_dataframe`.
    export_HTML
        Optional HTML export path. When supplied, the rendered viewer is saved
        after model-specific styles have been applied.

    Returns
    -------
    py3Dmol.view
        Rendered viewer.
    """

    renderer = Py3DmolGridRenderer(scene)
    viewer = renderer.render()
    _apply_conformer_model_styles(viewer, scene)
    if export_HTML:
        renderer.write_html(export_HTML)
        print(f"HTML export successful: {export_HTML}")
    return viewer


def _validate_mode(mode: str) -> str:
    if mode not in VALID_CONFORMER_MODES:
        valid = ", ".join(sorted(VALID_CONFORMER_MODES))
        raise ValueError(f"Invalid conformer mode {mode!r}. Expected one of: {valid}.")
    return mode


def _resolve_color_by(color_by: str | None, mode: str) -> str:
    resolved = "cluster" if mode == "cluster" and color_by is None else color_by or "energy"
    if resolved not in VALID_COLOR_MODES:
        valid = ", ".join(sorted(VALID_COLOR_MODES))
        raise ValueError(f"Invalid color_by {resolved!r}. Expected one of: {valid}.")
    return resolved


def _validate_opacity(value: float, *, name: str) -> float:
    opacity = float(value)
    if opacity < 0.0 or opacity > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return opacity


def _validate_positive_float(value: float, *, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return number


def _resolve_coords_col(df: pd.DataFrame, coords_col: str | None) -> str:
    if coords_col is not None:
        if coords_col not in df.columns:
            raise KeyError(f"Coordinate column {coords_col!r} is not present.")
        return str(coords_col)
    cols = coordinate_columns(df)
    if not cols:
        raise ValueError("No coordinate columns found for conformer display.")
    return cols[-1]


def _resolve_energy_col(df: pd.DataFrame, energy_col: str | None) -> str | None:
    if energy_col is not None:
        if energy_col not in df.columns:
            raise KeyError(f"Energy column {energy_col!r} is not present.")
        return str(energy_col)
    cols = energy_columns(df)
    if cols:
        return str(cols[-1])
    if "energy_uff" in df.columns:
        return "energy_uff"
    return None


def _select_conformer_group(
    df: pd.DataFrame,
    *,
    row_index: int,
    group_cols: Sequence[str] | None,
) -> pd.DataFrame:
    resolved_cols = _resolve_group_cols(df, group_cols)
    if not resolved_cols:
        return df

    selected = df.iloc[int(row_index)]
    mask = pd.Series(True, index=df.index)
    for col in resolved_cols:
        value = selected[col]
        if _is_missing_scalar(value):
            mask &= df[col].isna()
        else:
            mask &= df[col] == value
    return df.loc[mask]


def _selected_family_row_indices(
    df: pd.DataFrame,
    *,
    row_indices: Sequence[int] | str | None,
    group_cols: Sequence[str] | None,
) -> list[int]:
    resolved_cols = _resolve_group_cols(df, group_cols)
    if row_indices is None:
        return _stable_family_row_indices(df, resolved_cols)
    if isinstance(row_indices, str):
        if row_indices == "all":
            return _stable_family_row_indices(df, resolved_cols)
        raise ValueError("row_indices must be 'all', None, or one or more positional row numbers.")

    positions = _coerce_row_positions(row_indices, n_rows=len(df))
    return _dedupe_positions_by_family(df, positions, resolved_cols)


def _resolve_group_cols(df: pd.DataFrame, group_cols: Sequence[str] | None) -> list[str]:
    resolved_cols = list(group_cols) if group_cols is not None else _default_group_cols(df)
    missing = [col for col in resolved_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing conformer group column(s): {missing}.")
    return resolved_cols


def _stable_family_row_indices(df: pd.DataFrame, group_cols: Sequence[str]) -> list[int]:
    if not group_cols:
        return [0]
    return _dedupe_positions_by_family(df, range(len(df)), group_cols)


def _coerce_row_positions(row_indices: Sequence[int] | int, *, n_rows: int) -> list[int]:
    if isinstance(row_indices, (int, np.integer)):
        raw_positions = [int(row_indices)]
    else:
        raw_positions = [int(row_index) for row_index in row_indices]
    if not raw_positions:
        raise ValueError("row_indices must contain at least one row number.")
    invalid = [row_index for row_index in raw_positions if row_index < 0 or row_index >= n_rows]
    if invalid:
        raise IndexError(f"row_indices contains row(s) outside dataframe bounds: {invalid}.")
    return raw_positions


def _dedupe_positions_by_family(
    df: pd.DataFrame,
    positions: Sequence[int] | range,
    group_cols: Sequence[str],
) -> list[int]:
    if not group_cols:
        return [int(next(iter(positions)))]

    seen: set[tuple[Any, ...]] = set()
    out: list[int] = []
    for position in positions:
        row = df.iloc[int(position)]
        key = tuple(_group_key_value(row[col]) for col in group_cols)
        if key in seen:
            continue
        seen.add(key)
        out.append(int(position))
    return out


def _group_key_value(value: Any) -> Any:
    if _is_missing_scalar(value):
        return ("__missing__", None)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _resolve_grid_columns(columns: int | None, *, n_cells: int) -> int:
    if columns is not None:
        resolved = int(columns)
        if resolved < 1:
            raise ValueError("columns must be >= 1.")
        return resolved
    if n_cells <= 1:
        return 1
    if n_cells == 2:
        return 2
    return min(3, int(n_cells))


def _default_group_cols(df: pd.DataFrame) -> list[str]:
    cols = infer_group_columns(df)
    if cols:
        return cols
    for fallback in ("structure_id", "custom_name", "substrate_name"):
        if fallback in df.columns:
            return [fallback]
    return []


def _rows_with_valid_coords(df: pd.DataFrame, coords_col: str) -> pd.DataFrame:
    mask = df[coords_col].map(lambda value: _valid_coord_array(value, expected_atoms=None))
    return df.loc[mask]


def _valid_coord_array(value: Any, *, expected_atoms: int | None) -> bool:
    try:
        arr = _coerce_coord_array(value)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 2 or arr.shape[1] != 3:
        return False
    if expected_atoms is not None and arr.shape[0] != expected_atoms:
        return False
    return bool(np.isfinite(arr).all())


def _coerce_coord_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        try:
            rows = [np.asarray(row, dtype=float) for row in value]
            return np.asarray(rows, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Could not coerce coordinates to a float array.") from exc


def _normalize_group_bonds(row: pd.Series) -> list[tuple[int, int]]:
    from tooltoad.scene3d import normalize_bond_pairs

    return normalize_bond_pairs(row.get("connectivity_bonds")) or []


def _resolve_core_atoms(
    row: pd.Series,
    *,
    core_atoms: Sequence[int] | str,
    atom_count: int,
) -> list[int]:
    if isinstance(core_atoms, str):
        if core_atoms != "auto":
            raise ValueError("core_atoms must be 'auto' or a sequence of atom indices.")
        resolved = _auto_core_atoms(row)
    else:
        resolved = [int(atom_idx) for atom_idx in core_atoms]

    unique = sorted(set(resolved))
    if len(unique) < 3:
        raise ValueError(
            "Conformer alignment requires at least three core atoms. "
            "Pass explicit `core_atoms=[...]` if auto-detection is not suitable."
        )
    invalid = [idx for idx in unique if idx < 0 or idx >= atom_count]
    if invalid:
        raise ValueError(f"core_atoms contains invalid atom indices: {invalid}.")
    return unique


def _auto_core_atoms(row: pd.Series) -> list[int]:
    roles = row.get("constraint_roles")
    if isinstance(roles, Mapping) and roles:
        return [int(atom_idx) for atom_idx in roles.values()]
    atoms = row.get("constraint_atoms")
    if isinstance(atoms, np.ndarray):
        return [int(atom_idx) for atom_idx in atoms.tolist()]
    if isinstance(atoms, Sequence) and not isinstance(atoms, (str, bytes)):
        return [int(atom_idx) for atom_idx in atoms]
    return []


def _build_conformer_records(
    group: pd.DataFrame,
    *,
    atoms: Sequence[str],
    coords_col: str,
    energy_col: str | None,
    core_atoms: Sequence[int],
    energy_scale_to_kcal: float | None,
) -> list[_ConformerRecord]:
    rows = []
    energies = _energy_values_kcal(
        group,
        energy_col=energy_col,
        energy_scale_to_kcal=energy_scale_to_kcal,
    )
    reference_position = _reference_position(group, energies)
    reference_coords = _coords_array(
        group.loc[reference_position, coords_col],
        atom_count=len(atoms),
    )

    finite_energies = [energy for energy in energies.values() if energy is not None]
    minimum_energy = min(finite_energies) if finite_energies else None
    for display_order, (row_key, row) in enumerate(group.iterrows()):
        coords = _coords_array(row[coords_col], atom_count=len(atoms))
        aligned = _align_coords_to_reference(coords, reference_coords, core_atoms)
        energy = energies.get(row_key)
        rel_energy = None if energy is None or minimum_energy is None else energy - minimum_energy
        rows.append(
            _ConformerRecord(
                row_key=row_key,
                display_order=display_order,
                row=row,
                coords=coords,
                aligned_coords=aligned,
                relative_energy_kcal=rel_energy,
            )
        )
    return sorted(rows, key=_record_sort_key)


def _energy_values_kcal(
    group: pd.DataFrame,
    *,
    energy_col: str | None,
    energy_scale_to_kcal: float | None,
) -> dict[Any, float | None]:
    if energy_col is None:
        return {idx: None for idx in group.index}
    scale = _energy_scale(energy_col, energy_scale_to_kcal=energy_scale_to_kcal)
    values: dict[Any, float | None] = {}
    for idx, value in group[energy_col].items():
        try:
            if pd.isna(value):
                values[idx] = None
                continue
            values[idx] = float(value) * scale
        except (TypeError, ValueError):
            values[idx] = None
    return values


def _energy_scale(energy_col: str, *, energy_scale_to_kcal: float | None) -> float:
    if energy_scale_to_kcal is not None:
        return float(energy_scale_to_kcal)
    if energy_col == "energy_uff":
        return 1.0
    return HARTREE_TO_KCAL_MOL


def _reference_position(group: pd.DataFrame, energies: Mapping[Any, float | None]) -> Any:
    finite = [(idx, energy) for idx, energy in energies.items() if energy is not None]
    if finite:
        return min(finite, key=lambda item: item[1])[0]
    return group.index[0]


def _coords_array(value: Any, *, atom_count: int) -> np.ndarray:
    arr = _coerce_coord_array(value)
    if arr.shape != (atom_count, 3) or not np.isfinite(arr).all():
        raise ValueError(
            f"Expected coordinates with shape ({atom_count}, 3), got {arr.shape}."
        )
    return arr


def _align_coords_to_reference(
    coords: np.ndarray,
    reference_coords: np.ndarray,
    core_atoms: Sequence[int],
) -> np.ndarray:
    source = coords[list(core_atoms)]
    target = reference_coords[list(core_atoms)]
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat)) or 1.0
    rotation = u_mat @ correction @ vt_mat
    return (coords - source_center) @ rotation + target_center


def _filter_conformer_records(
    records: list[_ConformerRecord],
    *,
    top_n: int | None,
    energy_window_kcal: float | None,
) -> list[_ConformerRecord]:
    out = records
    if energy_window_kcal is not None:
        out = [
            record
            for record in out
            if record.relative_energy_kcal is None
            or record.relative_energy_kcal <= float(energy_window_kcal)
        ]
    if top_n is not None:
        out = out[: int(top_n)]
    return out


def _record_sort_key(record: _ConformerRecord) -> tuple[float, int]:
    energy = record.relative_energy_kcal
    return (float("inf") if energy is None else float(energy), record.display_order)


def _with_cluster_labels(
    records: list[_ConformerRecord],
    *,
    atoms: Sequence[str],
    mobile_atoms: Sequence[int],
    threshold: float,
    n_clusters: int | None,
) -> list[_ConformerRecord]:
    labels = _cluster_labels(
        records,
        atoms=atoms,
        mobile_atoms=mobile_atoms,
        threshold=threshold,
        n_clusters=n_clusters,
    )
    return [
        _ConformerRecord(
            row_key=record.row_key,
            display_order=record.display_order,
            row=record.row,
            coords=record.coords,
            aligned_coords=record.aligned_coords,
            relative_energy_kcal=record.relative_energy_kcal,
            cluster=int(label),
        )
        for record, label in zip(records, labels)
    ]


def _cluster_labels(
    records: list[_ConformerRecord],
    *,
    atoms: Sequence[str],
    mobile_atoms: Sequence[int],
    threshold: float,
    n_clusters: int | None,
) -> np.ndarray:
    if len(records) == 1:
        return np.array([1], dtype=int)
    rmsd_atoms = [idx for idx in mobile_atoms if str(atoms[idx]).upper() != "H"] or list(mobile_atoms)
    distances = np.zeros((len(records), len(records)), dtype=float)
    for i, left in enumerate(records):
        left_coords = left.aligned_coords[rmsd_atoms]
        for j in range(i + 1, len(records)):
            right_coords = records[j].aligned_coords[rmsd_atoms]
            diff = left_coords - right_coords
            distances[i, j] = distances[j, i] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    if np.allclose(distances, 0.0):
        return np.ones(len(records), dtype=int)
    tree = linkage(squareform(distances), method="average")
    if n_clusters is not None:
        return fcluster(tree, int(n_clusters), criterion="maxclust")
    return fcluster(tree, float(threshold), criterion="distance")


def _ensemble_models(
    records: list[_ConformerRecord],
    *,
    atoms: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    core_atoms: Sequence[int],
    mobile_atoms: Sequence[int],
    mode: str,
    color_by: str,
    conformer_index: int,
    show_labels: bool,
    show_charges: bool,
    kekulize: bool,
    cloud_opacity: float,
    cloud_radius: float,
    cloud_color: str,
) -> list[MoleculeModel]:
    reference = records[0]
    models = [
        _subset_model(
            atoms,
            reference.aligned_coords,
            bonds,
            atom_indices=core_atoms,
            style=CORE_STYLE,
            show_labels=show_labels,
            show_charges=show_charges,
            kekulize=kekulize,
        )
    ]
    if mode == "single":
        selected = _record_at_position(records, conformer_index)
        selected_style = _representative_style(
            selected,
            records,
            color_by=color_by,
            default_color=cloud_color,
        )
        models.append(
            _subset_model(
                atoms,
                selected.aligned_coords,
                bonds,
                atom_indices=mobile_atoms,
                style=selected_style,
                show_labels=show_labels,
                show_charges=show_charges,
                kekulize=kekulize,
            )
        )
        models.extend(
            _connector_models(
                atoms,
                selected.aligned_coords,
                bonds,
                core_atoms=core_atoms,
                mobile_atoms=mobile_atoms,
                style=selected_style,
                kekulize=kekulize,
            )
        )
        return models

    for record in records:
        record_style = _cloud_style(
            record,
            records,
            color_by=color_by,
            cloud_opacity=cloud_opacity,
            cloud_radius=cloud_radius,
            cloud_color=cloud_color,
        )
        models.append(
            _subset_model(
                atoms,
                record.aligned_coords,
                bonds,
                atom_indices=mobile_atoms,
                style=record_style,
                show_labels=show_labels,
                show_charges=show_charges,
                kekulize=kekulize,
            )
        )
        models.extend(
            _connector_models(
                atoms,
                record.aligned_coords,
                bonds,
                core_atoms=core_atoms,
                mobile_atoms=mobile_atoms,
                style=record_style,
                kekulize=kekulize,
            )
        )

    if mode == "representatives+cloud":
        representative = records[0]
        representative_style = _representative_style(
            representative,
            records,
            color_by=color_by,
            default_color=cloud_color,
        )
        models.append(
            _subset_model(
                atoms,
                representative.aligned_coords,
                bonds,
                atom_indices=mobile_atoms,
                style=representative_style,
                show_labels=show_labels,
                show_charges=show_charges,
                kekulize=kekulize,
            )
        )
        models.extend(
            _connector_models(
                atoms,
                representative.aligned_coords,
                bonds,
                core_atoms=core_atoms,
                mobile_atoms=mobile_atoms,
                style=representative_style,
                kekulize=kekulize,
            )
        )
    elif mode == "cluster":
        for representative in _cluster_representatives(records):
            representative_style = _representative_style(
                representative,
                records,
                color_by="cluster",
                default_color=cloud_color,
            )
            models.append(
                _subset_model(
                    atoms,
                    representative.aligned_coords,
                    bonds,
                    atom_indices=mobile_atoms,
                    style=representative_style,
                    show_labels=show_labels,
                    show_charges=show_charges,
                    kekulize=kekulize,
                )
            )
            models.extend(
                _connector_models(
                    atoms,
                    representative.aligned_coords,
                    bonds,
                    core_atoms=core_atoms,
                    mobile_atoms=mobile_atoms,
                    style=representative_style,
                    kekulize=kekulize,
                )
            )
    return models


def _record_at_position(records: list[_ConformerRecord], conformer_index: int) -> _ConformerRecord:
    idx = int(conformer_index)
    if idx < 0 or idx >= len(records):
        raise IndexError(f"conformer_index {idx} is outside the selected conformer range.")
    return records[idx]


def _subset_model(
    atoms: Sequence[str],
    coords: np.ndarray,
    bonds: Sequence[tuple[int, int]],
    *,
    atom_indices: Sequence[int],
    style: dict[str, Any],
    show_labels: bool,
    show_charges: bool,
    kekulize: bool,
) -> MoleculeModel:
    index_map = {int(parent): local for local, parent in enumerate(atom_indices)}
    subset_bonds = [
        (index_map[int(begin)], index_map[int(end)])
        for begin, end in bonds
        if int(begin) in index_map and int(end) in index_map
    ]
    return MoleculeModel(
        atoms=[atoms[idx] for idx in atom_indices],
        coords=[coords[idx].tolist() for idx in atom_indices],
        bonds=subset_bonds,
        style=style,
        show_atom_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
    )


def _connector_models(
    atoms: Sequence[str],
    coords: np.ndarray,
    bonds: Sequence[tuple[int, int]],
    *,
    core_atoms: Sequence[int],
    mobile_atoms: Sequence[int],
    style: dict[str, Any],
    kekulize: bool,
) -> list[MoleculeModel]:
    core = set(int(atom_idx) for atom_idx in core_atoms)
    mobile = set(int(atom_idx) for atom_idx in mobile_atoms)
    connectors = []
    connector_style = _connector_style(style)
    for begin, end in bonds:
        begin_i, end_i = int(begin), int(end)
        if not (
            (begin_i in core and end_i in mobile)
            or (begin_i in mobile and end_i in core)
        ):
            continue
        connectors.append(
            MoleculeModel(
                atoms=[atoms[begin_i], atoms[end_i]],
                coords=[coords[begin_i].tolist(), coords[end_i].tolist()],
                bonds=[(0, 1)],
                style=connector_style,
                show_atom_labels=False,
                show_charges=False,
                kekulize=kekulize,
            )
        )
    return connectors


def _connector_style(style: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stick = dict(style.get("stick", {}))
    if "radius" in stick:
        stick["radius"] = max(0.035, float(stick["radius"]) * 0.9)
    return {"stick": stick}


def _cloud_style(
    record: _ConformerRecord,
    records: Sequence[_ConformerRecord],
    *,
    color_by: str,
    cloud_opacity: float,
    cloud_radius: float,
    cloud_color: str,
) -> dict[str, Any]:
    style = _style_copy(CLOUD_STYLE)
    style_color = _record_color(
        record,
        records,
        color_by=color_by,
        default_color=cloud_color,
    )
    style["stick"]["radius"] = cloud_radius
    style["stick"]["opacity"] = cloud_opacity
    style["sphere"]["radius"] = cloud_radius * 2.2
    style["sphere"]["opacity"] = cloud_opacity
    style["stick"]["color"] = style_color
    style["sphere"]["color"] = style_color
    return style


def _representative_style(
    record: _ConformerRecord,
    records: Sequence[_ConformerRecord],
    *,
    color_by: str,
    default_color: str,
) -> dict[str, Any]:
    style = _style_copy(REPRESENTATIVE_STYLE)
    style_color = _record_color(
        record,
        records,
        color_by=color_by,
        default_color=default_color,
    )
    style["stick"]["color"] = style_color
    style["sphere"]["color"] = style_color
    return style


def _style_copy(style: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {key: dict(value) for key, value in style.items()}


def _record_color(
    record: _ConformerRecord,
    records: Sequence[_ConformerRecord],
    *,
    color_by: str,
    default_color: str,
) -> str:
    if color_by == "uniform":
        return default_color
    if color_by == "cluster":
        cluster = 1 if record.cluster is None else int(record.cluster)
        return CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
    return _energy_color(record, records, default_color=default_color)


def _energy_color(
    record: _ConformerRecord,
    records: Sequence[_ConformerRecord],
    *,
    default_color: str,
) -> str:
    energies = [item.relative_energy_kcal for item in records if item.relative_energy_kcal is not None]
    if record.relative_energy_kcal is None or not energies:
        return default_color
    high = max(energies)
    if high <= 0:
        frac = 0.0
    else:
        frac = min(1.0, max(0.0, float(record.relative_energy_kcal) / high))
    rgb = [
        int(round(low + frac * (high_value - low)))
        for low, high_value in zip(ENERGY_LOW_COLOR, ENERGY_HIGH_COLOR)
    ]
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _cluster_representatives(records: Sequence[_ConformerRecord]) -> list[_ConformerRecord]:
    by_cluster: dict[int, list[_ConformerRecord]] = {}
    for record in records:
        by_cluster.setdefault(int(record.cluster or 1), []).append(record)
    return [
        sorted(cluster_records, key=_record_sort_key)[0]
        for _, cluster_records in sorted(by_cluster.items())
    ]


def _ensemble_title(
    row: pd.Series,
    *,
    coords_col: str,
    energy_col: str | None,
    mode: str,
    n_conformers: int,
) -> str:
    label = _row_label(row)
    energy_label = "no energy" if energy_col is None else energy_col
    return f"{label}\n{mode} | {n_conformers} conformers | {coords_col} | {energy_label}"


def _row_label(row: pd.Series) -> str:
    substrate_name = row.get("substrate_name", row.get("custom_name", "molecule"))
    rpos = row.get("rpos")
    if rpos is None or _is_missing_scalar(rpos):
        return str(substrate_name)
    return f"{substrate_name} r{rpos}"


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _apply_conformer_model_styles(viewer: Any, scene: GridScene) -> None:
    """Apply per-model styles after Tooltoad's broad cell styling."""
    columns = max(1, int(scene.columns))
    for cell_idx, cell in enumerate(scene.cells):
        viewer_position = (cell_idx // columns, cell_idx % columns)
        for model_idx, model in enumerate(cell.models):
            if model.style is not None:
                viewer.setStyle({"model": model_idx}, model.style, viewer=viewer_position)
