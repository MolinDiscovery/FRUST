"""Scene-based RMSD comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tooltoad.scene3d import (
    AtomHighlight,
    DistanceOverlay,
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    ScreenLabelOverlay,
    SceneCell,
)

from frust.schema import normalize_dataframe
from frust.utils.RMSD import (
    compare_symbols_coords_rmsd,
    compare_xyz_rmsd as _compare_xyz_rmsd_compute,
)
from frust.vis.scenes import DEFAULT_SCENE_BACKGROUND

REFERENCE_STYLE: dict[str, Any] = {
    "stick": {"radius": 0.16},
    "sphere": {"radius": 0.28},
}
PROBE_STYLE: dict[str, Any] = {
    "stick": {"radius": 0.08, "color": "orange"},
    "sphere": {"radius": 0.18, "color": "orange"},
}
VALID_SHOW_MODES = {"deviations", "overlay", "none"}
DEVIATION_LABEL_OFFSET = {"x": 10, "y": 34}
DEVIATION_LABEL_STEP = 24


def compare_xyz_rmsd(
    probe_xyz_path: str,
    ref_xyz_path: str,
    *,
    atom_scope: str = "heavy",
    charge: int = 0,
    show: str = "deviations",
    render: bool = True,
    show_table: bool = False,
    top_n: int = 3,
    table_rows: int = 15,
    print_summary: bool = True,
    cell_size: tuple[int, int] = (520, 480),
    linked: bool = False,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    hide_hydrogens: bool = True,
    export_HTML: str = "",
    show_overlay_plot: bool | None = None,
    show_deviation_overlay: bool | None = None,
) -> dict[str, Any]:
    """Compare two XYZ structures and optionally render a scene overlay.

    Parameters
    ----------
    probe_xyz_path
        Path to the probe XYZ structure that will be aligned.
    ref_xyz_path
        Path to the reference XYZ structure.
    atom_scope
        Atom scope used for RMSD. Currently only ``"heavy"`` is supported,
        meaning hydrogens are ignored during atom mapping and RMSD.
    charge
        Total molecular charge used during RDKit bond perception.
    show
        Scene mode. Use ``"deviations"`` to draw the largest mapped atom
        deviations, ``"overlay"`` to show only the aligned structures, or
        ``"none"`` to skip scene creation.
    render
        If ``True``, render the scene with py3Dmol. If ``False``, return the
        scene object without creating a viewer.
    show_table
        Display a compact per-atom deviation table.
    top_n
        Number of largest deviations to show in ``show="deviations"`` mode.
    table_rows
        Number of rows to display when ``show_table=True``.
    print_summary
        Print a short text summary.
    cell_size
        Width and height of the comparison scene cell in pixels.
    linked
        Link rotations and zoom when the scene has multiple cells. The default
        comparison scene has one cell.
    background_color
        Viewer background as a color or ``(color, opacity)`` tuple.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw non-zero formal charges when available.
    kekulize
        Kekulize RDKit mol blocks before display.
    hide_hydrogens
        Hide hydrogen atoms in the rendered overlay while still aligning on
        heavy atoms.
    export_HTML
        Optional path for exporting the rendered scene to HTML.
    show_overlay_plot, show_deviation_overlay
        Compatibility flags from the old ``frust.vis.compare_xyz_rmsd`` import
        path. Prefer ``show`` in new code.

    Returns
    -------
    dict
        RMSD result with ``scene`` and ``viewer`` keys added when applicable.
    """
    show = _resolve_show_mode(
        show,
        show_overlay_plot=show_overlay_plot,
        show_deviation_overlay=show_deviation_overlay,
    )
    result = _compare_xyz_rmsd_compute(
        probe_xyz_path,
        ref_xyz_path,
        atom_scope=atom_scope,
        charge=charge,
        show_overlay_plot=False,
        show_deviation_overlay=False,
        show_table=False,
        top_n=top_n,
        table_rows=table_rows,
        print_summary=print_summary,
    )
    result["probe_label"] = Path(probe_xyz_path).stem
    result["ref_label"] = Path(ref_xyz_path).stem
    return _finalize_comparison_result(
        result,
        show=show,
        render=render,
        show_table=show_table,
        table_rows=table_rows,
        top_n=top_n,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        hide_hydrogens=hide_hydrogens,
        export_HTML=export_HTML,
    )


def compare_structure_rmsd(
    df: pd.DataFrame,
    *,
    probe_col: str,
    ref_col: str,
    row_index: int = 0,
    atom_scope: str = "heavy",
    charge: int = 0,
    show: str = "deviations",
    render: bool = True,
    show_table: bool = False,
    top_n: int = 3,
    table_rows: int = 15,
    print_summary: bool = True,
    cell_size: tuple[int, int] = (520, 480),
    linked: bool = False,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    hide_hydrogens: bool = True,
    export_HTML: str = "",
) -> dict[str, Any]:
    """Compare two coordinate columns from one FRUST dataframe row.

    Parameters
    ----------
    df
        FRUST dataframe containing ``atoms`` and the two coordinate columns.
    probe_col
        Coordinate column that will be aligned to the reference.
    ref_col
        Coordinate column used as the reference.
    row_index
        Positional row index to compare.
    atom_scope
        Atom scope used for RMSD. Currently only ``"heavy"`` is supported.
    charge
        Total molecular charge used during RDKit bond perception.
    show
        Scene mode. Use ``"deviations"``, ``"overlay"``, or ``"none"``.
    render
        If ``True``, render the scene with py3Dmol.
    show_table
        Display a compact per-atom deviation table.
    top_n
        Number of largest deviations to show in ``show="deviations"`` mode.
    table_rows
        Number of rows to display when ``show_table=True``.
    print_summary
        Print a short text summary.
    cell_size
        Width and height of the comparison scene cell in pixels.
    linked
        Link rotations and zoom when the scene has multiple cells.
    background_color
        Viewer background as a color or ``(color, opacity)`` tuple.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw non-zero formal charges when available.
    kekulize
        Kekulize RDKit mol blocks before display.
    hide_hydrogens
        Hide hydrogen atoms in the rendered overlay.
    export_HTML
        Optional path for exporting the rendered scene to HTML.

    Returns
    -------
    dict
        RMSD result with row metadata, ``scene``, and ``viewer`` keys.
    """
    show = _validate_show_mode(show)
    result = _compare_dataframe_row(
        df,
        probe_col=probe_col,
        ref_col=ref_col,
        row_index=row_index,
        atom_scope=atom_scope,
        charge=charge,
    )
    if print_summary:
        print(f"Row: {row_index}")
        print(f"Probe column: {probe_col}")
        print(f"Reference column: {ref_col}")
        print(f"Atom scope: {atom_scope}")
        print(f"Mapped atoms: {len(result['atom_map'])}")
        print(f"RMSD: {result['rmsd']:.6f} A")
        if not result["df_dev"].empty:
            worst = result["df_dev"].iloc[0]
            print(
                "Largest mapped deviation: "
                f'{worst["probe_symbol"]}{int(worst["probe_idx"])} -> '
                f'{worst["ref_symbol"]}{int(worst["ref_idx"])} = '
                f'{worst["distance_A"]:.4f} A'
            )
    return _finalize_comparison_result(
        result,
        show=show,
        render=render,
        show_table=show_table,
        table_rows=table_rows,
        top_n=top_n,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        hide_hydrogens=hide_hydrogens,
        export_HTML=export_HTML,
    )


def structure_comparison_scene_from_xyz(
    probe_xyz_path: str,
    ref_xyz_path: str,
    *,
    atom_scope: str = "heavy",
    charge: int = 0,
    show: str = "deviations",
    top_n: int = 3,
    cell_size: tuple[int, int] = (520, 480),
    linked: bool = False,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    hide_hydrogens: bool = True,
) -> GridScene:
    """Create an RMSD comparison scene from two XYZ files.

    Parameters
    ----------
    probe_xyz_path
        Path to the probe XYZ structure that will be aligned.
    ref_xyz_path
        Path to the reference XYZ structure.
    atom_scope
        Atom scope used for RMSD. Currently only ``"heavy"`` is supported.
    charge
        Total molecular charge used during RDKit bond perception.
    show
        Scene mode. Use ``"deviations"`` or ``"overlay"``.
    top_n
        Number of largest deviations to show in ``show="deviations"`` mode.
    cell_size
        Width and height of the scene cell in pixels.
    linked
        Link rotations and zoom when the scene has multiple cells.
    background_color
        Viewer background as a color or ``(color, opacity)`` tuple.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw non-zero formal charges when available.
    kekulize
        Kekulize RDKit mol blocks before display.
    hide_hydrogens
        Hide hydrogen atoms in the rendered overlay.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene ready for rendering with :func:`frust.vis.show_scene`.
    """
    result = _compare_xyz_rmsd_compute(
        probe_xyz_path,
        ref_xyz_path,
        atom_scope=atom_scope,
        charge=charge,
        show_overlay_plot=False,
        show_deviation_overlay=False,
        show_table=False,
        print_summary=False,
    )
    result["probe_label"] = Path(probe_xyz_path).stem
    result["ref_label"] = Path(ref_xyz_path).stem
    return structure_comparison_scene(
        result,
        show=show,
        top_n=top_n,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        hide_hydrogens=hide_hydrogens,
    )


def structure_comparison_scene_from_dataframe(
    df: pd.DataFrame,
    *,
    probe_col: str,
    ref_col: str,
    row_index: int = 0,
    atom_scope: str = "heavy",
    charge: int = 0,
    show: str = "deviations",
    top_n: int = 3,
    cell_size: tuple[int, int] = (520, 480),
    linked: bool = False,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    hide_hydrogens: bool = True,
) -> GridScene:
    """Create an RMSD comparison scene from two dataframe coordinate columns.

    Parameters
    ----------
    df
        FRUST dataframe containing ``atoms`` and the two coordinate columns.
    probe_col
        Coordinate column that will be aligned to the reference.
    ref_col
        Coordinate column used as the reference.
    row_index
        Positional row index to compare.
    atom_scope
        Atom scope used for RMSD. Currently only ``"heavy"`` is supported.
    charge
        Total molecular charge used during RDKit bond perception.
    show
        Scene mode. Use ``"deviations"`` or ``"overlay"``.
    top_n
        Number of largest deviations to show in ``show="deviations"`` mode.
    cell_size
        Width and height of the scene cell in pixels.
    linked
        Link rotations and zoom when the scene has multiple cells.
    background_color
        Viewer background as a color or ``(color, opacity)`` tuple.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw non-zero formal charges when available.
    kekulize
        Kekulize RDKit mol blocks before display.
    hide_hydrogens
        Hide hydrogen atoms in the rendered overlay.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene ready for rendering with :func:`frust.vis.show_scene`.
    """
    result = _compare_dataframe_row(
        df,
        probe_col=probe_col,
        ref_col=ref_col,
        row_index=row_index,
        atom_scope=atom_scope,
        charge=charge,
    )
    return structure_comparison_scene(
        result,
        show=show,
        top_n=top_n,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
        show_labels=show_labels,
        show_charges=show_charges,
        kekulize=kekulize,
        hide_hydrogens=hide_hydrogens,
    )


def structure_comparison_scene(
    result: dict[str, Any],
    *,
    show: str = "deviations",
    top_n: int = 3,
    cell_size: tuple[int, int] = (520, 480),
    linked: bool = False,
    background_color: str | tuple[str, float] = DEFAULT_SCENE_BACKGROUND,
    show_labels: bool = False,
    show_charges: bool = True,
    kekulize: bool = True,
    hide_hydrogens: bool = True,
) -> GridScene:
    """Create a scene from an RMSD comparison result.

    Parameters
    ----------
    result
        Result dictionary produced by ``compare_symbols_coords_rmsd`` or one
        of the higher-level comparison helpers.
    show
        Scene mode. Use ``"deviations"`` or ``"overlay"``.
    top_n
        Number of largest deviations to show in ``show="deviations"`` mode.
    cell_size
        Width and height of the comparison scene cell in pixels.
    linked
        Link rotations and zoom when the scene has multiple cells.
    background_color
        Viewer background as a color or ``(color, opacity)`` tuple.
    show_labels
        Draw atom labels before rendering.
    show_charges
        Draw non-zero formal charges when available.
    kekulize
        Kekulize RDKit mol blocks before display.
    hide_hydrogens
        Hide hydrogen atoms in the rendered overlay.

    Returns
    -------
    tooltoad.scene3d.GridScene
        Scene ready for rendering.
    """
    show = _validate_show_mode(show)
    if show == "none":
        raise ValueError("structure comparison scenes require show='deviations' or 'overlay'.")

    hidden = ("H",) if hide_hydrogens else ()
    ref_atom_count = result["ref_mol"].GetNumAtoms()
    overlays = []
    if show == "deviations":
        overlays = _deviation_overlays(
            result["df_dev"],
            probe_offset=ref_atom_count,
            top_n=top_n,
        )

    cell = SceneCell(
        title=_comparison_title(result),
        models=[
            MoleculeModel(
                mol=result["ref_mol"],
                style=REFERENCE_STYLE,
                kekulize=kekulize,
                show_atom_labels=show_labels,
                show_charges=show_charges,
                hide_elements=hidden,
            ),
            MoleculeModel(
                mol=result["probe_mol_aligned"],
                style=PROBE_STYLE,
                kekulize=kekulize,
                show_atom_labels=show_labels,
                show_charges=show_charges,
                hide_elements=hidden,
            ),
        ],
        overlays=overlays,
    )
    return GridScene(
        cells=[cell],
        columns=1,
        cell_size=cell_size,
        linked=linked,
        background_color=background_color,
    )


def _compare_dataframe_row(
    df: pd.DataFrame,
    *,
    probe_col: str,
    ref_col: str,
    row_index: int,
    atom_scope: str,
    charge: int,
) -> dict[str, Any]:
    normalized = normalize_dataframe(df)
    _require_columns(normalized, ["atoms", probe_col, ref_col])
    row = normalized.iloc[int(row_index)]
    result = compare_symbols_coords_rmsd(
        row["atoms"],
        row[probe_col],
        row["atoms"],
        row[ref_col],
        atom_scope=atom_scope,
        charge=charge,
    )
    result.update(
        {
            "row_index": int(row_index),
            "probe_col": probe_col,
            "ref_col": ref_col,
            "probe_label": probe_col,
            "ref_label": ref_col,
            "row_label": _row_label(row),
        }
    )
    return result


def _finalize_comparison_result(
    result: dict[str, Any],
    *,
    show: str,
    render: bool,
    show_table: bool,
    table_rows: int,
    top_n: int,
    cell_size: tuple[int, int],
    linked: bool,
    background_color: str | tuple[str, float],
    show_labels: bool,
    show_charges: bool,
    kekulize: bool,
    hide_hydrogens: bool,
    export_HTML: str,
) -> dict[str, Any]:
    scene = None
    viewer = None
    if show != "none":
        scene = structure_comparison_scene(
            result,
            show=show,
            top_n=top_n,
            cell_size=cell_size,
            linked=linked,
            background_color=background_color,
            show_labels=show_labels,
            show_charges=show_charges,
            kekulize=kekulize,
            hide_hydrogens=hide_hydrogens,
        )
        if render or export_HTML:
            viewer = _render_comparison_scene(
                scene,
                render=render,
                export_HTML=export_HTML,
                hide_hydrogens=hide_hydrogens,
            )

    if show_table:
        _display_deviation_table(result["df_dev"], rows=table_rows)

    result.update({"scene": scene, "viewer": viewer})
    return result


def _deviation_overlays(
    df_dev: pd.DataFrame,
    *,
    probe_offset: int,
    top_n: int,
) -> list[Any]:
    overlays: list[Any] = []
    for rank, (_, row) in enumerate(df_dev.head(int(top_n)).iterrows(), start=1):
        probe_idx = int(row["probe_idx"])
        ref_idx = int(row["ref_idx"])
        probe_atom = probe_offset + probe_idx
        label = (
            f'{row["probe_symbol"]}{probe_idx} -> '
            f'{row["ref_symbol"]}{ref_idx}: '
            f'{float(row["distance_A"]):.3f} A'
        )
        overlays.extend(
            [
                AtomHighlight(atom=ref_idx, color="cyan", radius=0.45, alpha=0.25),
                AtomHighlight(atom=probe_atom, color="orange", radius=0.45, alpha=0.30),
                DistanceOverlay(
                    atom1=probe_atom,
                    atom2=ref_idx,
                    color="green",
                    radius=0.035,
                ),
                ScreenLabelOverlay(
                    text=f"{rank}. {label}",
                    font_color="green",
                    background_color="white",
                    border_color=None,
                    font_size=12,
                    screen_offset={
                        "x": DEVIATION_LABEL_OFFSET["x"],
                        "y": DEVIATION_LABEL_OFFSET["y"]
                        + (rank - 1) * DEVIATION_LABEL_STEP,
                    },
                ),
            ]
        )
    return overlays


def _render_comparison_scene(
    scene: GridScene,
    *,
    render: bool,
    export_HTML: str,
    hide_hydrogens: bool,
):
    """Render a comparison scene and restore model-specific styles."""
    renderer = Py3DmolGridRenderer(scene)
    viewer = renderer.render()
    _apply_comparison_model_styles(viewer, hide_hydrogens=hide_hydrogens)
    if export_HTML:
        renderer.write_html(export_HTML)
    if render:
        viewer.show()
    return viewer


def _apply_comparison_model_styles(viewer: Any, *, hide_hydrogens: bool) -> None:
    """Apply per-model styles after Tooltoad's cell-wide default styling."""
    viewer_position = (0, 0)
    viewer.setStyle({"model": 0}, REFERENCE_STYLE, viewer=viewer_position)
    viewer.setStyle({"model": 1}, PROBE_STYLE, viewer=viewer_position)
    if hide_hydrogens:
        viewer.setStyle({"elem": "H"}, {}, viewer=viewer_position)


def _display_deviation_table(df_dev: pd.DataFrame, *, rows: int) -> None:
    from IPython.display import display

    display(
        df_dev[
            [
                "probe_idx",
                "ref_idx",
                "probe_symbol",
                "ref_symbol",
                "distance_A",
            ]
        ]
        .head(rows)
        .style.format({"distance_A": "{:.4f}"})
    )


def _comparison_title(result: dict[str, Any]) -> str:
    df_dev = result["df_dev"]
    max_dev = 0.0 if df_dev.empty else float(df_dev["distance_A"].max())
    left = result.get("probe_label", "probe")
    right = result.get("ref_label", "reference")
    row_label = result.get("row_label")
    prefix = f"{row_label}\n" if row_label else ""
    return (
        f"{prefix}{left} -> {right}\n"
        f"RMSD {float(result['rmsd']):.3f} A | "
        f"{len(result['atom_map'])} mapped | max {max_dev:.3f} A"
    )


def _row_label(row: pd.Series) -> str:
    substrate_name = row.get("substrate_name", row.get("custom_name"))
    rpos = row.get("rpos")
    if substrate_name is None:
        return f"row {row.name}"
    if rpos is None or pd.isna(rpos):
        return str(substrate_name)
    return f"{substrate_name} r{rpos}"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: [{available}]"
        )


def _resolve_show_mode(
    show: str,
    *,
    show_overlay_plot: bool | None,
    show_deviation_overlay: bool | None,
) -> str:
    if show_deviation_overlay:
        return "deviations"
    if show_overlay_plot:
        return "overlay"
    return _validate_show_mode(show)


def _validate_show_mode(show: str) -> str:
    if show not in VALID_SHOW_MODES:
        valid = ", ".join(sorted(VALID_SHOW_MODES))
        raise ValueError(f"Invalid show mode {show!r}. Expected one of: {valid}.")
    return show
