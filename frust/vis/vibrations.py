"""Vibration visualization helpers."""

from __future__ import annotations

from tooltoad.scene3d import Py3DmolGridRenderer

from frust.vis.scenes import (
    missing_vibrations as _missing_vibrations,
    select_vibration_column as _select_vibration_column,
    select_vibration_coords_column as _select_vibration_coords_column,
    vibration_scene_from_dataframe,
)


def plot_vibs(
    df,
    row_index=0,
    vId: int = 0,
    width: float = 600,
    height: float = 400,
    numFrames: int = 20,
    amplitude: float = 1,
    transparent: bool = False,
    fps: float | None = None,
    reps: int = 50,
    custom_coords_col_name: str | None = None,
    row_indices: list[int] | str | None = None,
    viewergrid: tuple[int, int] | None = None,
    columns: int | None = None,
    max_rows: int | None = None,
    cell_size: tuple[int, int] | None = None,
    linked: bool = True,
    freq_label: bool = True,
    legends: list[str] | None = None,
    legend_screen_offset: dict | None = None,
    export_HTML: str = "",
):
    """Display normal-mode vibrations from a FRUST dataframe.

    Parameters
    ----------
    df
        Dataframe containing ``atoms``, coordinates, and a vibration column.
    row_index
        Single row to display when ``row_indices`` is not provided.
    vId
        Vibration mode index to animate.
    width, height
        Historical viewer size. For multi-row grids without ``columns`` these
        are treated as total grid size. With ``columns``, prefer ``cell_size``
        for explicit per-cell sizing.
    numFrames
        Number of animation frames.
    amplitude
        Vibration displacement amplitude.
    transparent
        Use transparent background. Defaults to ``False`` so vibration grids
        match the standard molecule-grid appearance.
    fps
        Animation frames per second. ``None`` keeps py3Dmol's historical
        50 ms interval.
    reps
        Number of animation repetitions.
    custom_coords_col_name
        Explicit coordinate column override.
    row_indices
        Sequence of row positions, ``"all"``, or ``None`` for ``row_index``.
    viewergrid
        Historical explicit ``(rows, columns)`` grid shape.
    columns
        Plot-mols style column count for multi-row grids.
    max_rows
        Optional cap, mainly useful with ``row_indices="all"``.
    cell_size
        Per-cell size when using the scene-grid path.
    linked
        Link rotations/zoom across cells.
    freq_label
        Show frequency/title labels.
    legends
        Optional per-cell label text.
    legend_screen_offset
        Accepted for compatibility. Custom offsets are not yet implemented in
        the scene renderer.
    export_HTML
        Optional HTML export path.

    Returns
    -------
    py3Dmol.view
        Rendered viewer.
    """

    if legend_screen_offset is not None:
        print("legend_screen_offset is not yet supported by the scene renderer.")

    vibs_col = _select_vibration_column(df)
    coords_col = _select_vibration_coords_column(
        df,
        vibs_col,
        custom_coords_col_name=custom_coords_col_name,
    )
    print(f"vibs col {vibs_col}")
    print(f"coords col {coords_col}")

    scene = vibration_scene_from_dataframe(
        df,
        row_index=row_index,
        row_indices=row_indices,
        max_rows=max_rows,
        vId=vId,
        custom_coords_col_name=coords_col,
        columns=columns,
        viewergrid=viewergrid,
        width=width,
        height=height,
        cell_size=cell_size,
        numFrames=numFrames,
        amplitude=amplitude,
        transparent=transparent,
        fps=fps,
        reps=reps,
        linked=linked,
        freq_label=freq_label,
        legends=legends,
    )
    renderer = Py3DmolGridRenderer(scene)
    viewer = renderer.render()

    if export_HTML:
        try:
            renderer.write_html(export_HTML)
            print(f"HTML export successful: {export_HTML}")
        except Exception as e:
            print(f"Error exporting HTML to '{export_HTML}': {e}")

    return viewer
