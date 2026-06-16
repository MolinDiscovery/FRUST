"""Scene-based RMSD comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    read_xyz,
    read_xyz_block,
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


@dataclass(frozen=True)
class _StructureRecord:
    atoms: list[str]
    coords: Any
    label: str
    source_kind: str


def compare_rmsd(
    probe: Any,
    ref: Any,
    *,
    atom_scope: str = "heavy",
    charge: int = 0,
    mapping: str = "topology",
    atom_map: Sequence[tuple[int, int]] | None = None,
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
    """Compare two molecular structures by RMSD.

    This is the general structure-comparison entry point. It accepts two
    independently resolved structure-like inputs, aligns the ``probe`` to the
    ``ref`` structure, computes an RMSD over mapped atoms, and optionally
    renders a scene overlay with the largest atom deviations.

    The important idea is that each side can come from a different source. For
    example, a structure read from an XYZ file can be compared directly with
    coordinates stored in a FRUST dataframe row:

    Examples
    --------
    Compare an XYZ file on disk with a coordinate column from a dataframe:

    >>> import frust as ft
    >>> result = ft.vis.compare_rmsd(
    ...     {"path": "font2017.xyz", "label": "Font 2017"},
    ...     {
    ...         "df": df,
    ...         "row_index": 0,
    ...         "coords_col": "wb97-oc",
    ...         "label": "FRUST wb97",
    ...     },
    ...     top_n=5,
    ... )
    >>> result["rmsd"]
    >>> result["df_dev"].head()

    Compare two dataframe columns from the same row:

    >>> result = ft.vis.compare_rmsd(
    ...     {"df": df, "row_index": 0, "coords_col": "gxtb-oc"},
    ...     {"df": df, "row_index": 0, "coords_col": "orca-oc"},
    ... )

    Compare an XYZ block with tooltoad-style atoms and coordinates:

    >>> result = ft.vis.compare_rmsd(
    ...     {"xyz": xyz_block, "label": "external guess"},
    ...     (row["atoms"], row["coords_embedded"]),
    ...     mapping="index",
    ... )

    Accepted structure inputs are:

    ``str`` or ``pathlib.Path``
        Path to an XYZ file. Raw XYZ text is intentionally not inferred from a
        bare string; pass raw XYZ text as ``{"xyz": xyz_block}`` so accidental
        file/path mixups fail clearly.
    ``{"path": path, "label": label}``
        Explicit XYZ file input. ``label`` is optional and defaults to the file
        stem.
    ``{"xyz": xyz_block, "label": label}``
        Raw XYZ text. ``label`` is optional and defaults to ``"probe"`` or
        ``"reference"``.
    ``(atoms, coords)``
        Tooltoad-style pair where ``atoms`` is a sequence of atomic symbols and
        ``coords`` has shape ``(n_atoms, 3)``.
    ``{"atoms": atoms, "coords": coords, "label": label}``
        Explicit atoms/coordinates input.
    ``{"df": df, "row_index": i, "coords_col": col, "atoms_col": "atoms"}``
        Coordinates from ``df.iloc[i][col]`` and atoms from
        ``df.iloc[i]["atoms"]``. ``row_index`` defaults to ``0`` and
        ``atoms_col`` defaults to ``"atoms"``.
    ``{"row": row, "coords_col": col, "atoms_col": "atoms"}``
        Coordinates from a selected dataframe row or Series.

    Parameters
    ----------
    probe
        Structure that will be aligned to the reference. See the accepted input
        forms above.
    ref
        Reference structure. See the accepted input forms above.
    atom_scope
        Atom scope used for mapping and RMSD. Currently only ``"heavy"`` is
        supported, meaning hydrogens are ignored during automatic mapping and
        RMSD calculation. Hydrogens may still be present in the displayed
        structures.
    charge
        Total molecular charge used during RDKit bond perception when
        ``mapping="topology"``.
    mapping
        Automatic atom-mapping strategy used when ``atom_map`` is not supplied.
        Use ``"topology"`` for the default RDKit bond perception plus
        heavy-atom graph matching. This is the best default when the two inputs
        are the same molecule but atom order may differ. Use ``"index"`` when
        the atom order is already meaningful; heavy atoms are paired in input
        order, and RDKit bond perception/topology matching is skipped. This is
        useful for transition-state guesses, constrained geometries, or other
        structures where bond perception is unreliable.
    atom_map
        Optional explicit atom correspondence as ``(probe_idx, ref_idx)``
        pairs in the original atom ordering. When supplied, it takes precedence
        over ``mapping`` and also skips bond perception/topology matching. For
        ``atom_scope="heavy"``, mapped atoms must be non-hydrogen atoms with
        matching element symbols.
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

    Returns
    -------
    dict
        RMSD result containing:

        ``rmsd``
            RMSD after aligning the probe to the reference.
        ``mapping``
            Mapping strategy actually used: ``"topology"``, ``"index"``, or
            ``"explicit"``.
        ``atom_map``
            Atom-index pairs as ``(probe_idx, ref_idx)`` in original input
            ordering.
        ``df_dev``
            Per-atom mapped deviations sorted from largest to smallest.
        ``probe_mol_aligned`` and ``ref_mol``
            RDKit molecules with coordinates after alignment.
        ``scene`` and ``viewer``
            Scene and py3Dmol viewer objects when requested.

    Raises
    ------
    ValueError
        If a structure input cannot be resolved, coordinates are malformed,
        mapping options are unsupported, atom correspondence is invalid, or
        topology-based mapping fails.
    KeyError
        If a dataframe input is missing ``atoms_col`` or ``coords_col``.
    """
    show = _validate_show_mode(show)
    probe_record = _resolve_structure_input(probe, role="probe")
    ref_record = _resolve_structure_input(ref, role="reference")
    result = compare_symbols_coords_rmsd(
        probe_record.atoms,
        probe_record.coords,
        ref_record.atoms,
        ref_record.coords,
        atom_scope=atom_scope,
        charge=charge,
        mapping=mapping,
        atom_map=atom_map,
    )
    result.update(
        {
            "probe_label": probe_record.label,
            "ref_label": ref_record.label,
            "probe_source": probe_record.source_kind,
            "ref_source": ref_record.source_kind,
        }
    )
    if print_summary:
        _print_comparison_summary(result)
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


def _resolve_structure_input(value: Any, *, role: str) -> _StructureRecord:
    if isinstance(value, PathLike):
        return _structure_from_path(Path(value), label=None)

    if isinstance(value, str):
        if "\n" in value:
            raise ValueError(
                "Raw XYZ text must be passed as {'xyz': xyz_block}; bare "
                "strings are interpreted as XYZ file paths."
            )
        return _structure_from_path(Path(value), label=None)

    if isinstance(value, tuple) and len(value) == 2:
        atoms, coords = value
        return _StructureRecord(
            atoms=list(atoms),
            coords=coords,
            label=role,
            source_kind="atoms_coords",
        )

    if isinstance(value, Mapping):
        if "path" in value:
            return _structure_from_path(
                Path(value["path"]),
                label=_optional_label(value),
            )
        if "xyz" in value:
            label = _optional_label(value) or role
            atoms, coords = read_xyz_block(value["xyz"], source=label)
            return _StructureRecord(
                atoms=atoms,
                coords=coords,
                label=label,
                source_kind="xyz_block",
            )
        if "atoms" in value and "coords" in value:
            return _StructureRecord(
                atoms=list(value["atoms"]),
                coords=value["coords"],
                label=_optional_label(value) or role,
                source_kind="atoms_coords",
            )
        if "df" in value:
            df = value["df"]
            if not isinstance(df, pd.DataFrame):
                raise TypeError("{'df': ...} inputs require a pandas DataFrame.")
            row_index = int(value.get("row_index", 0))
            row = normalize_dataframe(df).iloc[row_index]
            return _structure_from_row(
                row,
                coords_col=_required_spec_key(value, "coords_col"),
                atoms_col=str(value.get("atoms_col", "atoms")),
                label=_optional_label(value),
            )
        if "row" in value:
            row = value["row"]
            if not isinstance(row, pd.Series):
                row = pd.Series(row)
            return _structure_from_row(
                row,
                coords_col=_required_spec_key(value, "coords_col"),
                atoms_col=str(value.get("atoms_col", "atoms")),
                label=_optional_label(value),
            )

    raise ValueError(
        f"Could not resolve {role} structure input. Expected an XYZ path, "
        "{'path': ...}, {'xyz': ...}, (atoms, coords), "
        "{'atoms': ..., 'coords': ...}, {'df': ..., 'coords_col': ...}, or "
        "{'row': ..., 'coords_col': ...}."
    )


def _structure_from_path(path: Path, *, label: str | None) -> _StructureRecord:
    atoms, coords = read_xyz(str(path))
    return _StructureRecord(
        atoms=atoms,
        coords=coords,
        label=label or path.stem,
        source_kind="xyz_path",
    )


def _structure_from_row(
    row: pd.Series,
    *,
    coords_col: str,
    atoms_col: str,
    label: str | None,
) -> _StructureRecord:
    _require_row_columns(row, [atoms_col, coords_col])
    return _StructureRecord(
        atoms=list(row[atoms_col]),
        coords=row[coords_col],
        label=label or _row_structure_label(row, coords_col),
        source_kind="dataframe",
    )


def _optional_label(spec: Mapping[str, Any]) -> str | None:
    label = spec.get("label")
    if label is None:
        return None
    return str(label)


def _required_spec_key(spec: Mapping[str, Any], key: str) -> str:
    if key not in spec:
        raise KeyError(f"Structure input is missing required key {key!r}.")
    return str(spec[key])


def _require_row_columns(row: pd.Series, columns: list[str]) -> None:
    missing = [column for column in columns if column not in row.index]
    if missing:
        available = ", ".join(map(str, row.index))
        raise KeyError(
            f"Missing required row field(s): {missing}. "
            f"Available fields: [{available}]"
        )


def _row_structure_label(row: pd.Series, coords_col: str) -> str:
    row_label = _row_label(row)
    if row_label:
        return f"{row_label} {coords_col}"
    return coords_col


def _print_comparison_summary(result: dict[str, Any]) -> None:
    print(f"Probe: {result['probe_label']}")
    print(f"Reference: {result['ref_label']}")
    print(f"Atom scope: {result['atom_scope']}")
    print(f"Mapping: {result['mapping']}")
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


def _validate_show_mode(show: str) -> str:
    if show not in VALID_SHOW_MODES:
        valid = ", ".join(sorted(VALID_SHOW_MODES))
        raise ValueError(f"Invalid show mode {show!r}. Expected one of: {valid}.")
    return show
