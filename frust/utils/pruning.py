"""Dataframe adapters for optional PRISM conformer pruning."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from frust.schema import infer_group_columns, normalize_dataframe
from frust.utils.timing import (
    build_step_timing,
    elapsed_seconds,
    monotonic_seconds,
    utc_timestamp,
)

DEFAULT_PRUNING_OPTIONS: dict[str, Any] = {
    "modes": ("moi", "rmsd"),
    "coords_col": None,
    "atoms_col": "atoms",
    "energy_col": None,
    "group_cols": None,
    "moi_max_deviation": 0.01,
    "rmsd_max_rmsd": 1.25,
    "rmsd_max_dev": None,
    "timeout_s": 60,
    "heavy_atoms_only": True,
    "graph_source": "connectivity_bonds",
}
VALID_PRUNING_MODES = {"moi", "rmsd", "rot_corr_rmsd"}


def normalize_pruning_options(value: bool | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize a workflow pruning option.

    Parameters
    ----------
    value : bool, mapping, or None
        ``False`` or ``None`` disables pruning. ``True`` returns the default
        PRISM pruning options. A mapping updates those defaults.

    Returns
    -------
    dict or None
        Normalized pruning options or ``None`` when disabled.
    """
    if value in (False, None):
        return None
    options = dict(DEFAULT_PRUNING_OPTIONS)
    if value is True:
        return options
    if isinstance(value, Mapping):
        options.update(dict(value))
        return options
    raise TypeError("prune_initial must be False, True, or a dictionary of pruning options")


def prune_conformers(
    df: pd.DataFrame,
    *,
    name: str = "initial_prune",
    modes: Sequence[str] = ("moi", "rmsd"),
    coords_col: str | None = None,
    atoms_col: str = "atoms",
    energy_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    moi_max_deviation: float = 0.01,
    rmsd_max_rmsd: float = 0.25,
    rmsd_max_dev: float | None = None,
    timeout_s: int = 60,
    heavy_atoms_only: bool = True,
    graph_source: str = "connectivity_bonds",
    debugfunction: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Prune geometrically redundant conformers from a FRUST dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        FRUST dataframe with one conformer per row.
    name : str, optional
        Step name recorded in ``df.attrs["frust_steps"]``.
    modes : sequence of str, optional
        PRISM pruning modes applied in order. Supported values are ``"moi"``,
        ``"rmsd"``, and ``"rot_corr_rmsd"``. ``"rot_corr_rmsd"`` is opt-in
        because it depends on molecular connectivity.
    coords_col : str or None, optional
        Coordinate column to prune on. If ``None``, the latest coordinate-like
        column is selected.
    atoms_col : str, optional
        Atom-symbol column matching the coordinate order.
    energy_col : str or None, optional
        Column used to sort rows within each group before pruning. Energies are
        not passed to PRISM in this adapter.
    group_cols : sequence of str or None, optional
        Columns defining independent conformer families. If ``None``, FRUST
        infers structure identity columns such as ``system_name``,
        ``substrate_name``, ``catalyst_name``, ``structure_type``,
        ``molecule_role``, and ``rpos``.
    moi_max_deviation : float, optional
        Relative moment-of-inertia deviation threshold for PRISM MOI pruning.
    rmsd_max_rmsd : float, optional
        RMSD threshold in Angstrom for PRISM RMSD pruning.
    rmsd_max_dev : float or None, optional
        Maximum single-atom deviation threshold in Angstrom. If ``None``, PRISM
        uses its default of ``2 * rmsd_max_rmsd``.
    timeout_s : int, optional
        PRISM timeout in seconds for iterative pruning modes.
    heavy_atoms_only : bool, optional
        Forwarded to PRISM RMSD pruning modes.
    graph_source : {"connectivity_bonds", "graphize"}, optional
        Connectivity source for ``"rot_corr_rmsd"``. ``"connectivity_bonds"``
        uses the dataframe column when available and falls back to PRISM
        distance-based graph construction.
    debugfunction : callable or None, optional
        Optional callback for PRISM debug messages.

    Returns
    -------
    pandas.DataFrame
        Pruned dataframe with original row indices and dataframe attrs
        preserved.
    """
    normalized_modes = _normalize_modes(modes)
    if not normalized_modes:
        out = normalize_dataframe(df)
        out.attrs.update(getattr(df, "attrs", {}))
        return out

    prune_moi, prune_rmsd, prune_rot_corr, graphize = _load_prism_functions()

    out = normalize_dataframe(df)
    out.attrs.update(getattr(df, "attrs", {}))
    resolved_coords_col = _resolve_coords_col(out, coords_col)
    _validate_columns(out, [resolved_coords_col, atoms_col])
    if energy_col is not None:
        _validate_columns(out, [energy_col])

    resolved_group_cols = _resolve_group_cols(out, group_cols)
    started_at = utc_timestamp()
    started = monotonic_seconds()
    kept: list[pd.DataFrame] = []
    group_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []

    for group_key, group in out.groupby(list(resolved_group_cols), dropna=False, sort=False):
        group_start = monotonic_seconds()
        work = group.copy()
        if energy_col is not None:
            work = work.sort_values(energy_col, na_position="last")

        atoms = np.asarray(work.iloc[0][atoms_col], dtype=str)
        if not all(list(row_atoms) == list(atoms) for row_atoms in work[atoms_col]):
            raise ValueError("All rows in a PRISM pruning group must have the same atom order.")

        coords = np.stack([np.asarray(coords, dtype=float) for coords in work[resolved_coords_col]])
        _validate_coords_shape(coords, atoms, resolved_coords_col)

        active_coords = coords
        active_indices = np.arange(len(work))
        mode_counts: list[dict[str, Any]] = []

        for mode in normalized_modes:
            before = int(len(active_indices))
            if before == 0:
                mode_counts.append(_mode_count(mode, before, before))
                continue

            if mode == "moi":
                _, mask = prune_moi(
                    active_coords,
                    atoms,
                    max_deviation=moi_max_deviation,
                    timeout_s=timeout_s,
                    debugfunction=debugfunction,
                )
            elif mode == "rmsd":
                _, mask = prune_rmsd(
                    active_coords,
                    atoms,
                    max_rmsd=rmsd_max_rmsd,
                    max_dev=rmsd_max_dev,
                    timeout_s=timeout_s,
                    heavy_atoms_only=heavy_atoms_only,
                    debugfunction=debugfunction,
                )
            elif mode == "rot_corr_rmsd":
                graph = _rot_corr_graph(
                    work.iloc[int(active_indices[0])],
                    atoms,
                    active_coords[0],
                    graph_source=graph_source,
                    graphize=graphize,
                )
                _, mask = prune_rot_corr(
                    active_coords,
                    atoms,
                    graph,
                    max_rmsd=rmsd_max_rmsd,
                    max_dev=rmsd_max_dev,
                    timeout_s=timeout_s,
                    heavy_atoms_only=heavy_atoms_only,
                    debugfunction=debugfunction,
                    logfunction=None,
                )
            else:  # pragma: no cover - guarded by _normalize_modes
                raise ValueError(f"Unsupported pruning mode: {mode!r}")

            mask = np.asarray(mask, dtype=bool)
            if len(mask) != len(active_indices):
                raise ValueError(
                    f"PRISM mode {mode!r} returned a mask of length {len(mask)} "
                    f"for {len(active_indices)} active rows"
                )
            active_coords = active_coords[mask]
            active_indices = active_indices[mask]
            mode_counts.append(_mode_count(mode, before, int(len(active_indices))))

        selected = work.iloc[active_indices]
        kept.append(selected)
        group_elapsed = elapsed_seconds(group_start)
        group_record = {
            "keys": _group_key_mapping(resolved_group_cols, group_key),
            "input_rows": int(len(group)),
            "output_rows": int(len(selected)),
            "dropped_rows": int(len(group) - len(selected)),
            "selected_cids": _cid_list(selected),
            "modes": mode_counts,
        }
        group_records.append(group_record)
        row_records.append(
            {
                "row_index": _format_group_key(group_key),
                "label": _format_group_key(group_key),
                "elapsed_s": group_elapsed,
                "normal_termination": True,
                "input_rows": int(len(group)),
                "output_rows": int(len(selected)),
            }
        )

    pruned = pd.concat(kept).sort_index() if kept else out.iloc[[]].copy()
    pruned.attrs.update(out.attrs)
    finished_at = utc_timestamp()
    steps = dict(pruned.attrs.get("frust_steps", {}))
    steps[str(name)] = {
        "engine": "prism_pruner",
        "columns": [],
        "options": {
            "modes": list(normalized_modes),
            "coords_col": resolved_coords_col,
            "atoms_col": atoms_col,
            "energy_col": energy_col,
            "group_cols": list(resolved_group_cols),
            "moi_max_deviation": float(moi_max_deviation),
            "rmsd_max_rmsd": float(rmsd_max_rmsd),
            "rmsd_max_dev": rmsd_max_dev,
            "timeout_s": int(timeout_s),
            "heavy_atoms_only": bool(heavy_atoms_only),
            "graph_source": graph_source,
        },
        "row_counts": {
            "input_rows": int(len(out)),
            "output_rows": int(len(pruned)),
            "dropped_rows": int(len(out) - len(pruned)),
        },
        "filtering": {
            "kind": "geometry_pruning",
            "modes": list(normalized_modes),
            "energy_col": energy_col,
            "group_cols": list(resolved_group_cols),
            "input_rows": int(len(out)),
            "output_rows": int(len(pruned)),
            "dropped_rows": int(len(out) - len(pruned)),
            "n_groups": int(len(group_records)),
            "groups": group_records,
        },
        "calculator": {
            "name": "prism_pruner",
            "mode": "geometry_pruning",
            "backend": "prism_pruner.pruner",
            "resources": {},
            "executables": {},
        },
        "timing": build_step_timing(
            started_at=started_at,
            finished_at=finished_at,
            elapsed_s=elapsed_seconds(started),
            input_rows=int(len(out)),
            output_rows=int(len(pruned)),
            processed_rows=int(len(out)),
            skipped_rows=0,
            row_records=row_records,
        ),
    }
    pruned.attrs["frust_steps"] = steps
    return pruned


def _load_prism_functions():
    try:
        pruner = importlib.import_module("prism_pruner.pruner")
        graph_module = importlib.import_module("prism_pruner.graph_manipulations")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("prism_pruner"):
            raise ImportError(
                "PRISM pruning requires the optional `prism_pruner` package. "
                "Install it in the active environment, for example with "
                "`pip install prism-pruner`."
            ) from exc
        raise
    return (
        pruner.prune_by_moment_of_inertia,
        pruner.prune_by_rmsd,
        pruner.prune_by_rmsd_rot_corr,
        graph_module.graphize,
    )


def _normalize_modes(modes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(modes, str):
        modes = (modes,)
    normalized = tuple(str(mode).strip().lower() for mode in modes)
    unknown = sorted(set(normalized) - VALID_PRUNING_MODES)
    if unknown:
        supported = ", ".join(sorted(VALID_PRUNING_MODES))
        raise ValueError(f"Unsupported pruning mode(s): {unknown}. Supported: {supported}")
    return normalized


def _resolve_coords_col(df: pd.DataFrame, coords_col: str | None) -> str:
    if coords_col is not None:
        _validate_columns(df, [coords_col])
        return str(coords_col)
    coord_cols = _coordinate_columns(df)
    if not coord_cols:
        raise ValueError(
            "cannot prune conformers: no coordinate column found. Expected "
            "'coords_embedded' or a column ending in '-oc'/'-opt_coords'."
        )
    return coord_cols[-1]


def _coordinate_columns(df: pd.DataFrame) -> list[str]:
    preferred = []
    if "coords_embedded" in df.columns:
        preferred.append("coords_embedded")
    preferred.extend(
        [
            str(col)
            for col in df.columns
            if "coords" in str(col) and str(col) not in preferred and not str(col).endswith("-opt_coords")
        ]
    )
    preferred.extend(
        [
            str(col)
            for col in df.columns
            if str(col).endswith("-opt_coords") or str(col).endswith("-oc")
        ]
    )
    return preferred


def _resolve_group_cols(
    df: pd.DataFrame,
    group_cols: Sequence[str] | None,
) -> tuple[str, ...]:
    if group_cols is None:
        resolved = infer_group_columns(df)
        if not resolved:
            raise ValueError("cannot prune conformers: no structure identity columns found")
        return tuple(resolved)

    resolved = tuple(str(col) for col in group_cols)
    if not resolved:
        raise ValueError("cannot prune conformers: group_cols must contain at least one column")
    _validate_columns(df, resolved)
    return resolved


def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Missing required dataframe column(s): {missing}. "
            f"Available columns: [{available}]"
        )


def _validate_coords_shape(coords: np.ndarray, atoms: np.ndarray, coords_col: str) -> None:
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(
            f"Coordinate column {coords_col!r} must contain arrays shaped (n_atoms, 3)"
        )
    if coords.shape[1] != len(atoms):
        raise ValueError(
            f"Coordinate column {coords_col!r} has {coords.shape[1]} atoms, "
            f"but atoms column has {len(atoms)} symbols"
        )


def _rot_corr_graph(
    row: pd.Series,
    atoms: np.ndarray,
    coords: np.ndarray,
    *,
    graph_source: str,
    graphize,
):
    source = str(graph_source).strip().lower()
    if source not in {"connectivity_bonds", "graphize"}:
        raise ValueError("graph_source must be 'connectivity_bonds' or 'graphize'")

    if source == "connectivity_bonds" and "connectivity_bonds" in row.index:
        bonds = _bond_pairs(row.get("connectivity_bonds"))
        if bonds:
            return _graph_from_bonds(atoms, bonds)
    return graphize(atoms, coords)


def _bond_pairs(value: Any) -> list[tuple[int, int]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return []

    pairs: list[tuple[int, int]] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if len(item) != 2:
            continue
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def _graph_from_bonds(atoms: np.ndarray, bonds: Sequence[tuple[int, int]]):
    nx = importlib.import_module("networkx")
    graph = nx.Graph()
    for idx, atom in enumerate(atoms):
        graph.add_node(int(idx), atoms=str(atom))
    graph.add_edges_from((int(i), int(j)) for i, j in bonds)
    return graph


def _mode_count(mode: str, input_rows: int, output_rows: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "dropped_rows": int(input_rows - output_rows),
    }


def _group_key_mapping(columns: Sequence[str], key: Any) -> dict[str, Any]:
    values = key if isinstance(key, tuple) else (key,)
    return {
        str(col): _metadata_scalar(value)
        for col, value in zip(columns, values)
    }


def _format_group_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(map(str, key))
    return str(key)


def _metadata_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cid_list(df: pd.DataFrame) -> list[int]:
    if "cid" not in df.columns:
        return []
    cids: list[int] = []
    for value in df["cid"].tolist():
        value = _metadata_scalar(value)
        if value is None:
            continue
        try:
            cids.append(int(value))
        except (TypeError, ValueError):
            continue
    return cids
