"""Top-level convenience API for FRUST.

Common user-facing functions are exposed lazily so ``import frust as ft`` stays
lightweight while notebook workflows can still use a compact namespace.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_MODULES: dict[str, str] = {
    "cluster": "frust.cluster",
    "pipelines": "frust.pipelines",
    "pipes": "frust.pipes",
    "screen": "frust.screen",
    "tsguess2": "frust.tsguess2",
    "utils": "frust.utils",
    "vis": "frust.vis",
    "workflows": "frust.workflows",
}

_PUBLIC_API: dict[str, tuple[str, str]] = {
    # Core calculation
    "Stepper": ("frust.stepper", "Stepper"),
    # Dataframe/result inspection
    "show_steps": ("frust.utils.dataframes", "show_steps"),
    "show_timing": ("frust.utils.dataframes", "show_timing"),
    "lowest_energy_rows": ("frust.utils.dataframes", "lowest_energy_rows"),
    "map_substrate_names": ("frust.utils.dataframes", "map_substrate_names"),
    "inspect_ts_vibrations": ("frust.utils.analytics", "inspect_ts_vibrations"),
    "summarize_ts_vibrations": ("frust.utils.analytics", "summarize_ts_vibrations"),
    # Structure preparation
    "create_mol_per_rpos": ("frust.utils.mols", "create_mol_per_rpos"),
    "create_ts_per_rpos": ("frust.utils.mols", "create_ts_per_rpos"),
    "embed_mols": ("frust.embedder", "embed_mols"),
    "embed_ts": ("frust.embedder", "embed_ts"),
    # File and structure IO
    "read_ts_type_from_xyz": ("frust.utils.io", "read_ts_type_from_xyz"),
    "write_xyz": ("frust.utils.io", "write_xyz"),
    "write_xyz_structures": ("frust.utils.io", "write_xyz_structures"),
    # Schema helpers
    "normalize_dataframe": ("frust.schema", "normalize_dataframe"),
    "energy_columns": ("frust.schema", "energy_columns"),
    "normal_termination_columns": ("frust.schema", "normal_termination_columns"),
    # Visualization
    "plot_vibs": ("frust.vis", "plot_vibs"),
    "plot_mols": ("frust.vis", "plot_mols"),
    "plot_row": ("frust.vis", "plot_row"),
    "plot_lig": ("frust.vis", "plot_lig"),
    "plot_rpos": ("frust.vis", "plot_rpos"),
    "plot_energy_profile": ("frust.vis", "plot_energy_profile"),
    "plot_regression_outliers": ("frust.vis", "plot_regression_outliers"),
    "MolTo3DGrid": ("frust.vis", "MolTo3DGrid"),
    "RxnTo3DGrid": ("frust.vis", "RxnTo3DGrid"),
    "DrawMolSvg": ("frust.vis", "DrawMolSvg"),
    "DrawUniqueChGrid": ("frust.vis", "DrawUniqueChGrid"),
    # Cluster helpers
    "ClusterConfig": ("frust.cluster", "ClusterConfig"),
    "Resources": ("frust.cluster", "Resources"),
    "submit_jobs": ("frust.cluster", "submit_jobs"),
    "submit_chain": ("frust.cluster", "submit_chain"),
    "submit_screen_chain": ("frust.cluster", "submit_screen_chain"),
}

__all__ = sorted({*_PUBLIC_MODULES, *_PUBLIC_API})


def __getattr__(name: str) -> Any:
    """Lazily resolve public FRUST API symbols."""
    if name in _PUBLIC_MODULES:
        module = import_module(_PUBLIC_MODULES[name])
        globals()[name] = module
        return module

    try:
        module_name, attr_name = _PUBLIC_API[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus lazy public API names."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    import frust.cluster as cluster
    import frust.pipelines as pipelines
    import frust.pipes as pipes
    import frust.screen as screen
    import frust.tsguess2 as tsguess2
    import frust.utils as utils
    import frust.vis as vis
    import frust.workflows as workflows
    from frust.cluster import ClusterConfig, Resources, submit_chain, submit_jobs, submit_screen_chain
    from frust.embedder import embed_mols, embed_ts
    from frust.schema import (
        energy_columns,
        normal_termination_columns,
        normalize_dataframe,
    )
    from frust.stepper import Stepper
    from frust.utils.analytics import inspect_ts_vibrations, summarize_ts_vibrations
    from frust.utils.dataframes import show_steps, show_timing, lowest_energy_rows, map_substrate_names
    from frust.utils.io import read_ts_type_from_xyz, write_xyz, write_xyz_structures
    from frust.utils.mols import create_mol_per_rpos, create_ts_per_rpos
    from frust.vis import (
        DrawMolSvg,
        DrawUniqueChGrid,
        MolTo3DGrid,
        RxnTo3DGrid,
        plot_energy_profile,
        plot_lig,
        plot_mols,
        plot_regression_outliers,
        plot_row,
        plot_rpos,
        plot_vibs,
    )
