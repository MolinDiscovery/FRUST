"""Visualization utilities for FRUST data and molecular structures.

The package keeps the historical ``frust.vis`` import surface while resolving
visualization helpers lazily so ``import frust as ft`` and ``ft.vis`` stay
lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_API: dict[str, tuple[str, str]] = {
    "DrawMolSvg": ("tooltoad.vis", "DrawMolSvg"),
    "MolTo3DGrid": ("tooltoad.vis", "MolTo3DGrid"),
    "RxnTo3DGrid": ("tooltoad.vis", "RxnTo3DGrid"),
    "GridScene": ("tooltoad.scene3d", "GridScene"),
    "SceneCell": ("tooltoad.scene3d", "SceneCell"),
    "MoleculeModel": ("tooltoad.scene3d", "MoleculeModel"),
    "VibrationAnimation": ("tooltoad.scene3d", "VibrationAnimation"),
    "AtomLabel": ("tooltoad.scene3d", "AtomLabel"),
    "AtomHighlight": ("tooltoad.scene3d", "AtomHighlight"),
    "DistanceOverlay": ("tooltoad.scene3d", "DistanceOverlay"),
    "Py3DmolGridRenderer": ("tooltoad.scene3d", "Py3DmolGridRenderer"),
    "DrawUniqueChGrid": ("frust.vis.aromatic", "DrawUniqueChGrid"),
    "plot_energy_profile": ("frust.vis.energy_profile", "plot_energy_profile"),
    "plot_lig": ("frust.vis.molecules", "plot_lig"),
    "plot_mols": ("frust.vis.molecules", "plot_mols"),
    "plot_row": ("frust.vis.molecules", "plot_row"),
    "plot_rpos": ("frust.vis.molecules", "plot_rpos"),
    "plot_regression_outliers": ("frust.vis.regression", "plot_regression_outliers"),
    "set_theme": ("frust.vis.theme", "set_theme"),
    "use_darkmode": ("frust.vis.theme", "use_darkmode"),
    "plot_vibs": ("frust.vis.vibrations", "plot_vibs"),
    "show_scene": ("frust.vis.scenes", "show_scene"),
    "molecule_scene_from_dataframe": ("frust.vis.scenes", "molecule_scene_from_dataframe"),
    "vibration_scene_from_dataframe": ("frust.vis.scenes", "vibration_scene_from_dataframe"),
    "ts_guess_scene_from_dataframe": ("frust.vis.scenes", "ts_guess_scene_from_dataframe"),
    "ts_guess_scene": ("frust.vis.scenes", "ts_guess_scene_from_dataframe"),
}

__all__ = sorted(_PUBLIC_API)


def __getattr__(name: str) -> Any:
    """Lazily resolve public visualization helpers."""
    try:
        module_name, attr_name = _PUBLIC_API[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus lazy public visualization names."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from tooltoad.vis import DrawMolSvg, MolTo3DGrid, RxnTo3DGrid
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

    from frust.vis.aromatic import DrawUniqueChGrid
    from frust.vis.energy_profile import plot_energy_profile
    from frust.vis.molecules import plot_lig, plot_mols, plot_row, plot_rpos
    from frust.vis.regression import plot_regression_outliers
    from frust.vis.scenes import (
        molecule_scene_from_dataframe,
        show_scene,
        ts_guess_scene_from_dataframe,
        vibration_scene_from_dataframe,
    )
    from frust.vis.theme import set_theme, use_darkmode
    from frust.vis.vibrations import plot_vibs
