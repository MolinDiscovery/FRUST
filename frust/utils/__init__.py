"""Utility helpers for FRUST."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_API: dict[str, tuple[str, str]] = {
    "show_steps": ("frust.utils.dataframes", "show_steps"),
    "lowest_energy_rows": ("frust.utils.dataframes", "lowest_energy_rows"),
    "summarize_ts_vibrations": ("frust.utils.analytics", "summarize_ts_vibrations"),
    "create_mol_per_rpos": ("frust.utils.mols", "create_mol_per_rpos"),
    "create_ts_per_rpos": ("frust.utils.mols", "create_ts_per_rpos"),
    "read_ts_type_from_xyz": ("frust.utils.io", "read_ts_type_from_xyz"),
    "write_xyz": ("frust.utils.io", "write_xyz"),
    "write_xyz_structures": ("frust.utils.io", "write_xyz_structures"),
}

__all__ = sorted(_PUBLIC_API)


def __getattr__(name: str) -> Any:
    """Lazily resolve public utility helpers."""
    try:
        module_name, attr_name = _PUBLIC_API[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus lazy public utility names."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from frust.utils.analytics import summarize_ts_vibrations
    from frust.utils.dataframes import show_steps, lowest_energy_rows
    from frust.utils.io import read_ts_type_from_xyz, write_xyz, write_xyz_structures
    from frust.utils.mols import create_mol_per_rpos, create_ts_per_rpos
