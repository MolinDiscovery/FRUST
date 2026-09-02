"""Screen-level substrate/catalyst workflows."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from frust.screen.core import create_ts_guesses, expand, read

_LAZY_API = {
    "ReferenceLibrary": ("frust.screen.references", "ReferenceLibrary"),
    "ReferenceRecord": ("frust.screen.references", "ReferenceRecord"),
    "open_reference_library": ("frust.screen.references", "open_reference_library"),
    "repair_reference_bindings": (
        "frust.screen.repairs",
        "repair_reference_bindings",
    ),
    "ScreenRun": ("frust.screen.runs", "ScreenRun"),
    "build_analysis": ("frust.screen.runs", "build_analysis"),
    "open_run": ("frust.screen.runs", "open_run"),
}

__all__ = [
    "ReferenceLibrary",
    "ReferenceRecord",
    "ScreenRun",
    "build_analysis",
    "create_ts_guesses",
    "expand",
    "open_reference_library",
    "open_run",
    "read",
    "repair_reference_bindings",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve run-analysis and reference-library helpers."""
    try:
        module_name, attribute = _LAZY_API[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus lazy public screen names."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from frust.screen.references import ReferenceLibrary, ReferenceRecord, open_reference_library
    from frust.screen.repairs import repair_reference_bindings
    from frust.screen.runs import ScreenRun, build_analysis, open_run
