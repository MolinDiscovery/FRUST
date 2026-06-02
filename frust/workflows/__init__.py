"""Lazy public namespace for FRUST workflow objects.

The workflow namespace exposes factory functions such as ``mols`` and
``screen_ts`` plus the ``methods`` submodule. Imports are resolved lazily so
``import frust as ft`` stays light while still supporting user-facing calls such
as ``ft.workflows.screen_ts(...)`` and ``ft.workflows.methods.preset(...)``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

_PUBLIC_MODULES: dict[str, str] = {
    "methods": "frust.workflows.methods",
}

_PUBLIC_API: dict[str, tuple[str, str]] = {
    "mols": ("frust.workflows.factories", "mols"),
    "screen_ts": ("frust.workflows.factories", "screen_ts"),
    "legacy_ts": ("frust.workflows.factories", "legacy_ts"),
    "int3": ("frust.workflows.factories", "int3"),
    "WorkflowTarget": ("frust.workflows.core", "WorkflowTarget"),
    "MethodPlan": ("frust.workflows.methods", "MethodPlan"),
    "CalculatorSpec": ("frust.workflows.methods", "CalculatorSpec"),
}

__all__ = sorted({*_PUBLIC_MODULES, *_PUBLIC_API})


def __getattr__(name: str) -> Any:
    """Lazily resolve workflow modules and public factories.

    Parameters
    ----------
    name : str
        Public workflow attribute requested from ``frust.workflows``.

    Returns
    -------
    object
        Imported public module, factory, or class.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the workflow public API.
    """
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
    """Return module globals plus lazy public workflow names.

    Returns
    -------
    list of str
        Names shown by ``dir(frust.workflows)``.
    """
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from frust.workflows import methods
    from frust.workflows.core import WorkflowTarget
    from frust.workflows.factories import int3, legacy_ts, mols, screen_ts
    from frust.workflows.methods import CalculatorSpec, MethodPlan
