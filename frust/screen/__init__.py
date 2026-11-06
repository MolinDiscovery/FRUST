"""Screen-level substrate/catalyst workflows."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from frust.screen.core import create_ts_guesses, expand, read
from frust.screen.stereo import resolve_stereoisomers

_LAZY_API = {
    "ScreenRun": ("frust.screen.runs", "ScreenRun"),
    "build_analysis": ("frust.screen.runs", "build_analysis"),
    "open_run": ("frust.screen.runs", "open_run"),
    "cleanup_submitit": ("frust.screen.cleanup", "cleanup_submitit"),
}

__all__ = [
    "ScreenRun",
    "build_analysis",
    "cleanup_submitit",
    "create_ts_guesses",
    "expand",
    "open_run",
    "read",
    "resolve_stereoisomers",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve run-analysis and cleanup helpers."""
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
    from frust.screen.cleanup import cleanup_submitit
    from frust.screen.runs import ScreenRun, build_analysis, open_run
