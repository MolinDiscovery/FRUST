"""Staged workflow modules for FRUST pipeline chains."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

_PUBLIC_MODULES: dict[str, str] = {
    "run_int3_per_rpos": "frust.pipelines.run_int3_per_rpos",
    "run_screen_ts_per_rpos": "frust.pipelines.run_screen_ts_per_rpos",
    "run_struct": "frust.pipelines.run_struct",
    "run_ts_per_rpos": "frust.pipelines.run_ts_per_rpos",
}

__all__ = sorted(_PUBLIC_MODULES)


def __getattr__(name: str) -> ModuleType:
    """Lazily resolve public staged pipeline modules."""
    try:
        module_name = _PUBLIC_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return module globals plus lazy staged module names."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from frust.pipelines import run_int3_per_rpos, run_screen_ts_per_rpos, run_struct, run_ts_per_rpos
