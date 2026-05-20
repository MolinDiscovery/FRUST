"""Lightweight calculator provenance helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping


def backend_name(fn: Callable[..., Any]) -> str:
    """Return a stable, human-readable callable name.

    Parameters
    ----------
    fn : callable
        Calculator backend function or callable object.

    Returns
    -------
    str
        Fully qualified callable name when available.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return str(qualname)
    return type(fn).__name__


def executable_entry(
    configured: str | Path | None,
    source: str,
    *,
    fallback_command: str | None = None,
    resolved_path: str | Path | None = None,
) -> dict[str, object]:
    """Describe executable provenance without raising on missing paths.

    Parameters
    ----------
    configured : str, pathlib.Path, or None
        User/environment value used to configure the executable.
    source : str
        Source label, such as ``"XTB_EXE"`` or ``"PATH"``.
    fallback_command : str or None, optional
        Command to resolve from ``PATH`` when ``configured`` is missing.
    resolved_path : str, pathlib.Path, or None, optional
        Already-resolved path from a stricter caller.

    Returns
    -------
    dict
        JSON-friendly executable metadata.
    """
    configured_str = str(configured) if configured is not None else None

    if resolved_path is not None:
        path = _display_path(resolved_path)
        return {
            "path": path,
            "configured": configured_str,
            "source": source,
            "resolved": path is not None,
        }

    if configured_str:
        path, resolved = _resolve_executable(configured_str)
        return {
            "path": path,
            "configured": configured_str,
            "source": source,
            "resolved": resolved,
        }

    if fallback_command:
        path, resolved = _resolve_executable(fallback_command)
        return {
            "path": path,
            "configured": fallback_command,
            "source": "PATH",
            "resolved": resolved,
        }

    return {
        "path": None,
        "configured": None,
        "source": source,
        "resolved": False,
    }


def env_executable(env_var: str, *, fallback_command: str | None = None) -> dict[str, object]:
    """Describe an executable configured by an environment variable.

    Parameters
    ----------
    env_var : str
        Environment variable name.
    fallback_command : str or None, optional
        Command to resolve from ``PATH`` when the environment variable is not
        set.

    Returns
    -------
    dict
        JSON-friendly executable metadata.
    """
    value = os.environ.get(env_var)
    if value:
        return executable_entry(value, env_var)
    return executable_entry(None, env_var, fallback_command=fallback_command)


def gxtb_executable(
    gxtb_exe: str | Path | None = None,
    *,
    resolved_path: str | Path | None = None,
) -> dict[str, object]:
    """Describe the g-xTB executable source.

    Parameters
    ----------
    gxtb_exe : str, pathlib.Path, or None, optional
        Explicit ``Stepper.orca(..., gxtb_exe=...)`` value.
    resolved_path : str, pathlib.Path, or None, optional
        Already-validated path from the ORCA-driven g-xTB setup.

    Returns
    -------
    dict
        JSON-friendly executable metadata.
    """
    if gxtb_exe is not None:
        return executable_entry(gxtb_exe, "gxtb_exe", resolved_path=resolved_path)
    return executable_entry(
        os.environ.get("GXTB_EXE"),
        "GXTB_EXE",
        resolved_path=resolved_path,
    )


def env_path(env_var: str) -> dict[str, object]:
    """Describe a path-like environment variable.

    Parameters
    ----------
    env_var : str
        Environment variable name.

    Returns
    -------
    dict
        JSON-friendly path metadata.
    """
    value = os.environ.get(env_var)
    return path_entry(value if value else None, env_var)


def path_entry(configured: str | Path | None, source: str) -> dict[str, object]:
    """Describe path provenance without requiring the path to exist.

    Parameters
    ----------
    configured : str, pathlib.Path, or None
        Configured path value.
    source : str
        Source label.

    Returns
    -------
    dict
        JSON-friendly path metadata.
    """
    if configured is None:
        return {
            "path": None,
            "configured": None,
            "source": source,
            "resolved": False,
        }

    configured_str = str(configured)
    path = Path(os.path.expandvars(configured_str)).expanduser()
    return {
        "path": str(path.resolve()) if path.exists() else str(path),
        "configured": configured_str,
        "source": source,
        "resolved": path.exists(),
    }


def oet_executable(name: str) -> dict[str, object]:
    """Describe an OET executable under ``OET_TOOLS/bin``.

    Parameters
    ----------
    name : str
        Executable filename, such as ``"oet_gxtb"``.

    Returns
    -------
    dict
        JSON-friendly executable metadata.
    """
    root = os.environ.get("OET_TOOLS")
    if not root:
        return executable_entry(None, "OET_TOOLS")
    return executable_entry(Path(root).expanduser() / "bin" / name, "OET_TOOLS")


def calculator_provenance(
    *,
    name: str,
    mode: str,
    backend: Callable[..., Any],
    resources: Mapping[str, Any] | None = None,
    executables: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    **metadata: Any,
) -> dict[str, object]:
    """Build a JSON-friendly calculator provenance block.

    Parameters
    ----------
    name : str
        Calculator name, such as ``"xtb"`` or ``"orca"``.
    mode : str
        Calculator mode, such as ``"direct"`` or ``"orca_external_uma"``.
    backend : callable
        Backend wrapper used by the Stepper stage.
    resources : mapping or None, optional
        Effective resource settings.
    executables : mapping or None, optional
        Executable provenance entries keyed by logical executable name.
    environment : mapping or None, optional
        Path-like environment provenance entries.
    **metadata
        Extra JSON-friendly calculator metadata.

    Returns
    -------
    dict
        Calculator provenance block suitable for ``DataFrame.attrs``.
    """
    data: dict[str, object] = {
        "name": name,
        "mode": mode,
        "backend": backend_name(backend),
        "resources": _json_ready(dict(resources or {})),
        "executables": _json_ready(dict(executables or {})),
    }
    if environment:
        data["environment"] = _json_ready(dict(environment))
    for key, value in metadata.items():
        if value is not None:
            data[key] = _json_ready(value)
    return data


def _resolve_executable(value: str) -> tuple[str | None, bool]:
    expanded = os.path.expandvars(value)
    path = Path(expanded).expanduser()
    is_pathlike = path.is_absolute() or any(sep and sep in expanded for sep in (os.sep, os.altsep))

    if is_pathlike:
        display = str(path.resolve()) if path.exists() else str(path)
        return display, path.exists() and os.access(path, os.X_OK)

    found = shutil.which(expanded)
    if found:
        return str(Path(found).resolve()), True
    return None, False


def _display_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
