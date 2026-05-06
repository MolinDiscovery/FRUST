from pathlib import Path
import os


def _dotenv_path() -> Path | None:
    """Return the first FRUST/Tooltoad dotenv path that exists."""
    configured = os.environ.get("TOOLTOAD_DOTENV_PATH")
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        if path.is_file():
            return path

    cwd_path = Path.cwd() / ".env"
    if cwd_path.is_file():
        return cwd_path

    try:
        home_path = Path.home() / ".env"
    except Exception:
        return None
    return home_path if home_path.is_file() else None


try:
    from dotenv import load_dotenv

    path = _dotenv_path()
    if path is not None:
        load_dotenv(path, override=True)
except Exception:
    pass

OET_TOOLS = os.environ.get("OET_TOOLS")


def get_oet_tools() -> Path:
    """Return the ORCA-External-Tools root."""
    root = os.environ.get("OET_TOOLS")
    if not root:
        raise RuntimeError("Set OET_TOOLS to the orca-external-tools root.")

    path = Path(root).expanduser()
    if not path.exists():
        raise RuntimeError(f"ORCA-External-Tools path does not exist: {path}")
    return path
