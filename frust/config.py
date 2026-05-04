from pathlib import Path
import os

try:
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".env")
except Exception:
    pass

OET_TOOLS = os.environ.get("OET_TOOLS")
UMA_TOOLS = os.environ.get("UMA_TOOLS")


def get_oet_tools() -> Path:
    """Return the ORCA-External-Tools root, preferring the generic env name."""
    root = os.environ.get("OET_TOOLS") or os.environ.get("UMA_TOOLS")
    if not root:
        raise RuntimeError(
            "Set OET_TOOLS, or legacy UMA_TOOLS, to the orca-external-tools root."
        )

    path = Path(root).expanduser()
    if not path.exists():
        raise RuntimeError(f"ORCA-External-Tools path does not exist: {path}")
    return path
