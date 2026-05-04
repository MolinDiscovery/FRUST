import os
import shlex
from pathlib import Path

from frust.config import get_oet_tools


def get_gxtb_exe(gxtb_exe: str | None = None) -> Path:
    """Return the g-xTB v2 xtb executable path."""
    value = gxtb_exe or os.environ.get("GXTB_EXE")
    if not value:
        raise RuntimeError("Set GXTB_EXE to the g-xTB v2 xtb executable.")
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"g-xTB executable does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"g-xTB executable is not executable: {path}")
    return path


def oet_gxtb_bin(*, tools: Path | None = None) -> Path:
    """Return the OET g-xTB wrapper executable."""
    root = tools or get_oet_tools()
    exe = root / "bin" / "oet_gxtb"
    if not exe.exists():
        raise RuntimeError(f"Expected OET g-xTB executable not found: {exe}")
    return exe


def gxtb_ext_params(
    *,
    gxtb_exe: str | None = None,
    extra_params: str | None = None,
) -> str:
    """Build Ext_Params for OET's g-xTB v2 wrapper."""
    args = ["--exe", str(get_gxtb_exe(gxtb_exe))]
    if extra_params:
        args.extend(shlex.split(extra_params))
    return shlex.join(args)


def gxtb_orca_block(
    *,
    gxtb_exe: str | None = None,
    ext_params: str | None = None,
    tools: Path | None = None,
) -> str:
    """Build the ORCA method block for OET g-xTB v2 external calculations."""
    prog = oet_gxtb_bin(tools=tools)
    params = gxtb_ext_params(gxtb_exe=gxtb_exe, extra_params=ext_params)
    return f"""
%method
ProgExt "{prog}"
Ext_Params "{params}"
end
%output
Print[P_EXT_OUT] 1
Print[P_EXT_GRAD] 1
end
""".strip()
