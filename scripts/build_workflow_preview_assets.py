"""Build calculation-free workflow preview assets.

Run from the repository root:

    python scripts/build_workflow_preview_assets.py
"""

from pathlib import Path

import pandas as pd

import frust as ft


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "1m1c.csv"
ASSET_DIR = ROOT / "docs" / "assets"
MOLS_HTML = ASSET_DIR / "workflow-mols-preview.html"
TS_HTML = ASSET_DIR / "workflow-ts-preview.html"
PREVIEW_COLUMNS = ["system_name", "state_id", "state_kind", "rpos", "cid"]


def _mols_preview() -> pd.DataFrame:
    """Create and export the molecule-workflow preview dataframe."""
    method = ft.workflows.methods.preset("r2scan-3c")
    workflow = ft.workflows.mols(
        csv_path=CSV_PATH,
        select_mols=["int1", "int2"],
        method=method,
        n_confs=None,
        dft=False,
    )
    preview = workflow.preview(n_confs=1)
    ft.plot_mols(
        preview,
        columns=2,
        cell_size=(330, 300),
        linked=False,
        export_HTML=str(MOLS_HTML),
    )
    return preview


def _ts_preview() -> pd.DataFrame:
    """Create and export the TS-workflow preview dataframe."""
    method = ft.workflows.methods.preset("r2scan-3c")
    workflow = ft.workflows.screen_ts(
        csv_path=CSV_PATH,
        ts_types=["TS1"],
        method=method,
        n_confs=None,
        dft=False,
        prune_initial=False,
    )
    preview = workflow.preview(n_confs=1)
    ft.plot_mols(
        preview,
        columns=2,
        cell_size=(330, 300),
        linked=False,
        export_HTML=str(TS_HTML),
    )
    return preview


def build_assets() -> list[Path]:
    """Generate MOLS and TS preview HTML assets from ``datasets/1m1c.csv``."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    mols = _mols_preview()
    ts = _ts_preview()

    missing = [
        column
        for column in PREVIEW_COLUMNS
        if column not in mols or column not in ts
    ]
    if missing:
        raise RuntimeError("preview assets are missing canonical columns: " + ", ".join(missing))
    if set(mols["state_id"]) != {"int1", "int2"}:
        raise RuntimeError("molecule preview did not generate int1 and int2")
    if set(ts["state_id"]) != {"TS1"}:
        raise RuntimeError("TS preview did not generate TS1")
    return [MOLS_HTML, TS_HTML]


if __name__ == "__main__":
    for path in build_assets():
        print(path.relative_to(ROOT))
