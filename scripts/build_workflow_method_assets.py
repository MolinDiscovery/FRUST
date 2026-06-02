"""Build workflow-method documentation assets.

Run from the repository root:

    python scripts/build_workflow_method_assets.py
"""

from pathlib import Path

import pandas as pd

import frust as ft


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
TS_GUESS_HTML = ASSET_DIR / "workflow-method-ts-guesses.html"


def build_assets() -> list[Path]:
    """Generate lightweight workflow-method tutorial assets."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    components = pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": [
                "C1=CC=CO1",
                "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B",
            ],
            "compound_name": ["furan", "TMP"],
            "rpos": ["", ""],
        }
    )

    systems = ft.screen.expand(ft.screen.read(components))
    ts_guesses = ft.screen.create_ts_guesses(
        systems,
        ts_types=["TS4"],
        n_confs=1,
        n_cores=1,
    )

    scene = ft.vis.ts_guess_scene(
        ts_guesses["TS4"],
        row_indices=[0, 1],
        show_roles=True,
        show_constraint_distances=True,
        columns=2,
        cell_size=(360, 360),
        linked=False,
    )
    ft.vis.Py3DmolGridRenderer(scene).write_html(str(TS_GUESS_HTML))
    return [TS_GUESS_HTML]


if __name__ == "__main__":
    for path in build_assets():
        print(path.relative_to(ROOT))
