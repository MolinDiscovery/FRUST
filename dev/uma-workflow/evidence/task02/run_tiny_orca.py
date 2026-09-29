"""One local FRUST → ORCA → OET UMA/ALPB water optimization smoke check."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

import frust as ft

ROOT = Path(__file__).resolve().parent
os.environ["ORCA_EXE"] = "/Users/jacobmolinnielsen/Library/orca_6_1_0/orca"
os.environ["OET_TOOLS"] = "/Users/jacobmolinnielsen/Library/orca-external-tools"
os.environ["XTB_EXE"] = "/opt/homebrew/Caskroom/miniconda/base/envs/UMA/bin/xtb"

water = pd.DataFrame(
    [
        {
            "substrate_name": "water",
            "custom_name": "water",
            "atoms": ["O", "H", "H"],
            "coords_embedded": [
                [0.0, 0.0, 0.0],
                [0.758602, 0.0, 0.504284],
                [-0.758602, 0.0, 0.504284],
            ],
        }
    ]
)

step = ft.Stepper(
    output_base=ROOT / "orca-opt",
    n_cores=1,
    memory_gb=2,
    save_calc_dirs=True,
    work_dir=str(ROOT / "orca-scratch"),
)
result = step.orca(
    water,
    name="uma_alpb_opt",
    options={"ExtOpt": None, "Opt": None},
    uma="omol@uma-s-1p2p1",
    uma_server=True,
    uma_inference_settings="batch",
    uma_xtb_alpb="chloroform",
    uma_xtb_exe=os.environ["XTB_EXE"],
    uma_offline=True,
    uma_keep_logs="always",
    uma_log_dir=str(ROOT / "orca-opt" / "server-logs"),
    save_step=True,
)
result.to_pickle(ROOT / "orca-opt-result.pkl")
stage = result.attrs["frust_steps"]["uma_alpb_opt"]
summary = {
    "normal_termination": bool(result.loc[0, "uma_alpb_opt-NT"]),
    "energy_Eh": float(result.loc[0, "uma_alpb_opt-EE"]),
    "uma_spec": stage["uma"],
    "uma_xtb_alpb": stage["uma_xtb_alpb"],
    "input": stage["input"],
    "calculator": stage["calculator"],
}
(ROOT / "orca-smoke-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
saved_dirs = list((ROOT / "orca-opt").glob("FRUST_results-*/uma_alpb_opt/water_0"))
if not saved_dirs:
    raise RuntimeError("ORCA calculation directory was not saved")
saved_dir = max(saved_dirs, key=lambda path: path.stat().st_mtime)
smoke = ROOT / "orca-smoke"
smoke.mkdir(exist_ok=True)
for filename in ("input.inp", "orca.out", "input_EXT.uma.json", "input.xyz"):
    shutil.copy2(saved_dir / filename, smoke / filename)
input_copy = smoke / "input.inp"
input_copy.write_text(
    "\n".join(line.rstrip() for line in input_copy.read_text().splitlines()).rstrip() + "\n",
    encoding="utf-8",
)
markers = (
    "Ext_Params",
    "GEOMETRY OPTIMIZATION CYCLE",
    "THE OPTIMIZATION HAS CONVERGED",
    "FINAL SINGLE POINT ENERGY",
    "ORCA TERMINATED NORMALLY",
)
excerpt = [
    line.rstrip()
    for line in (smoke / "orca.out").read_text(errors="replace").splitlines()
    if any(marker in line for marker in markers)
]
(smoke / "orca_excerpt.txt").write_text("\n".join(excerpt) + "\n", encoding="utf-8")
print(f"normal_termination={summary['normal_termination']}")
print(f"energy_Eh={summary['energy_Eh']:.12f}")
print(f"saved_input={smoke / 'input.inp'}")
