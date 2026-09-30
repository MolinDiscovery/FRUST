"""Build a reproducible notebook for the final gas-phase TS3 attempt."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "ts3_gas_review.ipynb"


def main() -> None:
    """Write the TS3 review notebook without executing cluster calculations."""
    cells = [
        nbf.v4.new_markdown_cell(
            """# UMA gas-phase TS3 review

The final attempt starts from the **ALPB-optimized TS3 geometry**, displaces
its two shallow peripheral imaginary modes, then runs **gas-phase UMA OptTS
and NumFreq**. The seed is not a gas-phase profile reference; only the new
gas-phase optimized coordinates can become one.

Run this notebook with the **UMA conda environment** after the submitted job
has finished. The first code cell copies its result from the cluster. The
notebook shows termination, all negative frequencies, reactive-core distances,
and the animated mode. Accept TS3 only if OptTS and NumFreq terminated, there
is **exactly one** negative frequency, and its motion follows the intended
TS3 reactive coordinate. Otherwise keep TS3 quarantined."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from IPython.display import display
import frust as ft

REMOTE = (
    "HPC5:/lustre/hpc/kemi/jmni/results/"
    "uma-task04-ts3-gas-final-20260930/workflow/ts3_gas/final.parquet"
)
LOCAL = Path(
    "/Users/jacobmolinnielsen/Developer/FrustActivationProject/fruits/"
    "results-2026/uma-task04-ts3-gas-final-20260930/final.parquet"
)
LOCAL.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["scp", "-q", REMOTE, str(LOCAL)], check=True)
result = pd.read_parquet(LOCAL)
assert len(result) == 1, f"Expected one TS3 row, got {len(result)}"
row = result.iloc[0]
print("Copied:", LOCAL)
print("State:", row["state_id"], "Source:", row["source_xyz"])
print("OptTS normal termination:", bool(row["uma_ts_opt-NT"]))
print("NumFreq normal termination:", bool(row["uma_freq-NT"]))
for stage in ("uma_ts_opt", "uma_freq"):
    error = row.get(f"{stage}-error")
    if isinstance(error, str) and error:
        print(f"{stage} error: {error}")"""
        ),
        nbf.v4.new_code_cell(
            """raw_modes = row.get("uma_freq-vibs")
modes = list(raw_modes) if isinstance(raw_modes, (list, tuple, np.ndarray)) else []
frequency_table = pd.DataFrame(
    {"mode_index": range(len(modes)),
     "frequency_cm1": [float(mode["frequency"]) for mode in modes]}
)
negative = frequency_table.loc[frequency_table.frequency_cm1 < 0]
display(frequency_table.head(12))
print("All imaginary modes:")
display(negative)
print("Exactly one imaginary mode:", len(negative) == 1)

previous = Path(
    "/Users/jacobmolinnielsen/Developer/FrustActivationProject/fruits/"
    "results-2026/uma-task04-references-20260929/ts3_gas.parquet"
)
if previous.exists():
    old = pd.read_parquet(previous).iloc[0]
    print("Original gas TS3 negatives:",
          [float(m["frequency"]) for m in old["uma_freq-vibs"]
           if float(m["frequency"]) < 0])"""
        ),
        nbf.v4.new_code_cell(
            """if bool(row["uma_ts_opt-NT"]):
    coords = np.vstack(row["uma_ts_opt-oc"]).astype(float)
    roles = {name: int(index) for name, index in row["constraint_roles"].items()}
    pairs = [
        ("transfer_H", "cat_B"),
        ("transfer_H", "pin_B"),
        ("transfer_H", "substrate_C"),
        ("cat_B", "substrate_C"),
        ("pin_B", "substrate_C"),
        ("pin_B", "cat_B"),
    ]
    def distance(xyz, first, second):
        return float(np.linalg.norm(xyz[roles[first]] - xyz[roles[second]]))

    if len(negative) == 1:
        mode_index = int(negative.iloc[0]["mode_index"])
        displacement = np.vstack(modes[mode_index]["mode"]).astype(float)
        before, after = coords - 0.25 * displacement, coords + 0.25 * displacement
    else:
        mode_index = None
        before = after = coords
    display(pd.DataFrame([
        {"pair": f"{first}–{second}",
         "optimized_A": distance(coords, first, second),
         "mode_minus_A": distance(before, first, second),
         "mode_plus_A": distance(after, first, second)}
        for first, second in pairs
    ]).round(3))
else:
    print("No optimized geometry: inspect the OptTS error above.")"""
        ),
        nbf.v4.new_code_cell(
            """if bool(row["uma_ts_opt-NT"]) and len(negative) == 1:
    viewer_path = LOCAL.parent / "ts3_gas_reactive_mode.html"
    viewer = ft.plot_vibs(
        result,
        row_index=0,
        vId=mode_index,
        custom_coords_col_name="uma_ts_opt-oc",
        export_HTML=str(viewer_path),
        width=700,
        height=500,
        reps=100,
    )
    print("Saved animated viewer:", viewer_path)
    display(viewer)
else:
    print("No unique reactive imaginary mode to animate.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Review decision

- [ ] OptTS and NumFreq both terminated normally.
- [ ] Exactly one imaginary frequency remains.
- [ ] The animated mode moves the intended TS3 B–H/B–C reactive core, rather
      than only the pinacol or catalyst substituents.
- [ ] Reactive distances and overall geometry are chemically plausible.

Record the observed mode index, frequency, and any concern here. A failed
check means the reference remains **quarantined**; do not silently replace it
with the ALPB geometry or the ωB97 profile."""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python (UMA)",
        "language": "python",
        "name": "python3",
    }
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
