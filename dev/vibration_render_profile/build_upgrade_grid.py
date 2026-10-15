"""Rebuild saved vibration data through the installed FRUST/py3Dmol renderer."""

import argparse
import json
from pathlib import Path
import re

import pandas as pd
import py3Dmol

import frust as ft


def main():
    """Read saved model inputs without executing or writing the notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    output = next(
        output for cell in notebook["cells"] for output in cell.get("outputs", [])
        if "application/3dmoljs_load.v0" in output.get("data", {})
    )
    html = output["data"]["text/html"]
    if isinstance(html, list):
        html = "".join(html)
    decoder = json.JSONDecoder()

    def arguments(method):
        return [decoder.raw_decode(html[match.end():])[0]
                for match in re.finditer(r"viewergrid_\d+\[\d+\]\[\d+\]\." + method + r"\(", html)]

    blocks = arguments("addModel")
    properties = arguments("mapAtomProperties")
    legends = arguments("addLabel")
    assert blocks and len(blocks) == len(properties) == len(legends)
    grid = re.search(r"rows:\s*(\d+),\s*cols:\s*(\d+)", html)
    size = re.search(r"width:\s*(\d+)px;\s*height:\s*(\d+)px", html)
    vibration = re.search(r"\.vibrate\((\d+),([^,]+),true\)", html)
    if not (grid and size and vibration):
        raise ValueError("Expected a py3Dmol vibration grid with pixel dimensions.")
    grid_rows, columns = map(int, grid.groups())
    width, height = map(int, size.groups())
    num_frames, amplitude = int(vibration[1]), float(vibration[2])
    animation = arguments("animate")[0]
    rows = []
    for block, props in zip(blocks, properties):
        lines = block.splitlines()
        records = [line.split() for line in lines[2:2 + int(lines[0])]]
        rows.append({
            "atoms": [record[0] for record in records],
            "uma_ts_opt-oc": [[float(value) for value in record[1:4]] for record in records],
            "uma_freq-vibs": [{
                "frequency": -1.0,
                "mode": [[atom["props"][key] for key in ("dx", "dy", "dz")] for atom in props],
            }],
        })
    viewer = ft.plot_vibs(
        pd.DataFrame(rows), columns=columns,
        cell_size=(width // columns, height // grid_rows),
        custom_coords_col_name="uma_ts_opt-oc", amplitude=amplitude,
        numFrames=num_frames, fps=1000 / animation.get("interval", 50),
        linked=False, legends=legends, reps=animation.get("reps", 0),
    )
    viewer.write_html(str(args.output))
    print(json.dumps({"py3Dmol_version": py3Dmol.__version__, "py3Dmol_path": py3Dmol.__file__,
                      "rows": len(rows), "html": str(args.output)}))


if __name__ == "__main__":
    main()
