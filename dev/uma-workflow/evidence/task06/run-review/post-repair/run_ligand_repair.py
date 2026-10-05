"""Reoptimize the Task 06 N-methylpyrrole ligand from a torsion-displaced seed."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import frust as ft
from frust.workflows.methods import preset


ROOT = Path(__file__).resolve().parent
SOURCE = pd.read_parquet(ROOT / "original_ligand_final.parquet")
METHOD = preset("wb97xd3-631g")


def update_status(**fields: object) -> None:
    """Write the current attempt and stage for later inspection.

    Parameters
    ----------
    **fields : object
        JSON-serializable status fields.
    """
    path = ROOT / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields)
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    print(status, flush=True)


def frequencies(frame: pd.DataFrame) -> list[float]:
    """Return the real frequencies from one ligand frequency result.

    Parameters
    ----------
    frame : pandas.DataFrame
        One-row FRUST calculation result.

    Returns
    -------
    list of float
        Vibrational frequencies in cm-1.
    """
    return [float(v["frequency"]) for v in frame.iloc[0]["dft_freq-vibs"]]


def main() -> None:
    """Try displaced seeds, then a previously validated conformer if needed."""
    os.environ.setdefault("ORCA_EXE", "/Users/jacobmolinnielsen/Library/orca_6_1_0/orca")
    source = SOURCE.iloc[0]
    mode = min(source["dft_freq-vibs"], key=lambda v: float(v["frequency"]))
    assert float(mode["frequency"]) < 0
    displacement = np.vstack(mode["mode"]).astype(float)
    displacement *= 0.30 / np.max(np.linalg.norm(displacement, axis=1))
    current_coords = np.vstack(source["dft_opt-oc"]).astype(float)

    old_path = Path(
        "/Users/jacobmolinnielsen/Developer/FrustActivationProject/fruits/"
        "results-2025/results_1m_mols_methylpyrrole_DFT/"
        "run_mols_batch_3d7abf0.parquet"
    )
    old = pd.read_parquet(old_path).iloc[0]
    assert list(old["atoms"]) == list(source["atoms"])
    old_coords = np.vstack(
        old["DFT-Opt-wB97X-D3-6-31G**-Freq-opt_coords"]
    ).astype(float)
    candidates = [
        ("mode_plus", current_coords + displacement),
        ("mode_minus", current_coords - displacement),
        ("prior_minimum", old_coords),
    ]
    template = SOURCE.drop(
        columns=[column for column in SOURCE if column.endswith("-oc")]
        + [column for column in SOURCE if column.startswith("dft_")]
    ).copy()

    for label, coords in candidates:
        attempt = ROOT / label
        attempt.mkdir(exist_ok=True)
        frame = template.copy()
        frame.at[frame.index[0], "coords_embedded"] = coords.tolist()
        frame.to_parquet(attempt / "seed.parquet", index=False)
        update_status(state="running", candidate=label, stage="dft_opt")
        stepper = ft.Stepper(
            n_cores=10,
            memory_gb=20,
            output_base=attempt / "orca_outputs",
            work_dir=str(attempt / "scratch"),
            save_output_dir=True,
            save_calc_dirs=True,
        )
        successful = True
        for stage in ("dft_opt", "dft_freq", "dft_solv_sp"):
            update_status(state="running", candidate=label, stage=stage)
            spec = METHOD.stages[stage]
            frame = stepper.orca(
                frame,
                name=stage,
                options=spec.options,
                xtra_inp_str=spec.xtra_inp_str,
                save_step=True,
            )
            frame.to_parquet(attempt / f"{stage}.parquet", index=False)
            if not bool(frame.iloc[0].get(f"{stage}-NT", False)):
                update_status(state="retrying", candidate=label, failed_stage=stage)
                successful = False
                break
            if stage == "dft_freq":
                freqs = frequencies(frame)
                update_status(
                    state="running", candidate=label, stage=stage,
                    lowest_frequency_cm1=min(freqs),
                )
                if min(freqs) < 0:
                    successful = False
                    break
        if successful:
            frame.to_parquet(ROOT / "ligand_repaired_final.parquet", index=False)
            update_status(
                state="complete", candidate=label, stage="done",
                lowest_frequency_cm1=min(frequencies(frame)),
                dft_opt_energy_hartree=float(frame.iloc[0]["dft_opt-EE"]),
                dft_solv_sp_energy_hartree=float(frame.iloc[0]["dft_solv_sp-EE"]),
            )
            return
    update_status(state="failed", stage="done")
    raise RuntimeError("No candidate converged to a minimum with all positive frequencies")


if __name__ == "__main__":
    main()
