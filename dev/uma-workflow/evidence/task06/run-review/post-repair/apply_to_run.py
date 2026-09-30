"""Apply the checked Mac ligand result to the Task 06 portable run."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from frust.screen.runs import build_analysis


LOCAL = Path(__file__).resolve().parent
RUN = Path(
    "/Users/Mounts/HPC3/results/uma-task06-smoke-20260930/"
    "alpb_full/workflow"
)
REFS = RUN / "calculations/references"
LEAF = REFS / "ligand__n_methyl_pyrrole"
STAGES = ("dft_opt", "dft_freq", "dft_solv_sp")
REPAIR_ID = "ligand_methyl_torsion_mac_20260930"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Save one dataframe through a same-directory temporary path.

    Parameters
    ----------
    frame : pandas.DataFrame
        Updated run result.
    path : pathlib.Path
        Destination parquet file.
    """
    temp = path.with_name(f".{path.name}.{REPAIR_ID}.tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def atomic_json(data: dict, path: Path) -> None:
    """Save one JSON document through a same-directory temporary path.

    Parameters
    ----------
    data : dict
        JSON-compatible document.
    path : pathlib.Path
        Destination JSON file.
    """
    temp = path.with_name(f".{path.name}.{REPAIR_ID}.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    """Back up, merge repaired DFT fields, and rebuild run analysis."""
    status = json.loads((LOCAL / "status.json").read_text())
    if status.get("state") != "complete":
        raise ValueError("Local ligand calculation is not complete")
    repaired = pd.read_parquet(LOCAL / "ligand_repaired_final.parquet")
    if len(repaired) != 1:
        raise ValueError("Expected exactly one repaired ligand row")
    row = repaired.iloc[0]
    if any(not bool(row[f"{stage}-NT"]) for stage in STAGES):
        raise ValueError("A repaired DFT stage did not terminate normally")
    lowest = min(float(v["frequency"]) for v in row["dft_freq-vibs"])
    if lowest <= 0:
        raise ValueError(f"Repaired ligand is not a minimum: {lowest}")

    affected = [
        LEAF / "init.dft_opt.parquet",
        LEAF / "init.dft_opt.dft_freq.parquet",
        LEAF / "init.dft_opt.dft_freq.dft_solv_sp.parquet",
        REFS / "computed.parquet",
        REFS / "merged.parquet",
        RUN / "analysis/states.parquet",
        RUN / "analysis/dimer_references.parquet",
        RUN / "analysis/barriers.parquet",
        RUN / "analysis/states_by_level.parquet",
        RUN / "analysis/dimer_references_by_level.parquet",
        RUN / "analysis/barriers_by_level.parquet",
        RUN / "analysis/report.json",
        RUN / "run_report.json",
    ]
    backup = RUN / "repair_backups" / REPAIR_ID
    if backup.exists():
        raise FileExistsError(f"Repair backup already exists: {backup}")
    for path in affected:
        destination = backup / path.relative_to(RUN)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    timestamp = datetime.now(timezone.utc).isoformat()
    repair_meta = {
        "id": REPAIR_ID,
        "applied_at": timestamp,
        "reason": "Remove a -65.61 cm-1 methyl torsion in the free ligand reference",
        "source": str(LOCAL / "ligand_repaired_final.parquet"),
        "seed": "Original DFT geometry plus 0.30 Angstrom along the imaginary mode",
        "compute_host": "local Mac",
        "n_cores": 10,
        "lowest_frequency_cm1": lowest,
        "backup": str(backup),
    }
    stage_frames = {
        stage: pd.read_parquet(LOCAL / "mode_plus" / f"{stage}.parquet")
        for stage in STAGES
    }
    paths = affected[:5]
    for path in paths:
        frame = pd.read_parquet(path)
        mask = frame["state_id"].astype(str).eq("ligand")
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected exactly one ligand row in {path}")
        idx = frame.index[mask][0]
        if path.name == "init.dft_opt.parquet":
            stages = STAGES[:1]
        elif path.name == "init.dft_opt.dft_freq.parquet":
            stages = STAGES[:2]
        else:
            stages = STAGES
        for stage in stages:
            for column in (name for name in repaired if name.startswith(f"{stage}-")):
                if column in frame:
                    frame.at[idx, column] = row[column]
        attrs = dict(frame.attrs)
        attrs["frust_run_repair"] = repair_meta
        if path.parent == LEAF:
            steps = dict(attrs.get("frust_steps", {}))
            for stage in stages:
                steps[stage] = stage_frames[stage].attrs["frust_steps"][stage]
            attrs["frust_steps"] = steps
        frame.attrs = attrs
        atomic_parquet(frame, path)

    analysis = build_analysis(RUN)
    states = pd.read_parquet(RUN / "analysis/states.parquet")
    ligand = states.loc[states["state_id"].eq("ligand")]
    if len(ligand) != 1 or ligand.iloc[0]["quality_status"] != "ready":
        raise RuntimeError("Rebuilt analysis did not classify ligand as ready")
    run_report_path = RUN / "run_report.json"
    run_report = json.loads(run_report_path.read_text())
    run_report["analysis"] = analysis
    run_report.setdefault("post_run_repairs", []).append(
        {
            "id": REPAIR_ID,
            "applied_at": timestamp,
            "report": "repair_reports/" + REPAIR_ID + ".json",
        }
    )
    atomic_json(run_report, run_report_path)
    report_path = RUN / "repair_reports" / f"{REPAIR_ID}.json"
    report_path.parent.mkdir(exist_ok=True)
    atomic_json(
        {
            "schema_version": 1,
            **repair_meta,
            "old_dft_opt_energy_hartree": float(
                pd.read_parquet(backup / LEAF.relative_to(RUN) / "init.dft_opt.parquet")
                .iloc[0]["dft_opt-EE"]
            ),
            "new_dft_opt_energy_hartree": float(row["dft_opt-EE"]),
            "new_smd_single_point_energy_hartree": float(row["dft_solv_sp-EE"]),
            "state_quality": analysis["state_quality"],
            "barrier_quality": analysis["barrier_quality"],
            "publication_status": "Run-local repair only; shared reference library not modified",
        },
        report_path,
    )
    print(json.dumps({"backup": str(backup), "analysis": analysis}, indent=2))


if __name__ == "__main__":
    main()
