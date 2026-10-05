"""Submit the ωB97 follow-on once the gas UMA bundle is finalized."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import frust as ft
from frust.cluster import ClusterConfig, Resources


def submit(base: Path) -> dict:
    """Submit one selected UMA candidate and its ωB97 references.

    Parameters
    ----------
    base : pathlib.Path
        Parent directory containing the completed ``gas_full/workflow`` run.

    Returns
    -------
    dict
        Job IDs, output path, and exact parent candidate ID.
    """
    base = base.resolve()
    parent = ft.screen.open_run(base / "gas_full" / "workflow")
    selected = parent.candidate_barriers()
    selected = selected[selected["selected"]]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one selected TS1 UMA candidate; found {len(selected)}"
        )
    root = base / "comparison"
    root.mkdir(parents=True, exist_ok=False)
    workflow = ft.workflows.wb97_comparison(
        parent,
        candidates="selected",
        reference_store=root / "reference-store",
        reuse_policy="auto_valid",
    )
    workflow.plan()[["branch", "state_id", "target", "action"]].to_csv(
        root / "plan.csv", index=False
    )
    result = workflow.submit(
        out_dir=root / "workflow",
        cluster=ClusterConfig(
            partition="kemi1",
            log_dir=root / "submitit",
            extra_slurm_parameters={"nodelist": "node066"},
        ),
        execution="dft_staged",
        stage_resources={
            "init": Resources(cpus=4, mem_gb=32, timeout_min=720),
            "dft_preopt": Resources(cpus=4, mem_gb=32, timeout_min=720),
            "dft_opt": Resources(cpus=4, mem_gb=32, timeout_min=720),
            "dft_hessian": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_ts_opt": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_freq": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_solv_sp": Resources(cpus=4, mem_gb=32, timeout_min=720),
        },
        target_retention="all",
        artifact_policy="standard",
    )
    record = {
        "submission_host": socket.gethostname(),
        "parent_run": str(parent.path),
        "parent_uma_result_id": str(selected.iloc[0]["ts_result_id"]),
        "run_dir": str(root / "workflow"),
        "jobs": {
            branch: {
                "job_ids": entry.job_ids,
                "targets": entry.tags,
                "directories": entry.save_dirs,
                "collection_job_id": entry.collection_job_id,
            }
            for branch, entry in result.child_submissions.items()
        },
        "finalization_job_id": result.finalization_job_id,
    }
    (root / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    """Submit the follow-on or write a durable failure report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    try:
        record = submit(base)
    except Exception as exc:
        failure = {
            "status": "not_submitted",
            "reason": f"{type(exc).__name__}: {exc}",
            "parent_run": str(base / "gas_full" / "workflow"),
        }
        (base / "comparison_submission_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        raise
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
