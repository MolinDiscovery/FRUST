"""Repair the failed ωB97 TS branch while retaining its running references."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.workflows.screening import _finalize_run


REFERENCE_COLLECTION_JOB = "65713656"
CHECKOUT = Path("/lustre/hpc/kemi/jmni/dev/FRUST")


def comparison(base: Path):
    """Rebuild the comparison workflow from its unchanged UMA parent.

    Parameters
    ----------
    base : pathlib.Path
        Task 05 repaired run root.

    Returns
    -------
    Wb97ComparisonWorkflow
        Workflow with the original selected candidate and reference policy.
    """
    parent = ft.screen.open_run(base / "gas_full" / "workflow")
    return ft.workflows.wb97_comparison(
        parent,
        candidates="selected",
        reference_store=base / "comparison" / "reference-store",
        reuse_policy="auto_valid",
    )


def submit(base: Path) -> dict:
    """Submit the repaired TS branch and a dependent finalizer.

    Parameters
    ----------
    base : pathlib.Path
        Task 05 repaired run root.

    Returns
    -------
    dict
        New scheduler IDs and the retained reference collection job.
    """
    root = base / "comparison"
    run = root / "workflow"
    branch = run / "calculations" / "transition_states"
    if (branch / "merged.parquet").exists():
        raise FileExistsError("The comparison TS branch already has a merged result")
    if (run / "run_report.json").exists():
        raise FileExistsError("The comparison run already has a finalization report")
    workflow = comparison(base)
    ts = workflow.children()["transition_states"]
    ts.seed_dir = run / "comparison" / "seeds"
    result = ts.submit(
        out_dir=branch,
        cluster=ClusterConfig(
            partition="kemi1",
            log_dir=root / "repair-submitit",
            extra_slurm_parameters={"nodelist": "node066"},
        ),
        execution="dft_staged",
        stage_resources={
            "init": Resources(cpus=4, mem_gb=32, timeout_min=720),
            "dft_hessian": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_ts_opt": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_freq": Resources(cpus=4, mem_gb=40, timeout_min=720),
            "dft_solv_sp": Resources(cpus=4, mem_gb=32, timeout_min=720),
        },
        collect=True,
        collect_output=branch / "merged.parquet",
        collect_report=branch / "collection_report.json",
        target_retention="all",
        artifact_policy="standard",
    )
    collector = str(result.collection_job_id)
    cmd = [
        "sbatch", "--parsable", "--partition=kemi1", "--nodelist=node066",
        "--cpus-per-task=2", "--mem=4G", "--time=02:00:00",
        f"--dependency=afterany:{collector}:{REFERENCE_COLLECTION_JOB}",
        f"--export=ALL,TASK05_BASE={base}",
        f"--output={root / 'repair-finalizer-%j.log'}",
        str(CHECKOUT / "dev/uma-method-variants/evidence/task05/cluster_comparison_repair.sh"),
        "finalize",
    ]
    finalizer = subprocess.check_output(cmd, text=True).strip()
    record = {
        "frust_revision": subprocess.check_output(
            ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "run_dir": str(run),
        "ts_job_ids": result.job_ids,
        "ts_collection_job_id": collector,
        "retained_reference_collection_job_id": REFERENCE_COLLECTION_JOB,
        "finalization_job_id": finalizer,
    }
    (root / "repair_submission.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def finalize(base: Path) -> dict:
    """Build the original portable comparison after both branches collect.

    Parameters
    ----------
    base : pathlib.Path
        Task 05 repaired run root.

    Returns
    -------
    dict
        Finalization report.
    """
    run = base / "comparison" / "workflow"
    return _finalize_run(
        comparison(base),
        run,
        expected_report_paths=[
            str(run / "calculations/transition_states/collection_report.json"),
            str(run / "calculations/references/collection_report.json"),
        ],
        artifact_policy="standard",
    )


def main() -> None:
    """Submit or finalize the comparison repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "finalize"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = args.output.resolve()
    result = submit(base) if args.mode == "submit" else finalize(base)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
