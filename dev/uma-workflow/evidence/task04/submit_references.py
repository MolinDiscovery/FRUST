"""Submit selected UMA profile reference jobs from the HPC login node."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from frust.cluster import ClusterConfig, Resources

from reference_workflow import SOURCE_FILES, UMAReferenceWorkflow, read_reference_xyz


def main() -> None:
    """Submit separate state/environment jobs using one server per job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", nargs="+", choices=SOURCE_FILES, required=True)
    parser.add_argument(
        "--environments",
        nargs="+",
        choices=("gas", "alpb-chloroform"),
        required=True,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = Path("/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu")
    revision = os.environ.get("FRUST_TASK04_REVISION")
    submitted = []
    for state in args.states:
        atoms, _, roles = read_reference_xyz(state)
        for environment in args.environments:
            run_root = output / environment / state.lower()
            run_root.mkdir(parents=True, exist_ok=True)
            workflow = UMAReferenceWorkflow(
                state=state,
                environment=environment,
                output_root=run_root,
            )
            submission = workflow.submit(
                out_dir=run_root / "workflow",
                cluster=ClusterConfig(
                    partition="kemi1",
                    log_dir=run_root / "submitit",
                    extra_slurm_parameters={"nodelist": "node066"},
                ),
                execution="single_job",
                stage_resources={
                    "single_job": Resources(cpus=4, mem_gb=32, timeout_min=240)
                },
                collect=False,
                save_output_dir=True,
                target_retention="all",
                uma_oet_tools=runtime,
            )
            record = {
                "state": state,
                "environment": environment,
                "source_xyz": SOURCE_FILES[state],
                "atom_count": len(atoms),
                "constraint_roles": roles,
                "job_ids": submission.job_ids,
                "target_directories": submission.save_dirs,
                "oet_runtime": str(runtime),
                "frust_revision": revision,
            }
            submitted.append(record)
            (run_root / "submission.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
    summary = {
        "submission_host": socket.gethostname(),
        "frust_revision": revision,
        "jobs": submitted,
    }
    (output / "submissions.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
