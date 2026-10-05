"""Bounded scheduler check; no embedding, calculator, or UMA service.

On the HPC checkout after updating it from GitHub::

    python dev/slurm-arrays/evidence/task02/smoke.py --partition kemi1 \
        --output /path/to/new/task02-run

After both collection jobs finish::

    python dev/slurm-arrays/evidence/task02/smoke.py --verify \
        --output /path/to/new/task02-run

Use ``--backend local`` to check the script locally. Local array submission
waits for execution slots and worker completion before submitting collection.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import submitit

from frust.cluster import ClusterConfig, Resources
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget


class SleepWorkflow(BaseWorkflow):
    """Internal smoke worker that exercises the actual workflow submit path."""

    workflow_name = "array_smoke"

    def __init__(self, count):
        super().__init__()
        self.count = count

    def _build_targets(self):
        return [WorkflowTarget(tag=f"target_{i}", payload={"fail": self.count > 1 and i == 1})
                for i in range(self.count)]

    def _stage_defs(self):
        return [StageDef(id="init", name="sleep", kind="prepare")]

    def _run_stage_group(self, target, stages, *, input_df, save_dir, options):
        directory = Path(save_dir)
        record = {
            "target": target.tag, "start": time.time(), "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_index": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        }
        try:
            time.sleep(4)
            if target.payload["fail"]:
                raise RuntimeError("Deliberate Task 02 worker failure")
            return pd.DataFrame({"target": [target.tag], "smoke-NT": [True]})
        finally:
            record["finish"] = time.time()
            (directory / "interval.json").write_text(json.dumps(record, indent=2) + "\n")


def _command_output(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def _verify(root):
    evidence = json.loads((root / "submitted.json").read_text())
    for name, count in (("array", 4), ("single", 1)):
        result = evidence[name]
        report = json.loads(Path(result["collection_report"]).read_text())
        intervals = [json.loads((Path(directory) / "interval.json").read_text())
                     for directory in result["save_dirs"]]
        events = [(r["start"], 1) for r in intervals] + [(r["finish"], -1) for r in intervals]
        active = maximum = 0
        for _, change in sorted(events):
            active += change
            maximum = max(maximum, active)
        assert maximum <= 2, (name, maximum)
        assert report["n_collected"] == (3 if name == "array" else 1), report
        assert report["n_missing"] == (1 if name == "array" else 0), report
        assert len(intervals) == count
        if evidence["backend"] == "slurm":
            if name == "array":
                assert len(result["array_job_ids"]) == 1, result
                assert all(r["array_job_id"] == result["array_job_ids"][0] for r in intervals)
                assert sorted(int(r["array_index"]) for r in intervals) == list(range(4))
            else:
                assert result["array_job_ids"] == [], result
                assert intervals[0]["array_job_id"] is None
            assert all(r["cpus"] == "1" for r in intervals)
        evidence[f"{name}_verified"] = {"maximum_running_workers": maximum, "report": report}
    if evidence["backend"] == "slurm":
        ids = [job_id for name in ("array", "single") for job_id in
               evidence[name]["array_job_ids"] + [str(evidence[name]["collection_job_id"])]]
        evidence["accounting"] = _command_output([
            "sacct", "-j", ",".join(ids), "--format=JobID,State,ExitCode,AllocCPUS,ReqMem", "-P",
        ])
    (root / "verified.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Verified scheduler smoke checks: {root / 'verified.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        _verify(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        evidence = {
            "backend": args.backend, "submitit_version": submitit.__version__,
            "revision": _command_output(["git", "rev-parse", "HEAD"]),
            "partition": args.partition,
        }
        if args.backend == "slurm":
            evidence["slurm_version"] = _command_output(["sbatch", "--version"])
        for name, count in (("array", 4), ("single", 1)):
            result = SleepWorkflow(count).submit(
                out_dir=args.output / name,
                cluster=ClusterConfig(backend=args.backend, partition=args.partition,
                                      log_dir=args.output / name / "logs"),
                execution="single_job", array=True, array_parallelism=2,
                stage_resources={"single_job": Resources(cpus=1, mem_gb=2, timeout_min=5)},
                collect_resources=Resources(cpus=1, mem_gb=2, timeout_min=5),
                target_retention="all",
            )
            evidence[name] = asdict(result)
            (args.output / "submitted.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(f"{name}: workers={result.job_ids}, collector={result.collection_job_id}")
