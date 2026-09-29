"""Small compute-node UMA SP, optimization, and NumFreq lifecycle check."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pandas as pd

from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from frust.workflows.methods import CalculatorSpec, MethodPlan


class WaterLifecycleWorkflow(BaseWorkflow):
    """Three UMA calculations on one water geometry in one target job."""

    workflow_name = "uma_water_lifecycle"

    def _build_targets(self) -> list[WorkflowTarget]:
        """Return the single water target."""
        return [WorkflowTarget("water", None)]

    def _prepare_initial_df(self, target, *, save_dir, options) -> pd.DataFrame:
        """Return the fixed three-atom water geometry."""
        return pd.DataFrame(
            {
                "substrate_name": ["water"],
                "atoms": [["O", "H", "H"]],
                "coords_embedded": [
                    [[0.0, 0.0, 0.0], [0.758602, 0.0, 0.504284], [-0.758602, 0.0, 0.504284]]
                ],
            }
        )

    def _stage_defs(self) -> list[StageDef]:
        """Run the three UMA stages in one job-local group."""
        files = ["input.inp", "orca.out", "input_EXT.uma.json"]
        return [
            StageDef("prepare", "prepare", kind="prepare"),
            StageDef("uma_sp", "UMA single point", save_files=files),
            StageDef("uma_opt", "UMA optimization", save_files=files),
            StageDef("uma_numfreq", "UMA numerical frequencies", save_files=files),
        ]


def build_workflow(root: Path) -> WaterLifecycleWorkflow:
    """Return the water workflow with one common UMA server configuration."""
    common = {
        "uma": "omol@uma-s-1p2p1",
        "uma_xtb_alpb": "chloroform",
        "uma_xtb_exe": os.environ["XTB_EXE"],
        "uma_offline": True,
        "uma_inference_settings": "batch",
        "uma_keep_logs": "always",
        "uma_log_dir": str(root / "server-logs"),
    }
    method = MethodPlan(
        "uma-water-lifecycle",
        {
            "uma_sp": CalculatorSpec("orca", {"ExtOpt": None}, kwargs=common),
            "uma_opt": CalculatorSpec("orca", {"ExtOpt": None, "Opt": None}, kwargs=common),
            "uma_numfreq": CalculatorSpec("orca", {"ExtOpt": None, "NumFreq": None}, kwargs=common),
        },
    )
    return WaterLifecycleWorkflow(method=method)


def main() -> None:
    """Run the lifecycle check directly inside an allocated compute job."""
    root = Path(os.environ["FRUST_UMA_TASK03_OUT"])
    root.mkdir(parents=True, exist_ok=True)
    runtime = os.environ["FRUST_UMA_TASK03_OET_TOOLS"]
    print(f"driver_host={socket.gethostname()} driver_pid={os.getpid()}", flush=True)
    print(f"slurm_job_id={os.environ.get('SLURM_JOB_ID')}", flush=True)
    print(f"oet_tools={runtime}", flush=True)
    result = build_workflow(root).run(
        targets=[0],
        out_dir=root / "workflow",
        execution="single_job",
        n_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")),
        mem_gb=12,
        save_output_dir=True,
        work_dir=os.environ.get("SLURM_TMPDIR"),
        target_retention="all",
        uma_oet_tools=runtime,
    )
    stage_names = ("uma_sp", "uma_opt", "uma_numfreq")
    stages = {}
    for name in stage_names:
        attrs = result.attrs["frust_steps"][name]
        stages[name] = {
            "normal_termination": bool(result.loc[0, f"{name}-NT"]),
            "energy_Eh": float(result.loc[0, f"{name}-EE"]),
            "server_pid": attrs["input"].get("uma_server_pid"),
            "server_hostname": attrs["input"].get("uma_server_hostname"),
            "server_bind": attrs["input"].get("uma_server_bind"),
        }
    summary = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "driver_hostname": socket.gethostname(),
        "driver_pid": os.getpid(),
        "frust_revision": os.environ.get("FRUST_TASK03_REVISION"),
        "oet_tools": runtime,
        "stages": stages,
    }
    (root / "lifecycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(stage["normal_termination"] for stage in stages.values()):
        raise RuntimeError("At least one UMA stage did not terminate normally")


if __name__ == "__main__":
    main()
