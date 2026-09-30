"""Refine UMA reference saddles after removing shallow unwanted modes."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd

from frust.cluster import ClusterConfig, Resources
from frust.workflows.core import StageDef

from reference_workflow import UMAReferenceWorkflow


class RetryUMAReferenceWorkflow(UMAReferenceWorkflow):
    """Reoptimize a saved TS after mode displacement and a fresh Hessian.

    Parameters
    ----------
    state : str
        Transition state identifier.
    environment : {"gas", "alpb-chloroform"}
        Potential for both new frequency calculations and optimization.
    output_root : pathlib.Path
        Directory for the job's UMA server log.
    seed : pathlib.Path
        Completed single-row reference parquet containing optimized geometry
        and numerical normal modes.
    displacements : tuple of (int, float)
        Saved mode index and signed Cartesian mode scale. For example,
        ``((1, 0.8),)`` moves away from a shallow second imaginary mode.
    """

    workflow_name = "uma_geometry_reference_retry"

    def __init__(
        self,
        *,
        state: str,
        environment: str,
        output_root: Path,
        seed: Path,
        displacements: tuple[tuple[int, float], ...],
    ) -> None:
        if not state.startswith("TS"):
            raise ValueError("Retry workflow requires a transition state")
        super().__init__(state=state, environment=environment, output_root=output_root)
        self.seed = Path(seed)
        self.displacements = displacements
        self.method = self.method.with_stage("uma_pre_freq", self.method.for_stage("uma_freq"))

    def _prepare_initial_df(self, target, *, save_dir, options) -> pd.DataFrame:
        """Seed the retry from a saved geometry and selected normal modes."""
        frame = super()._prepare_initial_df(target, save_dir=save_dir, options=options)
        seed = pd.read_parquet(self.seed)
        if len(seed) != 1 or str(seed.iloc[0]["state_id"]).upper() != self.state:
            raise ValueError(f"Seed must contain exactly one {self.state} row")
        row = seed.iloc[0]
        if list(row["atoms"]) != list(frame.iloc[0]["atoms"]):
            raise ValueError("Seed atom order does not match the source structure")
        coordinates = np.vstack(row["uma_ts_opt-oc"]).astype(float)
        for index, scale in self.displacements:
            vibration = row["uma_freq-vibs"][index]
            if float(vibration["frequency"]) >= 0:
                raise ValueError(f"Seed mode {index} is not imaginary")
            coordinates += scale * np.vstack(vibration["mode"]).astype(float)
        frame.at[0, "coords_embedded"] = coordinates.tolist()
        frame.attrs["uma_reference_seed"] = {
            "parquet": str(self.seed),
            "displacements": list(self.displacements),
        }
        return frame

    def _stage_defs(self) -> list[StageDef]:
        """Calculate a seed Hessian, follow the TS mode, then verify it."""
        files = ["input.inp", "orca.out", "input_EXT.uma.json", "input.xyz"]
        return [
            StageDef("prepare", "Load and displace prior UMA geometry", kind="prepare"),
            StageDef(
                "uma_pre_freq",
                "UMA seed numerical Hessian",
                read_files=["input.hess"],
                save_files=files,
            ),
            StageDef(
                "uma_ts_opt",
                "UMA saddle refinement",
                use_last_hess=True,
                save_files=files,
            ),
            StageDef("uma_freq", "UMA final numerical frequencies", save_files=files),
        ]


def main() -> None:
    """Submit one retry job with a single reusable compute-node UMA server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("TS1", "TS2", "TS3", "TS4"), required=True)
    parser.add_argument("--environment", choices=("gas", "alpb-chloroform"), required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--displace", nargs="*", default=[], metavar="MODE:SCALE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    displacements = tuple(
        (int(index), float(scale))
        for index, scale in (item.split(":", 1) for item in args.displace)
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workflow = RetryUMAReferenceWorkflow(
        state=args.state,
        environment=args.environment,
        output_root=output,
        seed=args.seed.resolve(),
        displacements=displacements,
    )
    runtime = Path("/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu")
    submission = workflow.submit(
        out_dir=output / "workflow",
        cluster=ClusterConfig(
            partition="kemi1",
            log_dir=output / "submitit",
            extra_slurm_parameters={"nodelist": "node066"},
        ),
        execution="single_job",
        stage_resources={"single_job": Resources(cpus=4, mem_gb=32, timeout_min=240)},
        collect=False,
        save_output_dir=True,
        target_retention="all",
        uma_oet_tools=runtime,
    )
    record = {
        "state": args.state,
        "environment": args.environment,
        "seed": str(args.seed.resolve()),
        "displacements": displacements,
        "job_ids": submission.job_ids,
        "target_directories": submission.save_dirs,
        "oet_runtime": str(runtime),
        "frust_revision": os.environ.get("FRUST_TASK04_REVISION"),
        "submission_host": socket.gethostname(),
    }
    (output / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
