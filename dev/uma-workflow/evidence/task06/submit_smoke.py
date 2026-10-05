"""Submit the bounded Task 06 UMA catalyst-screen checks from the login node."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.workflows.methods import ScreeningPlan, screening_preset

OET_RUNTIME = Path("/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu")
OET_SOURCE = Path("/lustre/hpc/kemi/jmni/software/orca-external-tools-src")


def revision(path: Path) -> str:
    """Return the Git revision of a local checkout.

    Parameters
    ----------
    path : pathlib.Path
        Repository checkout.

    Returns
    -------
    str
        Full commit hash.
    """
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def audit_runtime(root: Path) -> Path:
    """Create an OET client wrapper that records the client host and bind.

    Parameters
    ----------
    root : pathlib.Path
        Run output directory.

    Returns
    -------
    pathlib.Path
        OET runtime wrapper directory for ``uma_oet_tools``.
    """
    audit = root / "oet-audit"
    binary = audit / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    for name in ("oet_server", "oet_uma"):
        link = binary / name
        if not link.exists():
            link.symlink_to(OET_RUNTIME / "bin" / name)
    client = binary / "oet_client"
    client.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'host=%s pid=%s ppid=%s bind=%s\\n' "
        '"$(hostname)" "$$" "$PPID" "$3" '
        f'>> "{root / "client_calls.log"}"\n'
        f'exec "{OET_RUNTIME / "bin" / "oet_client"}" "$@"\n'
    )
    client.chmod(0o755)
    return audit


def screening_plan(name: str, root: Path) -> ScreeningPlan:
    """Enable persistent UMA server logs on one built-in screening plan.

    Parameters
    ----------
    name : str
        Built-in UMA screening preset.
    root : pathlib.Path
        Run output directory.

    Returns
    -------
    ScreeningPlan
        Equivalent potential with task-specific log retention.
    """
    base = screening_preset(name)
    stages = {}
    for stage, spec in base.stages.items():
        if stage.startswith("uma_"):
            spec = replace(
                spec,
                kwargs={
                    **spec.kwargs,
                    "uma_keep_logs": "always",
                    "uma_log_dir": str(root / "server-logs"),
                },
            )
        stages[stage] = spec
    return ScreeningPlan(name=f"{name}-task06-audit", stages=stages)


def main() -> None:
    """Submit one ALPB full screen or one gas TS screening target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("alpb_full", "gas_low_cost"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve() / args.mode
    root.mkdir(parents=True, exist_ok=False)

    components = pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"],
            "compound_name": ["n_methyl_pyrrole", "tmp_bcat"],
            "rpos": ["2", None],
        }
    )
    input_path = root / "screen.csv"
    components.to_csv(input_path, index=False)
    audit = audit_runtime(root)
    cluster = ClusterConfig(
        partition="kemi1",
        log_dir=root / "submitit",
        extra_slurm_parameters={"nodelist": "node066"},
    )
    common = {
        "csv_path": input_path,
        "ts_types": ["TS1"],
        "method": "wb97xd3-631g",
        "spec_profile": "wb97xd3-631g/gas",
        "spec_match": "exact",
        "n_confs": 2,
        "top_n": 1,
        "prune_initial": True,
    }
    if args.mode == "alpb_full":
        workflow = ft.workflows.catalyst_screen(
            **common,
            screening=screening_plan("uma-alpb-chloroform", root),
            level="full",
            scope="barriers",
            dimer_reference="dimer",
            include_dft_rank_sp=False,
        )
        resources = {
            "init": Resources(cpus=4, mem_gb=32, timeout_min=720),
            "dft_opt": Resources(cpus=4, mem_gb=32, timeout_min=1440),
            "dft_hessian": Resources(cpus=4, mem_gb=48, timeout_min=1440),
            "dft_ts_opt": Resources(cpus=4, mem_gb=32, timeout_min=1440),
            "dft_freq": Resources(cpus=4, mem_gb=48, timeout_min=1440),
            "dft_solv_sp": Resources(cpus=4, mem_gb=32, timeout_min=1440),
        }
        plan = workflow.plan()[["branch", "state_id", "target", "action"]]
        plan.to_csv(root / "plan.csv", index=False)
        submission = workflow.submit(
            out_dir=root / "workflow",
            cluster=cluster,
            execution="dft_staged",
            stage_resources=resources,
            target_retention="all",
            uma_oet_tools=audit,
        )
        jobs = {
            branch: {
                "job_ids": item.job_ids,
                "targets": item.tags,
                "directories": item.save_dirs,
                "collection_job_id": item.collection_job_id,
            }
            for branch, item in submission.child_submissions.items()
        }
        finalization_job_id = submission.finalization_job_id
    else:
        workflow = ft.workflows.screen_ts(
            **common,
            screening=screening_plan("uma-gas", root),
            calculation_level="low_cost",
        )
        submission = workflow.submit(
            out_dir=root / "workflow",
            cluster=cluster,
            execution="single_job",
            stage_resources={
                "single_job": Resources(cpus=4, mem_gb=32, timeout_min=720)
            },
            target_retention="all",
            uma_oet_tools=audit,
        )
        jobs = {
            "transition_states": {
                "job_ids": submission.job_ids,
                "targets": submission.tags,
                "directories": submission.save_dirs,
                "collection_job_id": submission.collection_job_id,
            }
        }
        finalization_job_id = None

    record = {
        "mode": args.mode,
        "submission_host": socket.gethostname(),
        "frust_revision": revision(Path(ft.__file__).resolve().parents[1]),
        "oet_revision": revision(OET_SOURCE),
        "oet_runtime": str(OET_RUNTIME),
        "input": str(input_path),
        "guess_profile": common["spec_profile"],
        "include_dft_rank_sp": False,
        "jobs": jobs,
        "finalization_job_id": finalization_job_id,
    }
    (root / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
