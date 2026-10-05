"""Submit bounded Task 05 calculations from an HPC login node."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.workflows.methods import (
    MethodPlan,
    RankingPlan,
    ScreeningPlan,
    preset,
    ranking_preset,
    screening_preset,
)

OET_RUNTIME = Path("/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu")
OET_SOURCE = Path("/lustre/hpc/kemi/jmni/software/orca-external-tools-src")
CHECKOUT = Path("/lustre/hpc/kemi/jmni/dev/FRUST")
MODEL = "omol@uma-s-1p2p1"


def revision(path: Path) -> str:
    """Return the full Git revision of a checkout.

    Parameters
    ----------
    path : pathlib.Path
        Checkout to inspect.

    Returns
    -------
    str
        Full commit hash.
    """
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def audit_runtime(root: Path) -> Path:
    """Wrap the OET client to record its compute host and loopback bind.

    Parameters
    ----------
    root : pathlib.Path
        Run directory receiving client calls and persistent server logs.

    Returns
    -------
    pathlib.Path
        Runtime wrapper for the ``uma_oet_tools`` workflow argument.
    """
    audit = root / "oet-audit"
    binary = audit / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    for name in ("oet_server", "oet_uma"):
        (binary / name).symlink_to(OET_RUNTIME / "bin" / name)
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


def audit_spec(spec, root: Path):
    """Keep the original calculator while preserving its UMA server log."""
    if "uma" not in spec.kwargs:
        return spec
    return replace(
        spec,
        kwargs={
            **spec.kwargs,
            "uma_keep_logs": "always",
            "uma_log_dir": str(root / "server-logs"),
        },
    )


def audit_screening(name: str, root: Path) -> ScreeningPlan:
    """Return a built-in screening plan with persistent UMA evidence."""
    plan = screening_preset(name)
    return ScreeningPlan(
        name=f"{name}-task05-audit",
        stages={key: audit_spec(spec, root) for key, spec in plan.stages.items()},
    )


def audit_method(name: str, root: Path) -> MethodPlan:
    """Return a full UMA plan with one compatible server log policy."""
    plan = preset(name)
    return replace(
        plan,
        name=f"{name}-task05-audit",
        stages={key: audit_spec(spec, root) for key, spec in plan.stages.items()},
    )


def audit_ranking(root: Path) -> RankingPlan:
    """Return gas UMA SP reranking with the same client audit."""
    plan = ranking_preset("uma-gas")
    return replace(plan, calculator=audit_spec(plan.calculator, root))


def input_components(root: Path) -> Path:
    """Write the one-system TS1 input used by all three checks."""
    path = root / "screen.csv"
    pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"],
            "compound_name": ["n_methyl_pyrrole", "tmp_bcat"],
            "rpos": ["2", None],
        }
    ).to_csv(path, index=False)
    return path


def submit(mode: str, base: Path) -> dict:
    """Submit one small mode and return its scheduler record.

    Parameters
    ----------
    mode : {"gas_full", "alpb_full", "hybrid_ts"}
        Full UMA gas, full UMA ALPB(chloroform), or g-xTB → UMA SP → ωB97 TS.
    base : pathlib.Path
        Parent directory for the three independent run bundles.

    Returns
    -------
    dict
        Job IDs, paths, source revisions, and selected methods.
    """
    root = base.resolve() / mode
    root.mkdir(parents=True, exist_ok=False)
    csv_path = input_components(root)
    audit = audit_runtime(root)
    cluster = ClusterConfig(
        partition="kemi1",
        log_dir=root / "submitit",
        extra_slurm_parameters={"nodelist": "node066"},
    )
    common = dict(
        csv_path=csv_path,
        ts_types=["TS1"],
        spec_profile="omol-uma-s-1p2p1/gas",
        spec_match="exact",
        prune_initial=True,
    )
    if mode in {"gas_full", "alpb_full"}:
        environment = "uma-gas" if mode == "gas_full" else "uma-alpb-chloroform"
        workflow = ft.workflows.catalyst_screen(
            **common,
            screening=audit_screening(environment, root),
            method=audit_method(environment, root),
            level="full",
            scope="barriers",
            dimer_reference="dimer",
            include_dft_rank_sp=False,
            reference_store=root / "reference-store",
            n_confs=2 if mode == "gas_full" else 1,
            top_n=2 if mode == "gas_full" else 1,
            ts_refine_n=2 if mode == "gas_full" else 1,
        )
        plan = workflow.plan()[["branch", "state_id", "target", "action"]]
        plan.to_csv(root / "plan.csv", index=False)
        result = workflow.submit(
            out_dir=root / "workflow",
            cluster=cluster,
            execution="single_job",
            stage_resources={
                "single_job": Resources(cpus=4, mem_gb=40, timeout_min=1440)
            },
            target_retention="all",
            artifact_policy="standard",
            uma_oet_tools=audit,
        )
        jobs = {
            branch: {
                "job_ids": entry.job_ids,
                "targets": entry.tags,
                "directories": entry.save_dirs,
                "collection_job_id": entry.collection_job_id,
            }
            for branch, entry in result.child_submissions.items()
        }
        finalizer = result.finalization_job_id
    elif mode == "hybrid_ts":
        composed = ft.workflows.catalyst_screen(
            **common,
            screening="gxtb-default",
            ranking=audit_ranking(root),
            method="wb97xd3-631g",
            level="full",
            include_dft_rank_sp=False,
            n_confs=2,
            top_n=2,
            uma_rank_top_n=1,
        )
        workflow = composed.children()["transition_states"]
        workflow.show_stages(execution="dft_staged").to_csv(
            root / "stage_plan.csv", index=False
        )
        result = workflow.submit(
            out_dir=root / "workflow",
            cluster=cluster,
            execution="dft_staged",
            stage_resources={
                "init": Resources(cpus=4, mem_gb=32, timeout_min=720),
                "dft_preopt": Resources(cpus=4, mem_gb=32, timeout_min=720),
                "dft_hessian": Resources(cpus=4, mem_gb=40, timeout_min=720),
                "dft_ts_opt": Resources(cpus=4, mem_gb=40, timeout_min=720),
                "dft_freq": Resources(cpus=4, mem_gb=40, timeout_min=720),
                "dft_solv_sp": Resources(cpus=4, mem_gb=32, timeout_min=720),
            },
            target_retention="all",
            artifact_policy="standard",
            uma_oet_tools=audit,
        )
        jobs = {
            "transition_states": {
                "job_ids": result.job_ids,
                "targets": result.tags,
                "directories": result.save_dirs,
                "collection_job_id": result.collection_job_id,
            }
        }
        finalizer = None
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    record = {
        "mode": mode,
        "submission_host": socket.gethostname(),
        "frust_revision": revision(CHECKOUT),
        "oet_revision": revision(OET_SOURCE),
        "oet_runtime": str(OET_RUNTIME),
        "uma_model": MODEL,
        "guess_profile": common["spec_profile"],
        "input": str(csv_path),
        "run_dir": str(root / "workflow"),
        "jobs": jobs,
        "finalization_job_id": finalizer,
    }
    (root / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    """Parse the mode and submit the corresponding bounded calculation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gas_full", "alpb_full", "hybrid_ts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.mode, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
