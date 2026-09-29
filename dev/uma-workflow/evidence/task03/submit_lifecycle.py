"""Submit the focused UMA lifecycle workflow from a cluster login node."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from frust.cluster import ClusterConfig, Resources

from run_lifecycle import build_workflow


def main() -> None:
    """Submit one target job to node066 with client hostname auditing."""
    root = Path(os.environ["FRUST_UMA_TASK03_OUT"])
    root.mkdir(parents=True, exist_ok=True)
    runtime = Path("/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu")
    audit_root = root / "oet-audit"
    bin_dir = audit_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("oet_server", "oet_uma"):
        (bin_dir / name).symlink_to(runtime / "bin" / name)
    client = bin_dir / "oet_client"
    client.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'host=%s pid=%s ppid=%s bind=%s\\n' "
        '"$(hostname)" "$$" "$PPID" "$2" '
        '>> "$FRUST_UMA_TASK03_OUT/client_calls.log"\n'
        f'exec "{runtime / "bin" / "oet_client"}" "$@"\n'
    )
    client.chmod(0o755)
    os.environ["FRUST_UMA_TASK03_OET_TOOLS"] = str(audit_root)

    print(f"submission_host={socket.gethostname()} submission_pid={os.getpid()}", flush=True)
    print(f"runtime={runtime} audit_root={audit_root}", flush=True)
    submission = build_workflow(root).submit(
        out_dir=root / "workflow",
        cluster=ClusterConfig(
            partition="kemi1",
            log_dir=root / "submitit",
            extra_slurm_parameters={"nodelist": "node066"},
        ),
        execution="single_job",
        stage_resources={"single_job": Resources(cpus=2, mem_gb=24, timeout_min=90)},
        collect=False,
        save_output_dir=True,
        target_retention="all",
        uma_oet_tools=audit_root,
    )
    summary = {
        "submission_host": socket.gethostname(),
        "job_ids": submission.job_ids,
        "target_tags": submission.tags,
        "target_directories": submission.save_dirs,
        "oet_runtime": str(runtime),
        "oet_audit_root": str(audit_root),
    }
    (root / "submission.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
