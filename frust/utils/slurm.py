# frust/utils/slurm.py
from __future__ import annotations
import os, random
import subprocess, shlex, os, pathlib, textwrap, typing as _t
from submitit import SlurmExecutor




def detect_job_id(user_supplied: int | None, live: bool) -> int | None:
    """Return a sensible job-id.

    1. If the user deliberately passed something → respect it.
    2. If we are *live* inside Slurm → try SubmitIt then $SLURM_JOB_ID.
    3. Else (local notebook / test run) → deterministic pseudo-random ID.
    """
    if user_supplied is not None:
        return user_supplied

    if live:                                # “production” run on cluster
        try:
            import submitit
            return int(submitit.JobEnvironment().job_id)
        except Exception:
            sjid = os.getenv("SLURM_JOB_ID")
            if sjid:
                return int(sjid)

    # Notebook / unit test → short repeatable id
    random.seed(42)
    return random.randint(1000, 1999)


class SSHSlurmExecutor(SlurmExecutor):
    """
    Wrap every Slurm CLI call (`sbatch`, `squeue`, `scancel`, `scontrol`)
    in `ssh <login_host> …` so we can submit/rpc to Slurm *from inside*
    a compute-node job where the binaries are missing or blocked.
    """
    def __init__(self, *, login_host: str, **kwargs):
        super().__init__(**kwargs)
        self.login_host = login_host

    # ------------- internal helpers ------------------------------------

    def _ssh_run(
        self,
        remote_cmd: list[str],
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Execute `remote_cmd` (a list, e.g. ["sbatch", …]) on the login_host
        via SSH and return CompletedProcess.
        """
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", self.login_host, *remote_cmd]
        kwargs = dict(
            check=check,
            text=True,
            capture_output=capture,
        )
        return subprocess.run(ssh_cmd, **kwargs)

    # -------- override just four SlurmExecutor primitives --------------

    # 1) sbatch
    def _submit_command(self, script_path: str) -> str:
        # upstream builds ["sbatch", "--parsable", …]
        cmd = self._build_sbatch_command(script_path)
        cp = self._ssh_run(cmd)
        return cp.stdout.strip()

    # 2) squeue
    def _get_job_state(self, job_id: str) -> str:
        cmd = ["squeue", "--noheader", "--format=%T", "-j", job_id]
        cp  = self._ssh_run(cmd, check=False)
        return cp.stdout.strip() if cp.returncode == 0 else "UNKNOWN"

    # 3) scancel
    def _cancel(self, job_id: str) -> None:
        self._ssh_run(["scancel", job_id], check=False)

    # 4) scontrol (show job)
    def _get_scontrol_show(self, job_id: str) -> str:
        cp = self._ssh_run(["scontrol", "show", "job", job_id], check=False)
        return cp.stdout

    # (Submitit’s base logic takes care of parsing those results.)