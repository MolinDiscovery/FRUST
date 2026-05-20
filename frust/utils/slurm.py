# frust/utils/slurm.py
from __future__ import annotations
import os


def detect_job_id(user_supplied: int | None, live: bool) -> int | None:
    """Return a sensible job-id.

    1. If the user deliberately passed something, respect it.
    2. If we are inside Slurm, use the scheduler job id.
    3. If SubmitIt exposes a job environment, use that job id.
    4. Otherwise return ``None`` so local runs can be labelled as local.
    """
    if user_supplied is not None:
        return user_supplied

    if live:
        sjid = os.getenv("SLURM_JOB_ID")
        if sjid:
            return int(sjid)

        try:
            import submitit
            return int(submitit.JobEnvironment().job_id)
        except Exception:
            return None

    return None
