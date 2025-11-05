# frust/utils/slurm.py
from __future__ import annotations
import os


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
            else:
                return 1111


    # Notebook / unit test → short repeatable id
    # random.seed(42)
    # return random.randint(1000, 1999)