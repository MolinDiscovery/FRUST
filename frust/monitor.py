from nuse import start_monitoring
from submitit import JobEnvironment
import logging

log = logging.getLogger(__name__)

def maybe_start_nuse(live: bool):
    """
    If live and in SLURM, start NUSE monitoring.
    """
    if live:
        try:
            job = JobEnvironment()
            log.info(f"SLURM job {job.job_id}, starting NUSE")
            start_monitoring(filter_cgroup=True)
        except Exception as e:
            log.warning("NUSE monitoring skipped: %s", e)