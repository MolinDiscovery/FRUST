import os, socket, subprocess, time, textwrap
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from pandas import Series

from frust.config import UMA_TOOLS as TOOLS

def _free_local_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close(); return port

@contextmanager
def _uma_server(task: str, log_dir: str = ".", use_gpu: bool = False):
    port = _free_local_port()
    env = os.environ.copy()
    n = str(env.get("SLURM_CPUS_PER_TASK", "12"))
    env["OMP_NUM_THREADS"] = n
    env["VECLIB_MAXIMUM_THREADS"] = n
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    env.pop("MPLBACKEND", None)
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = "" if not use_gpu else env.get("CUDA_VISIBLE_DEVICES", "0")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"umaserver_{port}.log"
    logf = open(log_path, "wb")
    cmd = [f"{TOOLS}/umaserver.sh", "-m", task, "-b", f"127.0.0.1:{port}", "-n", "1"]
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, close_fds=True)
    ready = False
    for _ in range(120):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                ready = True; break
        except OSError:
            time.sleep(1)
    if not ready:
        p.terminate(); logf.close()
        raise RuntimeError(f"UMA server failed to start. See log: {log_path}")
    try:
        yield port, str(log_path)
    finally:
        try:
            p.terminate(); p.wait(timeout=10)
        except Exception:
            p.kill()
        logf.flush(); logf.close()