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
def _uma_server(
    task: str,
    log_dir: str = ".",
    use_gpu: bool = False,
    workers: int | None = None,         # total UMA workers (processes/threads in the server)
    threads_per_worker: int | None = None,  # BLAS/torch threads per UMA worker
    port: int | None = None,
):
    port = port or _free_local_port()
    env = os.environ.copy()

    # Total logical CPUs available (prefer SLURM allocation if present)
    total = int(env.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))

    # # Sensible defaults: a few light workers
    # if threads_per_worker is None:
    #     threads_per_worker = 2 if total >= 4 else 1
    # if workers is None:
    #     workers = max(1, min(8, total // max(1, threads_per_worker)))

    threads_per_worker = 1
    workers = total - 30 # leave some headroom.
    
    print(f"[DEBUG]: total {total}")
    print(f"[DEBUG]: Threads per worker {threads_per_worker}")
    print(f"[DEBUG]: Workers {workers}")

    # Threading hygiene (avoid oversubscription inside each UMA worker)
    t = str(threads_per_worker)
    env["OMP_NUM_THREADS"] = t
    env["OPENBLAS_NUM_THREADS"] = t
    env["MKL_NUM_THREADS"] = t
    env["NUMEXPR_NUM_THREADS"] = t
    env["VECLIB_MAXIMUM_THREADS"] = t
    # Force CPU unless you explicitly opt into a GPU
    env["CUDA_VISIBLE_DEVICES"] = "" if not use_gpu else env.get("CUDA_VISIBLE_DEVICES", "0")
    # Headless matplotlib (prevents crashes on clusters / macOS)
    env["MPLBACKEND"] = "Agg"
    # Make sure the HF cache is consistent across nodes
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"umaserver_{port}.log"
    logf = open(log_path, "wb")

    cmd = [
        f"{TOOLS}/umaserver.sh",
        "-m", task,
        "-b", f"127.0.0.1:{port}",
        "-n", str(workers),             # << spawn multiple UMA workers
    ]
    # Optional: write a little header into the log
    header = f"[launcher] port={port} workers={workers} threads/worker={threads_per_worker} total={total}\n"
    logf.write(header.encode()); logf.flush()

    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, close_fds=True)

    # Wait for server to accept connections
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

# @contextmanager
# def _uma_server(task: str, log_dir: str = ".", use_gpu: bool = False):
#     port = _free_local_port()
#     env = os.environ.copy()
#     n = str(env.get("SLURM_CPUS_PER_TASK", "12"))
#     env["OMP_NUM_THREADS"] = n
#     env["VECLIB_MAXIMUM_THREADS"] = n
#     env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
#     env.pop("MPLBACKEND", None)
#     env["MPLBACKEND"] = "Agg"
#     env["CUDA_VISIBLE_DEVICES"] = "" if not use_gpu else env.get("CUDA_VISIBLE_DEVICES", "0")
#     Path(log_dir).mkdir(parents=True, exist_ok=True)
#     log_path = Path(log_dir) / f"umaserver_{port}.log"
#     logf = open(log_path, "wb")
#     cmd = [f"{TOOLS}/umaserver.sh", "-m", task, "-b", f"127.0.0.1:{port}", "-n", "1"]
#     p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, close_fds=True)
#     ready = False
#     for _ in range(120):
#         try:
#             with socket.create_connection(("127.0.0.1", port), timeout=0.5):
#                 ready = True; break
#         except OSError:
#             time.sleep(1)
#     if not ready:
#         p.terminate(); logf.close()
#         raise RuntimeError(f"UMA server failed to start. See log: {log_path}")
#     try:
#         yield port, str(log_path)
#     finally:
#         try:
#             p.terminate(); p.wait(timeout=10)
#         except Exception:
#             p.kill()
#         logf.flush(); logf.close()