import os
import shutil
import shlex
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from frust.config import get_oet_tools

DEFAULT_UMA_MODEL = "uma-s-1p1"
LOCAL_BIND_HOST = "127.0.0.1"


@dataclass(frozen=True)
class UmaSpec:
    task: str
    model: str = DEFAULT_UMA_MODEL
    device: str = "cpu"
    cache_dir: str | None = None
    offline: bool = False


@dataclass
class UmaServerHandle:
    bind: str
    log_path: str
    _preserve_dir: Path
    _preserved_path: str | None = None

    def __iter__(self):
        yield self.bind
        yield self.log_path

    def preserve(self) -> str:
        """Copy the transient server log to the configured preserved-log directory."""
        if self._preserved_path is not None:
            return self._preserved_path
        src = Path(self.log_path)
        self._preserve_dir.mkdir(parents=True, exist_ok=True)
        dest = self._preserve_dir / src.name
        if src.exists() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        self._preserved_path = str(dest)
        return self._preserved_path


def _free_local_port() -> int:
    s = socket.socket()
    s.bind((LOCAL_BIND_HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse_uma_spec(
    uma: str,
    *,
    device: str = "cpu",
    cache_dir: str | None = None,
    offline: bool = False,
) -> UmaSpec:
    """Parse FRUST's ``task`` or ``task@model`` UMA shorthand."""
    value = uma.strip() if isinstance(uma, str) else ""
    if not value:
        raise ValueError("UMA spec must be a non-empty string")

    if "@" in value:
        task, model = value.split("@", 1)
        task = task.strip()
        model = model.strip()
    else:
        task = value
        model = DEFAULT_UMA_MODEL

    if not task:
        raise ValueError(f"UMA spec {uma!r} is missing a task before '@'")
    if not model:
        raise ValueError(f"UMA spec {uma!r} is missing a model after '@'")

    return UmaSpec(
        task=task,
        model=model,
        device=device,
        cache_dir=cache_dir,
        offline=offline,
    )


def oet_bin(name: str, *, tools: Path | None = None) -> Path:
    """Return an OET 2 executable path and validate it exists."""
    root = tools or get_oet_tools()
    exe = root / "bin" / name
    if not exe.exists():
        raise RuntimeError(f"Expected OET executable not found: {exe}")
    return exe


def uma_ext_args(spec: UmaSpec) -> list[str]:
    args = ["-t", spec.task, "-m", spec.model, "-d", spec.device]
    if spec.cache_dir:
        args.extend(["-c", spec.cache_dir])
    if spec.offline:
        args.extend(["-o", "True"])
    return args


def uma_ext_params(spec: UmaSpec, *, bind: str | None = None) -> str:
    args = []
    if bind:
        args.extend(["-b", bind])
    args.extend(uma_ext_args(spec))
    return shlex.join(args)


def uma_orca_block(
    spec: UmaSpec,
    *,
    server: bool,
    bind: str | None = None,
    tools: Path | None = None,
) -> str:
    if server:
        if not bind:
            raise ValueError("server UMA ORCA block requires a bind address")
        prog = oet_bin("oet_client", tools=tools)
        ext_params = uma_ext_params(spec, bind=bind)
    else:
        prog = oet_bin("oet_uma", tools=tools)
        ext_params = uma_ext_params(spec)

    return f"""
%method
ProgExt "{prog}"
Ext_Params "{ext_params}"
end
%output
Print[P_EXT_OUT] 1
Print[P_EXT_GRAD] 1
end
""".strip()


def _server_env(*, use_gpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "" if not use_gpu else env.get("CUDA_VISIBLE_DEVICES", "0")
    env["MPLBACKEND"] = "Agg"
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return env


def _healthz_ready(bind: str) -> bool:
    with urllib.request.urlopen(f"http://{bind}/healthz", timeout=0.5) as response:
        return response.status == 200


def _normalize_log_policy(keep_logs: bool | str) -> str:
    if keep_logs is True:
        return "always"
    if keep_logs is False:
        return "never"
    if keep_logs in {"always", "on_failure", "never"}:
        return keep_logs
    raise ValueError("uma_keep_logs must be True, False, 'always', 'on_failure', or 'never'")


@contextmanager
def uma_server(
    *,
    log_dir: str | None = None,
    keep_logs: bool | str = "on_failure",
    use_gpu: bool = False,
    server_cores: int | None = None,
    memory_per_thread_mib: int = 500,
    port: int | None = None,
):
    """Run an OET 2 UMA server bound to localhost for the current process."""
    log_policy = _normalize_log_policy(keep_logs)
    preserve_dir = Path(log_dir or "UMA-logs")
    temp_log_dir = log_dir is None and log_policy != "always"
    active_log_dir = Path(tempfile.mkdtemp(prefix="frust-uma-")) if temp_log_dir else preserve_dir

    port = port or _free_local_port()
    bind = f"{LOCAL_BIND_HOST}:{port}"
    env = _server_env(use_gpu=use_gpu)

    if server_cores is None:
        server_cores = int(env.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
    server_cores = max(1, int(server_cores))

    active_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = active_log_dir / f"oet_uma_server_{port}.log"
    logf = open(log_path, "wb")

    cmd = [
        str(oet_bin("oet_server")),
        "uma",
        "--bind",
        bind,
        "--nthreads",
        str(server_cores),
        "--memory-per-thread",
        str(int(memory_per_thread_mib)),
    ]

    header = (
        f"[launcher] bind={bind} server_cores={server_cores} "
        f"memory_per_thread_mib={memory_per_thread_mib} "
        f"slurm_job_id={env.get('SLURM_JOB_ID', '')} "
        f"slurm_job_nodelist={env.get('SLURM_JOB_NODELIST', '')} "
        f"cmd={shlex.join(cmd)}\n"
    )
    logf.write(header.encode())
    logf.flush()

    p = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env=env,
        close_fds=True,
        start_new_session=True,
    )

    ready = False
    for _ in range(120):
        if p.poll() is not None:
            break
        try:
            if _healthz_ready(bind):
                ready = True
                break
        except Exception:
            time.sleep(1)

    if not ready:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            p.terminate()
        logf.close()
        handle = UmaServerHandle(bind=bind, log_path=str(log_path), _preserve_dir=preserve_dir)
        preserved = handle.preserve()
        if temp_log_dir:
            shutil.rmtree(active_log_dir, ignore_errors=True)
        raise RuntimeError(f"OET UMA server failed to start. See log: {preserved}")

    handle = UmaServerHandle(bind=bind, log_path=str(log_path), _preserve_dir=preserve_dir)
    failed = True
    try:
        yield handle
        failed = False
    except Exception:
        if log_policy == "on_failure":
            handle.preserve()
        raise
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=10)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
        logf.flush()
        logf.close()
        if log_policy == "always":
            handle.preserve()
        elif log_policy == "never" and not temp_log_dir:
            try:
                Path(handle.log_path).unlink()
            except FileNotFoundError:
                pass
        elif log_policy == "on_failure" and not failed and handle._preserved_path is None:
            try:
                Path(handle.log_path).unlink()
            except FileNotFoundError:
                pass
        elif failed and log_policy == "on_failure":
            handle.preserve()
        if temp_log_dir:
            shutil.rmtree(active_log_dir, ignore_errors=True)
