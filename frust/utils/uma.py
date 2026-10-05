import os
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from frust.config import get_oet_tools

DEFAULT_UMA_MODEL = "uma-s-1p1"
LOCAL_BIND_HOST = "127.0.0.1"


@dataclass(frozen=True)
class UmaSpec:
    """OET UMA settings passed through an ORCA external-method input.

    Attributes
    ----------
    task : str
        FairChem domain task, normally ``"omol"`` for molecules.
    model : str
        UMA checkpoint name such as ``"uma-s-1p2p1"``.
    device : str
        Inference device, ``"cpu"`` or ``"cuda"``.
    cache_dir : str or None
        Optional checkpoint cache directory.
    offline : bool
        Whether OET must use locally cached model files.
    xtb_alpb : str or None
        ``"chloroform"`` adds GFN2-xTB ALPB minus gas energies and gradients;
        ``None`` keeps gas-phase UMA.
    xtb_exe : str or None
        Optional normal xTB executable for the solvent correction.
    inference_settings : str or None
        FairChem inference mode. ``None`` uses OET's ``"batch"`` default.
    """

    task: str
    model: str = DEFAULT_UMA_MODEL
    device: str = "cpu"
    cache_dir: str | None = None
    offline: bool = False
    xtb_alpb: str | None = None
    xtb_exe: str | None = None
    inference_settings: str | None = None


@dataclass
class UmaServerHandle:
    bind: str
    log_path: str
    _preserve_dir: Path
    pid: int | None = None
    hostname: str | None = None
    _preserved_path: str | None = None

    def __iter__(self):
        yield self.bind
        yield self.log_path

    def preserve(self) -> str:
        """Copy the transient server log to the configured preserved-log directory."""
        src = Path(self.log_path)
        self._preserve_dir.mkdir(parents=True, exist_ok=True)
        dest = Path(self._preserved_path) if self._preserved_path else self._preserve_dir / src.name
        if src.exists() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        self._preserved_path = str(dest)
        return self._preserved_path


class UmaJobServerScope:
    """Own one lazily started UMA server for a workflow stage group.

    The scope is created inside the executing job. Separate submitted stage
    groups therefore have separate server lifetimes, while all UMA calls in
    one group share the same process and loopback bind address.
    """

    def __init__(self, *, oet_tools: str | None = None) -> None:
        self._owner_pid = os.getpid()
        self._oet_tools = oet_tools
        self._stack = ExitStack()
        self._handle: UmaServerHandle | None = None
        self._settings: tuple[object, ...] | None = None
        self._broken = False
        self._startup_elapsed_s = 0.0

    def acquire(
        self,
        *,
        log_dir: str | None,
        keep_logs: bool | str,
        use_gpu: bool,
        server_cores: int,
        memory_per_thread_mib: int,
    ) -> UmaServerHandle:
        """Return the job's server, starting it on the first UMA call.

        Parameters
        ----------
        log_dir : str or None
            Directory for retained server logs.
        keep_logs : bool or {"always", "on_failure", "never"}
            Log retention policy.
        use_gpu : bool
            Whether the server may use its allocated GPU.
        server_cores : int
            Server thread budget.
        memory_per_thread_mib : int
            Server memory budget per thread in MiB.

        Returns
        -------
        UmaServerHandle
            The single live server owned by this stage group.
        """
        if os.getpid() != self._owner_pid:
            raise RuntimeError("UMA job server scope cannot be reused after a process fork")
        if self._broken:
            raise RuntimeError("The job's UMA server is unavailable")
        if self._oet_tools is not None:
            os.environ["OET_TOOLS"] = self._oet_tools
        settings = (
            log_dir,
            _normalize_log_policy(keep_logs),
            bool(use_gpu),
            int(server_cores),
            int(memory_per_thread_mib),
            str(get_oet_tools()),
        )
        if self._handle is None:
            started = time.monotonic()
            try:
                self._handle = self._stack.enter_context(uma_server(
                    log_dir=log_dir,
                    keep_logs=keep_logs,
                    use_gpu=use_gpu,
                    server_cores=server_cores,
                    memory_per_thread_mib=memory_per_thread_mib,
                ))
            except BaseException:
                self._broken = True
                raise
            finally:
                self._startup_elapsed_s = time.monotonic() - started
            self._settings = settings
        elif settings != self._settings:
            raise ValueError(
                "UMA stages in one job must use the same server resources, "
                "log policy, and OET_TOOLS runtime"
            )
        else:
            self.ensure_healthy()
        return self._handle

    def ensure_healthy(self) -> None:
        """Check an acquired server before continuing with another target.

        Raises
        ------
        RuntimeError
            If startup failed, the server is unavailable, or the scope belongs
            to another process. An unused lazy scope requires no server check.
        """
        if os.getpid() != self._owner_pid:
            raise RuntimeError("UMA job server scope cannot be reused after a process fork")
        if self._broken:
            raise RuntimeError("The job's UMA server is unavailable")
        if self._handle is not None:
            try:
                if not _healthz_ready(self._handle.bind):
                    raise RuntimeError("UMA health check failed")
            except Exception as exc:
                self._broken = True
                raise RuntimeError("The job's UMA server stopped before the next stage") from exc

    def close(self, exc_info=(None, None, None)) -> None:
        """Stop the job's server, preserving failure logs when appropriate."""
        self._stack.__exit__(*exc_info)


_UMA_JOB_SCOPE: ContextVar[UmaJobServerScope | None] = ContextVar(
    "frust_uma_job_scope", default=None
)


def current_uma_job_scope() -> UmaJobServerScope | None:
    """Return the active workflow UMA server scope, if one exists."""
    scope = _UMA_JOB_SCOPE.get()
    if scope is not None and scope._owner_pid != os.getpid():
        raise RuntimeError("UMA job server scope cannot be inherited by a child process")
    return scope


@contextmanager
def uma_job_server_scope(*, oet_tools: str | None = None, reuse: bool = False):
    """Keep one UMA server available throughout a workflow stage group.

    Parameters
    ----------
    oet_tools : str or None, optional
        Explicit OET runtime for this executing job. If omitted, use the
        configured ``OET_TOOLS`` environment variable.
    reuse : bool, optional
        Reuse an active scope with a compatible runtime. The outer owner alone
        closes the server. Defaults to False, which rejects nested ownership.

    Yields
    ------
    UmaJobServerScope
        A lazily started server owner shared by UMA calls in the group.
    """
    active = current_uma_job_scope()
    if active is not None:
        if not reuse:
            raise RuntimeError("A UMA job server scope is already active")
        if oet_tools is not None:
            runtime = active._oet_tools or str(get_oet_tools())
            if Path(oet_tools).resolve() != Path(runtime).resolve():
                raise ValueError("Cannot reuse a UMA scope with a different OET runtime")
        active.ensure_healthy()
        yield active
        return
    previous_oet = os.environ.get("OET_TOOLS")
    if oet_tools is not None:
        os.environ["OET_TOOLS"] = str(oet_tools)
    scope = UmaJobServerScope(oet_tools=None if oet_tools is None else str(oet_tools))
    token = _UMA_JOB_SCOPE.set(scope)
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def stop_on_sigterm(signum, frame):
            if callable(previous_sigterm):
                previous_sigterm(signum, frame)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        try:
            yield scope
        except BaseException:
            scope.close(sys.exc_info())
            raise
        else:
            scope.close()
    finally:
        _UMA_JOB_SCOPE.reset(token)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if oet_tools is not None:
            if previous_oet is None:
                os.environ.pop("OET_TOOLS", None)
            else:
                os.environ["OET_TOOLS"] = previous_oet


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
    xtb_alpb: str | None = None,
    xtb_exe: str | None = None,
    inference_settings: str | None = None,
) -> UmaSpec:
    """Parse FRUST's ``task`` or ``task@model`` UMA shorthand.

    Parameters
    ----------
    uma : str
        Domain task with an optional checkpoint, for example
        ``"omol@uma-s-1p2p1"``.
    device : str, default ``"cpu"``
        Inference device.
    cache_dir : str or None, optional
        Checkpoint cache directory.
    offline : bool, default ``False``
        Request cached model files only.
    xtb_alpb : str or None, optional
        ``"chloroform"`` adds the xTB ALPB correction; ``None`` keeps gas
        phase UMA.
    xtb_exe : str or None, optional
        xTB executable for the correction. Requires ``xtb_alpb``.
    inference_settings : str or None, optional
        ``"batch"`` avoids model compilation, while ``"default"`` and
        ``"turbo"`` use it. ``None`` uses OET's default.

    Returns
    -------
    UmaSpec
        Validated UMA and optional solvent settings.
    """
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
    if xtb_alpb not in {None, "chloroform"}:
        raise ValueError("uma_xtb_alpb currently supports only 'chloroform'")
    if xtb_exe and not xtb_alpb:
        raise ValueError("uma_xtb_exe requires uma_xtb_alpb")
    if inference_settings not in {None, "default", "batch", "turbo"}:
        raise ValueError("uma_inference_settings must be 'default', 'batch', or 'turbo'")

    return UmaSpec(
        task=task,
        model=model,
        device=device,
        cache_dir=cache_dir,
        offline=offline,
        xtb_alpb=xtb_alpb,
        xtb_exe=xtb_exe,
        inference_settings=inference_settings,
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
    if spec.xtb_alpb:
        args.extend(["--xtb-alpb", spec.xtb_alpb])
    if spec.xtb_exe:
        args.extend(["--xtb-exe", spec.xtb_exe])
    if spec.inference_settings:
        args.extend(["--inference-settings", spec.inference_settings])
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
    hostname = socket.gethostname()
    logf.write(f"[launcher] event=started pid={p.pid} hostname={hostname} bind={bind}\n".encode())
    logf.flush()

    handle = UmaServerHandle(
        bind=bind, log_path=str(log_path), _preserve_dir=preserve_dir,
        pid=p.pid, hostname=hostname,
    )
    failed = True
    startup_failed = False
    try:
        ready = False
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if p.poll() is not None:
                break
            try:
                if _healthz_ready(bind):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            startup_failed = True
            raise RuntimeError("OET UMA server failed to start; its log will be preserved")
        yield handle
        failed = False
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                if p.poll() is None:
                    p.kill()
            p.wait(timeout=10)
        logf.write(f"[launcher] event=stopped pid={p.pid} hostname={hostname}\n".encode())
        logf.flush()
        logf.close()
        if startup_failed or log_policy == "always":
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
        elif log_policy == "on_failure":
            handle.preserve()
        if temp_log_dir:
            shutil.rmtree(active_log_dir, ignore_errors=True)
