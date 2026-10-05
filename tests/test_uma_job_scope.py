"""UMA server ownership across serial workflow stages."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from frust.stepper import Stepper
from frust.utils.uma import current_uma_job_scope, uma_job_server_scope
from frust.workflows.core import BaseWorkflow, StageDef, WorkflowTarget
from frust.workflows.methods import CalculatorSpec, MethodPlan


def _water() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "substrate_name": ["water"],
            "atoms": [["O", "H", "H"]],
            "coords_embedded": [
                [[0.0, 0.0, 0.0], [0.758602, 0.0, 0.504284], [-0.758602, 0.0, 0.504284]]
            ],
        }
    )


def _fake_orca(atoms, coords, n_cores, scr, data2file, options, xtra_inp_str, memory, read_files):
    return {
        "normal_termination": True,
        "electronic_energy": -76.0,
        "opt_coords": coords,
    }


class _TinyUmaWorkflow(BaseWorkflow):
    workflow_name = "tiny_uma_lifecycle"

    def _build_targets(self):
        return [WorkflowTarget("water", None)]

    def _prepare_initial_df(self, target, *, save_dir, options):
        return _water()

    def _stage_defs(self):
        return [
            StageDef("prepare", "prepare", kind="prepare"),
            StageDef("uma_sp", "UMA single point", n_cores=2),
            StageDef("uma_opt", "UMA optimization", n_cores=2),
            StageDef("uma_numfreq", "UMA numerical frequencies"),
        ]


def _method() -> MethodPlan:
    common = {"uma": "omol@uma-s-1p2p1", "uma_keep_logs": "always"}
    return MethodPlan(
        "tiny-uma-lifecycle",
        {
            "uma_sp": CalculatorSpec("orca", {"ExtOpt": None}, kwargs=common),
            "uma_opt": CalculatorSpec("orca", {"ExtOpt": None, "Opt": None}, kwargs=common),
            "uma_numfreq": CalculatorSpec("orca", {"ExtOpt": None, "NumFreq": None}, kwargs=common),
        },
    )


def _oet_root(root: Path) -> Path:
    runtime = root / "oet"
    bin_dir = runtime / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("oet_client", "oet_uma", "oet_server"):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
    return runtime


def test_workflow_reuses_one_server_for_sp_opt_and_numfreq(tmp_path):
    runtime = _oet_root(tmp_path)
    starts = []
    requests = []

    @contextmanager
    def fake_server(**kwargs):
        starts.append(("start", dict(kwargs), os.environ["OET_TOOLS"]))
        yield SimpleNamespace(
            bind="127.0.0.1:12345",
            pid=44221,
            hostname="node066",
            preserve=lambda: None,
        )
        starts.append(("stop",))

    def fake_stepper(**kwargs):
        step = Stepper(**kwargs)

        def record_orca(
            atoms, coords, n_cores, scr, data2file, options,
            xtra_inp_str, memory, read_files,
        ):
            requests.append(xtra_inp_str)
            return _fake_orca(
                atoms, coords, n_cores, scr, data2file, options,
                xtra_inp_str, memory, read_files,
            )

        step.orca_fn = record_orca
        return step

    with (
        patch("frust.utils.uma.uma_server", fake_server),
        patch("frust.utils.uma._healthz_ready", return_value=True),
        patch("frust.workflows.core.Stepper", side_effect=fake_stepper),
    ):
        result = _TinyUmaWorkflow(method=_method()).run(
            targets=[0], n_cores=4, mem_gb=2,
            save_output_dir=False, uma_oet_tools=runtime,
        )

    assert [event[0] for event in starts] == ["start", "stop"]
    assert starts[0][2] == str(runtime)
    assert starts[0][1]["server_cores"] == 4
    assert len(requests) == 3
    assert all("-b 127.0.0.1:12345" in request for request in requests)
    assert all(
        result.attrs["frust_steps"][stage]["input"]["uma_server_pid"] == 44221
        for stage in ("uma_sp", "uma_opt", "uma_numfreq")
    )
    assert all(
        result.attrs["frust_steps"][stage]["input"]["uma_server_cores"] == 4
        for stage in ("uma_sp", "uma_opt", "uma_numfreq")
    )
    assert current_uma_job_scope() is None


def test_job_scope_rejects_second_server_configuration_and_cleans_up(tmp_path):
    runtime = _oet_root(tmp_path)
    events = []

    @contextmanager
    def fake_server(**kwargs):
        events.append("start")
        try:
            yield SimpleNamespace(bind="127.0.0.1:12345")
        finally:
            events.append("stop")

    with patch("frust.utils.uma.uma_server", fake_server):
        with pytest.raises(ValueError, match="same server resources"):
            with uma_job_server_scope(oet_tools=str(runtime)) as scope:
                scope.acquire(
                    log_dir=None, keep_logs="on_failure", use_gpu=False,
                    server_cores=1, memory_per_thread_mib=500,
                )
                scope.acquire(
                    log_dir=None, keep_logs="on_failure", use_gpu=False,
                    server_cores=2, memory_per_thread_mib=500,
                )

    assert events == ["start", "stop"]
    assert current_uma_job_scope() is None


def test_failed_job_passes_exception_to_server_and_restores_runtime(tmp_path):
    runtime = _oet_root(tmp_path)
    received = []
    before = os.environ.get("OET_TOOLS")

    @contextmanager
    def fake_server(**kwargs):
        try:
            yield SimpleNamespace(bind="127.0.0.1:12345")
        except RuntimeError as exc:
            received.append(str(exc))
            raise

    with patch("frust.utils.uma.uma_server", fake_server):
        with pytest.raises(RuntimeError, match="stage failed"):
            with uma_job_server_scope(oet_tools=str(runtime)) as scope:
                scope.acquire(
                    log_dir=None, keep_logs="on_failure", use_gpu=False,
                    server_cores=1, memory_per_thread_mib=500,
                )
                raise RuntimeError("stage failed")

    assert received == ["stage failed"]
    assert os.environ.get("OET_TOOLS") == before


def test_sigterm_stops_server_during_active_job(tmp_path):
    runtime = _oet_root(tmp_path)
    server = runtime / "bin" / "oet_server"
    server.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            bind = sys.argv[sys.argv.index("--bind") + 1]
            host, port = bind.split(":")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()

                def log_message(self, format, *args):
                    pass

            ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
            """
        )
    )
    server.chmod(0o755)
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        from frust.utils.uma import uma_job_server_scope

        with uma_job_server_scope(oet_tools=sys.argv[1]) as scope:
            handle = scope.acquire(
                log_dir=sys.argv[2], keep_logs="always", use_gpu=False,
                server_cores=1, memory_per_thread_mib=500,
            )
            print(handle.pid, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(runtime), str(tmp_path / "logs")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 128 + signal.SIGTERM, result.stderr
    server_pid = int(result.stdout.strip())
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
    log = next((tmp_path / "logs").glob("oet_uma_server_*.log")).read_text()
    assert f"event=started pid={server_pid}" in log
    assert f"event=stopped pid={server_pid}" in log
