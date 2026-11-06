import subprocess
from collections import Counter
from types import SimpleNamespace

import pytest

from frust.cli import main
from frust.cluster import progress
from frust.cluster.config import Resources
from frust.cluster.submission import PlannedGroup, SubmissionLedger, _atomic_write_submission_json, _mark_completed


def _submit(root, batches, jobs, *, created_ns=20, collector=None, previous=None, mode="single_job"):
    group = PlannedGroup(mode, Resources(1, 1, 1), "final.parquet", tuple(batches), len(batches))
    ledger = SubmissionLedger(root, [group], mode=mode, backend="slurm", array=True)
    ledger.created_ns = created_ns
    ledger.previous_attempts = previous or {}
    ledger.submitted_many(mode, {i: SimpleNamespace(job_id=job) for i, job in enumerate(jobs)})
    if collector:
        ledger.collected(SimpleNamespace(job_id=collector))
    return ledger.attempt_id


def _batch(root, attempt, rows, *, index=0, finished=False):
    _atomic_write_submission_json(root / ".frust" / "batches" / attempt / f"{index}.json", {
        "attempt_id": attempt,
        "targets": [{"target": tag, "status": status} for tag, status in rows.items()],
        "finished_at": "2026-11-06T10:44:00Z" if finished else None,
    })


def _screen(root, attempt="screen", *, created_ns=10, job="200"):
    _atomic_write_submission_json(root / ".frust" / "screen_submissions" / f"{attempt}.json", {
        "attempt_id": attempt, "created_ns": created_ns, "backend": "slurm", "job_id": job,
    })


def _slurm(monkeypatch, *, queue="", accounting=""):
    calls = []

    def run(command, *, capture_output, text, check, timeout):
        assert capture_output and text and check and timeout > 0
        calls.append(command)
        result = {"squeue": queue, "sacct": accounting}[command[0]]
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(progress.subprocess, "run", run)
    return calls


def test_run_discovery_and_single_snapshot(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project" / "results"
    attempt = _submit(root, [("anisole",)], ["100_0"])
    _batch(root, attempt, {"anisole": "success"}, finished=True)
    monkeypatch.chdir(root.parent)
    calls = _slurm(monkeypatch)

    for argv in (["progress", "--once"], ["progress", str(root.parent)], ["progress", str(root)]):
        assert main(argv) == 0
        output = capsys.readouterr().out
        assert str(root) in output
        assert "100.0%" in output
        assert "\x1b" not in output
    assert calls == []

    another = root.with_name("second_run")
    _submit(another, [("TMP",)], ["101_0"])
    with pytest.raises(SystemExit) as error:
        main(["progress"])
    assert error.value.code == 1
    assert str(root) in capsys.readouterr().err
    with pytest.raises(ValueError, match="No tracked FRUST submission"):
        progress._find_run_root(tmp_path)


def test_screen_counts_targets_inside_running_batch(tmp_path, monkeypatch):
    root = tmp_path / "results"
    refs = root / "calculations" / "references"
    ts = root / "calculations" / "transition_states"
    _screen(root)
    ref_attempt = _submit(refs, [("TMP",)], ["101_0"], collector="102")
    ts_attempt = _submit(ts, [("a", "b", "c", "d", "e"), ("f",)], ["103_0", "103_1"], collector="104")
    _batch(ts, ts_attempt, {"a": "success", "b": "non_normal", "c": "running", "d": "unattempted", "e": "unattempted"})
    _atomic_write_submission_json(refs / "collection_report_screen.json", {
        "n_targets": 1, "n_collected": 1, "n_failures": 0,
        "target_results": [{"target": "TMP", "attempt_id": ref_attempt, "status": "success"}],
    })
    _mark_completed(refs, ref_attempt, "collect", 0)
    calls = _slurm(monkeypatch, queue="103_0|RUNNING\n103_1|PENDING\n104|PENDING\n200|PENDING\n")

    snapshot = progress._read_snapshot(root)
    assert snapshot.branches == [
        ("References", {"succeeded": 1}, "collected"),
        ("Transition states", {"succeeded": 1, "issues": 1, "running": 1, "waiting in batch": 2, "queued": 1}, "pending"),
    ]
    assert snapshot.finalization == "pending"
    assert len(calls) == 1
    assert calls[0][0] == "squeue"
    assert calls[0][calls[0].index("-j") + 1] == "103,104,200"


def test_cancelled_array_element_and_scheduler_outage(tmp_path, monkeypatch):
    attempt = _submit(tmp_path, [("ok", "interrupted"), ("unknown",)], ["103_0", "103_1"])
    _batch(tmp_path, attempt, {"ok": "success", "interrupted": "running"})
    calls = _slurm(monkeypatch, accounting="103_0|CANCELLED by 123\n103_0.batch|FAILED\n")
    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0][1] == {"succeeded": 1, "issues": 1, "unverified": 1}
    assert [command[0] for command in calls] == ["squeue", "sacct"]
    assert "--format=JobID%64,State%40" in calls[1]

    _slurm(monkeypatch, queue=FileNotFoundError(), accounting=subprocess.TimeoutExpired("sacct", 5))
    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0][1] == {"succeeded": 1, "unverified": 2}
    assert snapshot.scheduler_errors == ["squeue", "sacct"]


def test_retry_keeps_other_targets_and_rejects_old_final_report(tmp_path, monkeypatch):
    refs = tmp_path / "calculations" / "references"
    _screen(tmp_path, "first")
    first = _submit(refs, [("ok", "retry")], ["101_0"], collector="102")
    _batch(refs, first, {"ok": "success", "retry": "failed"}, finished=True)
    _atomic_write_submission_json(tmp_path / "run_report.json", {
        "overall_status": "partial",
        "submission": {"references": {"collection_report": str(refs / "collection_report_first.json")}},
    })
    _screen(tmp_path, "second", created_ns=30, job="201")
    second = _submit(refs, [("retry",)], ["105_0"], created_ns=40, collector="106", previous={"retry": first})
    _slurm(monkeypatch, queue="105_0|PENDING\n106|PENDING\n201|PENDING\n")

    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0][1] == {"succeeded": 1, "queued": 1}
    assert snapshot.finalization == "pending"

    _batch(refs, second, {"retry": "success"}, finished=True)
    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0] == ("References", {"succeeded": 2}, "pending")
    assert "100.0%" in progress._format_snapshot(snapshot)
    assert snapshot.finalization == "pending"

    report = {
        "overall_status": "success",
        "submission": {"references": {"collection_report": str(refs / "collection_report_second.json")}},
    }
    _atomic_write_submission_json(tmp_path / "run_report.json", report)
    assert progress._read_snapshot(tmp_path).finalization == "pending"

    report["submission"]["references"]["collection_report"] = str(refs / "collection_report_first.json")
    _atomic_write_submission_json(tmp_path / "run_report.json", report)
    _mark_completed(tmp_path, "second", "screen_finalize", 0)
    assert progress._read_snapshot(tmp_path).finalization == "finished without a verified report"
    for status, display in [("partial", "finished with issues"), ("success", "ready")]:
        _atomic_write_submission_json(tmp_path / "run_report.json", {
            "overall_status": status,
            "submission": {"references": {"collection_report": str(refs / "collection_report_second.json")}},
        })
        assert progress._read_snapshot(tmp_path).finalization == display


def test_standalone_collector_reads_only_matching_report(tmp_path, monkeypatch):
    attempt = _submit(tmp_path, [("anisole",)], ["100"], collector="101")
    _mark_completed(tmp_path, attempt, "collect", 0)
    report = {"n_targets": 1, "n_collected": 1, "n_failures": 0, "target_results": [
        {"target": "anisole", "attempt_id": "older", "status": "success"},
    ]}
    _atomic_write_submission_json(tmp_path / "collection_report.json", report)
    report["target_results"][0]["attempt_id"] = attempt
    _atomic_write_submission_json(tmp_path / ".frust" / "reports" / "custom.json", report)
    calls = _slurm(monkeypatch)

    assert progress._read_snapshot(tmp_path).branches == [("Workflow", {"succeeded": 1}, "collected")]
    assert calls == []


def test_progress_percentage_reserves_100_percent_for_all_targets():
    assert "99.9%" in progress._format_progress("Overall", Counter(succeeded=6_053, queued=1))


def test_live_view_restores_terminal_on_interrupt(tmp_path, monkeypatch, capsys):
    _submit(tmp_path, [("anisole",)], ["100_0"])
    _slurm(monkeypatch, queue="100_0|PENDING\n")
    monkeypatch.setattr(progress.sys.stdout, "isatty", lambda: True)

    def interrupt(seconds):
        assert seconds == 30
        raise KeyboardInterrupt

    monkeypatch.setattr(progress.time, "sleep", interrupt)
    assert main(["progress", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("\x1b[?1049h")
    assert output.endswith("\x1b[?1049l")


def test_staged_submission_reports_unsupported_mode(tmp_path):
    _submit(tmp_path, [("a",)], ["100_0"], mode="dft_staged")
    with pytest.raises(ValueError, match="slurm/dft_staged"):
        progress._read_snapshot(tmp_path)
