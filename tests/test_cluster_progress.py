import subprocess
from collections import Counter
from datetime import datetime, timezone
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
    assert snapshot.calculation.reason == "waiting for Slurm worker status"


def test_retry_keeps_other_targets_and_rejects_old_final_report(tmp_path, monkeypatch):
    refs = tmp_path / "calculations" / "references"
    submitted_ns = 1_794_000_000_000_000_000
    monkeypatch.setattr(progress.time, "time", lambda: 1_794_000_120)
    _screen(tmp_path, "first", created_ns=submitted_ns)
    first = _submit(refs, [("ok", "retry")], ["101_0"], created_ns=submitted_ns + 1, collector="102")
    _batch(refs, first, {"ok": "success", "retry": "failed"}, finished=True)
    _atomic_write_submission_json(tmp_path / "run_report.json", {
        "overall_status": "partial",
        "submission": {"references": {"collection_report": str(refs / "collection_report_first.json")}},
    })
    retried_ns = submitted_ns + 30_000_000_000
    _screen(tmp_path, "second", created_ns=retried_ns, job="201")
    second = _submit(refs, [("retry",)], ["105_0"], created_ns=retried_ns + 1,
                     collector="106", previous={"retry": first})
    _slurm(monkeypatch, queue="105_0|PENDING\n106|PENDING\n201|PENDING\n")

    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0][1] == {"succeeded": 1, "queued": 1}
    assert snapshot.finalization == "pending"
    assert "Elapsed 2m 00s" in progress._format_snapshot(snapshot)

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
    completed_ns = 1_794_000_060_000_000_000
    _mark_completed(tmp_path, "second", "screen_finalize", 0, finished_ns=completed_ns)
    assert progress._read_snapshot(tmp_path).finalization == "finished without a verified report"
    for status, display in [("partial", "finished with issues"), ("success", "ready")]:
        _atomic_write_submission_json(tmp_path / "run_report.json", {
            "overall_status": status,
            "submission": {"references": {"collection_report": str(refs / "collection_report_second.json")}},
            "finalized_at": datetime.fromtimestamp(1_794_000_000, timezone.utc).isoformat(),
        })
        snapshot = progress._read_snapshot(tmp_path)
        assert snapshot.finalization == display
        assert snapshot.ready_at == (completed_ns / 1e9 if status == "success" else None)
        output = progress._format_snapshot(snapshot)
        assert ("Results ready at:" in output) == (status == "success")
        if status == "success":
            assert "Elapsed 1m 00s" in output


def test_calculation_eta_accounts_for_running_and_queued_targets(tmp_path, monkeypatch):
    now = datetime(2026, 11, 6, 18, 45, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(progress.time, "time", lambda: now)
    _screen(tmp_path)
    branch = tmp_path / "calculations" / "transition_states"
    targets = [f"TS1__substrate__cat__r{index}" for index in range(13)]
    attempt = _submit(branch, [tuple(targets[:10]), (targets[10],), (targets[11],), (targets[12],)],
                      ["100_0", "100_1", "100_2", "100_3"], collector="101")
    _atomic_write_submission_json(tmp_path / "manifest.json", {
        "components": [{"compound_name": "substrate", "smiles": "c1ccccc1"},
                       {"compound_name": "cat", "smiles": "C"}],
        "systems": [{"system_name": "substrate__cat", "substrate_name": "substrate", "catalyst_name": "cat"}],
    })

    def stamp(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    def record(index, identity, payload):
        _atomic_write_submission_json(branch / ".frust" / "batches" / identity / f"{index}.json", payload)

    record(0, attempt, {
        "started_at": stamp(now - 1230), "finished_at": stamp(now - 30),
        "targets": [
            {"target": tag, "status": "success", "started_at": stamp(now - 1230 + index * 120),
             "finished_at": stamp(now - 1110 + index * 120)}
            for index, tag in enumerate(targets[:10])
        ],
    })

    def running(index, identity, age):
        record(index, identity, {
            "started_at": stamp(now - age),
            "targets": [{"target": targets[10], "status": "running", "started_at": stamp(now - age)}],
        })

    running(1, attempt, 30)
    _slurm(monkeypatch, queue="100_1|RUNNING\n100_2|PENDING\n100_3|PENDING\n101|PENDING\n200|PENDING\n")
    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.calculation.seconds == pytest.approx(330)
    output = progress._format_snapshot(snapshot)
    finish = datetime.fromtimestamp(now + 600).astimezone().strftime("%a %d %b %H:%M %Z")
    assert f"Calculation ETA: around {finish} at recent worker capacity" in output
    assert "Results still need: transition states collection, finalization" in output

    running(1, attempt, 150)
    moderate_outlier = progress._read_snapshot(tmp_path)
    assert moderate_outlier.calculation.reason is None
    assert 240 < moderate_outlier.calculation.seconds < 300

    running(1, attempt, 600)
    assert "outlasted comparable timings" in progress._read_snapshot(tmp_path).calculation.reason

    retry = _submit(branch, [(targets[10],), (targets[11],), (targets[12],)], ["102_0", "102_1", "102_2"],
                    created_ns=30, previous={tag: attempt for tag in targets[10:]})
    running(0, retry, 30)
    _slurm(monkeypatch, queue="102_0|RUNNING\n102_1|PENDING\n102_2|PENDING\n200|PENDING\n")
    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches[0][1]["succeeded"] == 10
    assert snapshot.calculation.reason == "waiting for more completed target timings (Transition states)"


def test_calculation_eta_absorbs_runaway_worker_when_parallel_pool_dominates(tmp_path, monkeypatch):
    now = datetime(2026, 11, 6, 18, 45, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(progress.time, "time", lambda: now)
    _screen(tmp_path)
    branch = tmp_path / "calculations" / "transition_states"
    targets = [f"TS1__substrate__cat__r{index}" for index in range(30)]
    batches = [tuple(targets[:10])] + [(tag,) for tag in targets[10:]]
    jobs = [f"100_{index}" for index in range(len(batches))]
    attempt = _submit(branch, batches, jobs, collector="101")
    _atomic_write_submission_json(tmp_path / "manifest.json", {
        "components": [{"compound_name": "substrate", "smiles": "c1ccccc1"},
                       {"compound_name": "cat", "smiles": "C"}],
        "systems": [{"system_name": "substrate__cat", "substrate_name": "substrate", "catalyst_name": "cat"}],
    })

    def stamp(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    def record(index, payload):
        _atomic_write_submission_json(branch / ".frust" / "batches" / attempt / f"{index}.json", payload)

    record(0, {
        "started_at": stamp(now - 1230), "finished_at": stamp(now - 30),
        "targets": [
            {"target": tag, "status": "success", "started_at": stamp(now - 1230 + index * 120),
             "finished_at": stamp(now - 1110 + index * 120)}
            for index, tag in enumerate(targets[:10])
        ],
    })
    # One worker is at 5x the comparable duration, while three healthy workers and 16 queued batches dominate.
    record(1, {
        "started_at": stamp(now - 600),
        "targets": [{"target": targets[10], "status": "running", "started_at": stamp(now - 600)}],
    })
    for index in (2, 3, 4):
        record(index, {
            "started_at": stamp(now - 30),
            "targets": [{"target": targets[9 + index], "status": "running", "started_at": stamp(now - 30)}],
        })
    queue_lines = [f"100_{index}|RUNNING" for index in range(1, 5)]
    queue_lines.extend(f"100_{index}|PENDING" for index in range(5, 21))
    queue_lines.extend(["101|PENDING", "200|PENDING"])
    _slurm(monkeypatch, queue="\n".join(queue_lines) + "\n")

    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.calculation.reason is None
    assert 500 < snapshot.calculation.seconds < 700

    # Once all other batches finish and only the runaway worker remains, withhold the ETA.
    for index in range(2, 21):
        record(index, {
            "started_at": stamp(now - 180), "finished_at": stamp(now - 60),
            "targets": [{"target": targets[9 + index], "status": "success",
                         "started_at": stamp(now - 180), "finished_at": stamp(now - 60)}],
        })
    _slurm(monkeypatch, queue="100_1|RUNNING\n101|PENDING\n200|PENDING\n")
    tail_snapshot = progress._read_snapshot(tmp_path)
    assert tail_snapshot.calculation.reason == "a limiting active target has outlasted comparable timings (Transition states)"


def test_standalone_collector_reads_only_matching_report(tmp_path, monkeypatch):
    submitted_ns = 1_794_000_000_000_000_000
    attempt = _submit(tmp_path, [("anisole",)], ["100"], created_ns=submitted_ns, collector="101")
    _mark_completed(tmp_path, attempt, "collect", 0, finished_ns=submitted_ns + 95_000_000_000)
    monkeypatch.setattr(progress.time, "time", lambda: 1_794_000_120)
    report = {"n_targets": 1, "n_collected": 1, "n_failures": 0, "target_results": [
        {"target": "anisole", "attempt_id": "older", "status": "success"},
    ]}
    _atomic_write_submission_json(tmp_path / "collection_report.json", report)
    report["target_results"][0]["attempt_id"] = attempt
    _atomic_write_submission_json(tmp_path / ".frust" / "reports" / "custom.json", report)
    calls = _slurm(monkeypatch)

    snapshot = progress._read_snapshot(tmp_path)
    assert snapshot.branches == [("Workflow", {"succeeded": 1}, "collected")]
    assert "Elapsed 1m 35s" in progress._format_snapshot(snapshot)
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
