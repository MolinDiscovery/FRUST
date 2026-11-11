"""Read and display target progress for tracked single-job Slurm workflows."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from frust.cluster.progress_eta import _Batch, _Estimate, _Target, _estimate_worker_time, _timestamp
from frust.cluster.submission import _completion_path, _submission_history
from frust.utils.timing import format_duration


_BRANCH_LABELS = {
    "references": "References",
    "transition_states": "Transition states",
    "cycle_molecules": "Molecular states",
    "int3": "Int3",
    "intermediates": "Intermediates",
}
_TERMINAL_JOBS = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
_ACTIVE_JOBS = {"RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT"}
_QUEUED_JOBS = {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"}
_TARGET_ISSUES = {"non_normal", "failed", "interrupted", "blocked", "missing", "stale", "unreadable"}
_FINISHED_TARGETS = {"succeeded", "issues"}
_PENDING_CONTROLS = {"pending", "running", "unverified"}


@dataclass(frozen=True)
class _ControlJob:
    job_id: str | None
    finished: bool
    report_status: str | None
    finished_at: float | None = None


@dataclass(frozen=True)
class _Branch:
    label: str
    batches: list[_Batch]
    collector: _ControlJob | None
    submitted_ns: int | None


@dataclass(frozen=True)
class _Snapshot:
    root: Path
    branches: list[tuple[str, Counter[str], str | None]]
    finalization: str | None
    scheduler_errors: list[str]
    observed_at: float
    calculation: _Estimate | None
    ready_at: float | None
    submitted_at: float | None


def _is_run_root(path: Path) -> bool:
    return any(next((path / ".frust" / name).glob("*.json"), None) is not None
               for name in ("screen_submissions", "submissions"))


def _find_run_root(directory: Path) -> Path:
    root = directory.expanduser().resolve()
    if _is_run_root(root):
        return root
    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")
    matches = sorted({path.resolve() for path in root.iterdir() if path.is_dir() and _is_run_root(path)})
    if not matches:
        raise ValueError(f"No tracked FRUST submission found in {root} or its immediate subdirectories")
    if len(matches) > 1:
        candidates = "\n  ".join(str(path) for path in matches)
        raise ValueError(f"Multiple FRUST runs found. Pass one of these directories:\n  {candidates}")
    return matches[0]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # The screen finalizer can be rewriting run_report.json during a poll.
        return {}


def _job_id(value: str | int | None) -> str | None:
    job_id = str(value)
    return job_id if re.fullmatch(r"\d+(?:_\d+)?", job_id) else None


def _finished_at(marker: Path) -> float | None:
    finished_ns = _read_json(marker).get("finished_ns")
    return finished_ns / 1e9 if isinstance(finished_ns, int) and finished_ns > 0 else None


def _standalone_report(root: Path, path: Path, attempt: dict, finished: bool) -> dict:
    expected = {(attempt["attempt_id"], record["target"]) for record in attempt["records"]}
    candidates = [path]
    if finished:
        # The collector mirrors reports here, including reports with custom paths.
        candidates.extend(sorted((root / ".frust" / "reports").glob("*.json"),
                                 key=lambda candidate: candidate.stat().st_mtime_ns, reverse=True))
    for candidate in candidates:
        report = _read_json(candidate)
        actual = {(row.get("attempt_id"), row["target"]) for row in report.get("target_results", [])}
        if actual and actual == expected:
            return report
    return {}


def _read_branch(root: Path, label: str, screens: list[dict]) -> _Branch:
    history = _submission_history(root)
    latest = {}
    outcomes = {}
    report = {}
    collector_marker = None
    screen_times = [screen["created_ns"] for screen in screens]
    for attempt in history:
        identity = attempt["attempt_id"]
        mode = attempt.get("mode")
        backend = attempt.get("backend")
        if mode != "single_job" or backend != "slurm":
            raise ValueError(f"{root} uses {backend}/{mode}. Progress requires a tracked Slurm single_job submission")
        for record in attempt["records"]:
            latest[record["target"]] = (identity, record)
        collector_marker = _completion_path(root, identity, "collect", 0) if attempt.get("collection_job_id") else None
        collector_finished = collector_marker is not None and collector_marker.is_file()
        if screens:
            index = bisect_right(screen_times, attempt["created_ns"]) - 1
            report = _read_json(root / f"collection_report_{screens[index]['attempt_id']}.json") if index >= 0 else {}
        else:
            report_dir = root / ".frust" / "retries" / identity if attempt.get("previous_attempts") else root
            report = _standalone_report(root, report_dir / "collection_report.json", attempt, collector_finished)
        for result in report.get("target_results", []):
            outcomes[result.get("attempt_id"), result["target"]] = result["status"]

    grouped = {}
    for tag, (identity, record) in latest.items():
        grouped.setdefault((identity, record["batch_index"]), []).append((tag, record))
    batches = []
    for (identity, index), records in sorted(grouped.items()):
        batch = _read_json(root / ".frust" / "batches" / identity / f"{index}.json")
        marker = _completion_path(root, identity, "single_job", index)
        ended = _timestamp(batch.get("finished_at"))
        finished = bool(batch.get("finished_at")) or marker.is_file()
        if ended is None and finished:
            ended = _finished_at(marker)
        recorded = {item["target"]: item for item in batch.get("targets", [])}
        targets = []
        for tag, _ in records:
            row = recorded.get(tag, {})
            status = outcomes.get((identity, tag))
            if status == "unattempted":
                status = "missing"
            if status not in _TARGET_ISSUES and status != "success":
                status = row.get("status")
            targets.append(_Target(tag, status, _timestamp(row.get("started_at")), _timestamp(row.get("finished_at"))))
        batches.append(_Batch(identity, _job_id(records[0][1].get("job_id")), tuple(targets),
                              _timestamp(batch.get("started_at")), ended, finished))

    collector = None
    if history and (job_id := _job_id(history[-1].get("collection_job_id"))):
        report_status = None
        if "n_targets" in report and "n_collected" in report:
            complete = report["n_targets"] == report["n_collected"] and not report.get("n_failures")
            report_status = "collected" if complete else "finished with issues"
        ended = _finished_at(collector_marker) if collector_finished and report_status == "collected" else None
        collector = _ControlJob(job_id, collector_finished, report_status, ended)
    submitted_ns = history[0].get("created_ns") if history else None
    return _Branch(label, batches, collector, submitted_ns)


def _finalization_report(root: Path, identity: str) -> tuple[str | None, float | None]:
    report = _read_json(root / "run_report.json")
    expected_report = f"collection_report_{identity}.json"
    current = any(Path(branch.get("collection_report") or "").name == expected_report
                  for branch in (report.get("submission") or {}).values())
    status = {"success": "ready", "partial": "finished with issues", "failed": "failed"}
    if not current:
        return None, None
    return status.get(report.get("overall_status")), _timestamp(report.get("finalized_at"))


def _slurm_states(command: list[str], errors: list[str]) -> dict[str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        errors.append(command[0])
        return {}
    states = {}
    for line in result.stdout.splitlines():
        job_id, _, state = line.partition("|")
        job_id = job_id.strip()
        if _job_id(job_id) and state.strip():
            states[job_id] = state.split()[0].rstrip("+")
    return states


def _query_slurm(jobs: set[str]) -> tuple[dict[str, str], list[str]]:
    if not jobs:
        return {}, []
    errors: list[str] = []
    parents = sorted({job.split("_")[0] for job in jobs})
    states = _slurm_states(["squeue", "-r", "-h", "-j", ",".join(parents), "-o", "%i|%T"], errors)
    unresolved = sorted({job.split("_")[0] for job in jobs - states.keys()})
    if unresolved:
        accounting = _slurm_states([
            "sacct", "-n", "-X", "-j", ",".join(unresolved), "-P", "--format=JobID%64,State%40",
        ], errors)
        states.update((job, state) for job, state in accounting.items()
                      if job in jobs and job not in states and state in _TERMINAL_JOBS)
    return states, errors


def _target_state(status: str | None, finished: bool, job_state: str | None) -> str:
    if status == "success":
        return "succeeded"
    if status in _TARGET_ISSUES or finished or job_state in _TERMINAL_JOBS:
        return "issues"
    if job_state in _ACTIVE_JOBS:
        return "running" if status == "running" else "waiting in batch"
    if job_state in _QUEUED_JOBS:
        return "queued"
    return "unverified"


def _control_state(job: _ControlJob | None, states: dict[str, str]) -> str | None:
    if job is None:
        return None
    state = states.get(job.job_id) if job.job_id is not None else None
    if job.finished or state in _TERMINAL_JOBS:
        if job.report_status == "finished with issues":
            return job.report_status
        if state in _TERMINAL_JOBS - {"COMPLETED"}:
            return "failed"
        return job.report_status or "finished without a verified report"
    if state in _ACTIVE_JOBS:
        return "running"
    if state in _QUEUED_JOBS:
        return "pending"
    return "unverified"


def _branch_calculation(
    branch: _Branch, remaining: list[_Batch], states: dict[str, str], manifest: Path, now: float,
) -> _Estimate:
    attempts = {batch.attempt for batch in remaining}
    if len(attempts) != 1:
        return _Estimate(reason="waiting for one active submission attempt")
    for batch in remaining:
        state = states.get(batch.job_id or "")
        if state != "RUNNING" and state not in _QUEUED_JOBS:
            return _Estimate(reason="waiting for Slurm worker status")
        if batch.start is not None and (batch.start > now or state != "RUNNING"):
            return _Estimate(reason="waiting for current batch timings")
        if any(t.start is not None and (batch.start is None or not batch.start <= t.start <= now)
               for t in batch.targets):
            return _Estimate(reason="waiting for current target timings")
    identity = attempts.pop()
    history = [batch for batch in branch.batches if batch.attempt == identity]
    return _estimate_worker_time(history, remaining, manifest, now, references=branch.label == "References")


def _read_snapshot(root: Path) -> _Snapshot:
    screens = sorted(
        (json.loads(path.read_text()) for path in (root / ".frust" / "screen_submissions").glob("*.json")),
        key=lambda screen: screen["created_ns"],
    )
    finalizer = None
    finalized_at = None
    if screens:
        if screens[-1].get("backend") != "slurm":
            raise ValueError(f"{root} is not a tracked Slurm screen")
        from frust.workflows.screening import _branch_dir

        branches = []
        for key, label in _BRANCH_LABELS.items():
            path = _branch_dir(root, key)
            if _is_run_root(path):
                branches.append(_read_branch(path, label, screens))
        identity = screens[-1]["attempt_id"]
        finalizer_marker = _completion_path(root, identity, "screen_finalize", 0)
        finalizer = _ControlJob(_job_id(screens[-1].get("job_id")),
                                finalizer_marker.is_file(), None)
    else:
        branches = [_read_branch(root, "Workflow", [])]

    jobs = {
        batch.job_id for branch in branches for batch in branch.batches
        if batch.job_id and any(_target_state(target.status, batch.finished, None) == "unverified"
                                for target in batch.targets)
    }
    for control in [*(branch.collector for branch in branches), finalizer]:
        if control and control.job_id and not control.finished:
            jobs.add(control.job_id)
    states, errors = _query_slurm(jobs)
    if finalizer and (finalizer.finished or states.get(finalizer.job_id or "") in _TERMINAL_JOBS):
        report_status, finalized_at = _finalization_report(root, identity)
        ended = _finished_at(finalizer_marker) if report_status == "ready" else None
        finalizer = replace(finalizer, report_status=report_status, finished_at=ended)
    now = time.time()
    summary = []
    calculation = None
    for branch in branches:
        counts = Counter()
        remaining = []
        for batch in branch.batches:
            pending = []
            for target in batch.targets:
                state = _target_state(target.status, batch.finished, states.get(batch.job_id or ""))
                counts[state] += 1
                if state not in _FINISHED_TARGETS:
                    pending.append(target)
            if pending:
                remaining.append(replace(batch, targets=tuple(pending)))
        summary.append((branch.label, counts, _control_state(branch.collector, states)))
        if remaining and (calculation is None or calculation.reason is None):
            estimate = _branch_calculation(branch, remaining, states, root / "manifest.json", now)
            if estimate.reason:
                suffix = f" ({branch.label})" if screens else ""
                calculation = _Estimate(reason=estimate.reason + suffix)
            else:
                calculation = _Estimate(seconds=max(estimate.seconds, calculation.seconds if calculation else 0))

    finalization = _control_state(finalizer, states)
    ready_at = None
    if finalizer and finalization == "ready" and finalizer.finished_at is not None:
        if (finalized_at or 0) <= finalizer.finished_at <= now:
            ready_at = finalizer.finished_at
    elif finalizer is None and summary[0][2] == "collected" and calculation is None:
        ended = branches[0].collector.finished_at
        if ended is not None and ended <= now:
            ready_at = ended
    submitted_ns = screens[0]["created_ns"] if screens else branches[0].submitted_ns
    submitted_at = submitted_ns / 1e9 if isinstance(submitted_ns, int) and submitted_ns > 0 else None
    return _Snapshot(root, summary, finalization, errors, now, calculation, ready_at, submitted_at)


def _format_progress(label: str, counts: Counter[str], collection: str | None = None) -> str:
    total = sum(counts.values())
    finished = counts["succeeded"] + counts["issues"]
    ratio = finished / total if total else 0.0
    filled = int(ratio * 20)
    bar = "#" * filled + "-" * (20 - filled)
    percentage = "99.9%" if finished < total and ratio >= 0.9995 else f"{ratio:.1%}"
    line = f"{label:18} [{bar}]  {finished:>6,} / {total:>6,}  {percentage:>6}"
    if collection:
        label = collection if collection == "collected" else f"collection {collection}"
        line = f"{line}  {label}"
    return line


def _format_snapshot(snapshot: _Snapshot) -> str:
    overall: Counter[str] = Counter()
    for _, counts, _ in snapshot.branches:
        overall.update(counts)
    updated = datetime.fromtimestamp(snapshot.observed_at).astimezone().strftime("%H:%M:%S %Z")
    lines = [f"FRUST progress  |  {snapshot.root}"]
    end = snapshot.ready_at if snapshot.ready_at is not None else snapshot.observed_at
    if snapshot.submitted_at is not None and snapshot.submitted_at <= end:
        submitted = datetime.fromtimestamp(snapshot.submitted_at).astimezone().strftime("%a %d %b %H:%M %Z")
        lines.append(f"Submitted {submitted}  |  Elapsed {format_duration(end - snapshot.submitted_at)}")
    lines.extend([f"Updated {updated}", "", "Finished targets"])
    if snapshot.finalization is None and len(snapshot.branches) == 1:
        lines.append(_format_progress("Overall", overall, snapshot.branches[0][2]))
    else:
        lines.append(_format_progress("Overall", overall))
        lines.extend(_format_progress(label, counts, collection) for label, counts, collection in snapshot.branches)
    labels = ("running", "waiting in batch", "queued", "issues", "unverified")
    lines.extend(["", "  |  ".join(f"{overall[label]:,} {label}" for label in labels)])
    if snapshot.calculation:
        if snapshot.calculation.seconds is None:
            lines.append(f"Calculation ETA unavailable: {snapshot.calculation.reason}")
        else:
            finish = math.ceil((snapshot.observed_at + snapshot.calculation.seconds) / 300) * 300
            local = datetime.fromtimestamp(finish).astimezone().strftime("%a %d %b %H:%M %Z")
            lines.append(f"Calculation ETA: around {local} at recent worker capacity")
            pending = [f"{label.lower()} collection" if snapshot.finalization else "collection"
                       for label, _, status in snapshot.branches if status in _PENDING_CONTROLS]
            if snapshot.finalization in _PENDING_CONTROLS:
                pending.append("finalization")
            if pending:
                lines.append(f"Results still need: {', '.join(pending)}")
    if snapshot.finalization:
        lines.append(f"Finalization: {snapshot.finalization}")
    if snapshot.ready_at is not None:
        local = datetime.fromtimestamp(snapshot.ready_at).astimezone().strftime("%a %d %b %H:%M %Z")
        lines.append(f"Results ready at: {local}")
    if snapshot.scheduler_errors:
        lines.append(f"Slurm status unavailable from {', '.join(snapshot.scheduler_errors)}")
    return "\n".join(lines)


def _show_progress(directory: Path, *, once: bool) -> None:
    root = _find_run_root(directory)
    snapshot = _format_snapshot(_read_snapshot(root))
    if once or not sys.stdout.isatty():
        print(snapshot)
        return

    print("\x1b[?1049h", end="", flush=True)
    try:
        while True:
            print(f"\x1b[H\x1b[J{snapshot}\n\nRefreshes every 30s. Ctrl+C closes the view.", flush=True)
            time.sleep(30)
            try:
                snapshot = _format_snapshot(_read_snapshot(root))
            except json.JSONDecodeError:
                continue
    finally:
        print("\x1b[?1049l", end="", flush=True)
