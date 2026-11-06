"""Read and display target progress for tracked single-job Slurm workflows."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from frust.cluster.submission import _completion_path, _submission_history


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


@dataclass(frozen=True)
class _ControlJob:
    job_id: str | None
    finished: bool
    report_status: str | None


@dataclass(frozen=True)
class _Target:
    job_id: str | None
    status: str | None
    worker_finished: bool


@dataclass(frozen=True)
class _Branch:
    label: str
    targets: list[_Target]
    collector: _ControlJob | None


@dataclass(frozen=True)
class _Snapshot:
    root: Path
    branches: list[tuple[str, Counter[str], str | None]]
    finalization: str | None
    scheduler_errors: list[str]


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
    collector_finished = False
    screen_times = [screen["created_ns"] for screen in screens]
    for attempt in history:
        identity = attempt["attempt_id"]
        mode = attempt.get("mode")
        backend = attempt.get("backend")
        if mode != "single_job" or backend != "slurm":
            raise ValueError(f"{root} uses {backend}/{mode}. Progress requires a tracked Slurm single_job submission")
        for record in attempt["records"]:
            latest[record["target"]] = (identity, record)
        collector_finished = bool(attempt.get("collection_job_id")) and _completion_path(root, identity, "collect", 0).is_file()
        if screens:
            index = bisect_right(screen_times, attempt["created_ns"]) - 1
            report = _read_json(root / f"collection_report_{screens[index]['attempt_id']}.json") if index >= 0 else {}
        else:
            report_dir = root / ".frust" / "retries" / identity if attempt.get("previous_attempts") else root
            report = _standalone_report(root, report_dir / "collection_report.json", attempt, collector_finished)
        for result in report.get("target_results", []):
            outcomes[result.get("attempt_id"), result["target"]] = result["status"]

    batches = {}
    targets = []
    for tag, (identity, record) in latest.items():
        index = record["batch_index"]
        key = (identity, index)
        if key not in batches:
            batch = _read_json(root / ".frust" / "batches" / identity / f"{index}.json")
            finished = bool(batch.get("finished_at")) or _completion_path(root, identity, "single_job", index).is_file()
            batches[key] = ({item["target"]: item["status"] for item in batch.get("targets", [])}, finished)
        batch_statuses, finished = batches[key]
        status = outcomes.get((identity, tag))
        if status == "unattempted":
            status = "missing"
        if status not in _TARGET_ISSUES and status != "success":
            status = batch_statuses.get(tag)
        targets.append(_Target(_job_id(record.get("job_id")), status, finished))

    collector = None
    if history and (job_id := _job_id(history[-1].get("collection_job_id"))):
        report_status = None
        if "n_targets" in report and "n_collected" in report:
            complete = report["n_targets"] == report["n_collected"] and not report.get("n_failures")
            report_status = "collected" if complete else "finished with issues"
        collector = _ControlJob(job_id, collector_finished, report_status)
    return _Branch(label, targets, collector)


def _finalization_report_status(root: Path, identity: str) -> str | None:
    report = _read_json(root / "run_report.json")
    expected_report = f"collection_report_{identity}.json"
    current = any(Path(branch.get("collection_report") or "").name == expected_report
                  for branch in report.get("submission", {}).values())
    status = {"success": "ready", "partial": "finished with issues", "failed": "failed"}
    return status.get(report.get("overall_status")) if current else None


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


def _read_snapshot(root: Path) -> _Snapshot:
    screens = sorted(
        (json.loads(path.read_text()) for path in (root / ".frust" / "screen_submissions").glob("*.json")),
        key=lambda screen: screen["created_ns"],
    )
    finalizer = None
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
        finalizer = _ControlJob(_job_id(screens[-1].get("job_id")),
                                _completion_path(root, identity, "screen_finalize", 0).is_file(), None)
    else:
        branches = [_read_branch(root, "Workflow", [])]

    jobs = {
        target.job_id for branch in branches for target in branch.targets
        if target.job_id and _target_state(target.status, target.worker_finished, None) == "unverified"
    }
    for control in [*(branch.collector for branch in branches), finalizer]:
        if control and control.job_id and not control.finished:
            jobs.add(control.job_id)
    states, errors = _query_slurm(jobs)
    if finalizer and (finalizer.finished or states.get(finalizer.job_id or "") in _TERMINAL_JOBS):
        finalizer = replace(finalizer, report_status=_finalization_report_status(root, identity))
    summary = []
    for branch in branches:
        counts = Counter(
            _target_state(target.status, target.worker_finished, states.get(target.job_id or ""))
            for target in branch.targets
        )
        summary.append((branch.label, counts, _control_state(branch.collector, states)))
    return _Snapshot(root, summary, _control_state(finalizer, states), errors)


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
    updated = datetime.now().astimezone().strftime("%H:%M:%S %Z")
    lines = [f"FRUST progress  |  {snapshot.root}", f"Updated {updated}", "", "Finished targets"]
    if snapshot.finalization is None and len(snapshot.branches) == 1:
        lines.append(_format_progress("Overall", overall, snapshot.branches[0][2]))
    else:
        lines.append(_format_progress("Overall", overall))
        lines.extend(_format_progress(label, counts, collection) for label, counts, collection in snapshot.branches)
    labels = ("running", "waiting in batch", "queued", "issues", "unverified")
    lines.extend(["", "  |  ".join(f"{overall[label]:,} {label}" for label in labels)])
    if snapshot.finalization:
        lines.append(f"Finalization: {snapshot.finalization}")
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
