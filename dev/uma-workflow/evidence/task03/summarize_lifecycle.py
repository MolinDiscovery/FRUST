"""Summarize one submitted UMA lifecycle check from saved run artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd


def summarize(root: Path) -> dict:
    """Return server, client, and stage evidence from one run directory.

    Parameters
    ----------
    root : pathlib.Path
        Directory written by ``submit_lifecycle.py`` and its target job.

    Returns
    -------
    dict
        Compact audit with process identities, hostnames, binds, and stage
        termination states.
    """
    submission = json.loads((root / "submission.json").read_text())
    target_dir = root / "workflow" / "water"
    result = pd.read_parquet(target_dir / "final.parquet")
    names = ("uma_sp", "uma_opt", "uma_numfreq")
    stages = {}
    for name in names:
        attrs = result.attrs["frust_steps"][name]["input"]
        stages[name] = {
            "normal_termination": bool(result.loc[0, f"{name}-NT"]),
            "energy_Eh": float(result.loc[0, f"{name}-EE"]),
            "server_pid": attrs["uma_server_pid"],
            "server_hostname": attrs["uma_server_hostname"],
            "server_bind": attrs["uma_server_bind"],
        }
    stages["uma_numfreq"]["frequencies_cm1"] = [
        float(item["frequency"]) for item in result.loc[0, "uma_numfreq-vibs"]
    ]

    logs = list((root / "server-logs").glob("oet_uma_server_*.log"))
    events = []
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            match = re.search(
                r"event=(started|stopped) pid=(\d+) hostname=(\S+)", line
            )
            if match:
                events.append(
                    {
                        "event": match.group(1),
                        "pid": int(match.group(2)),
                        "hostname": match.group(3),
                        "log": str(log),
                    }
                )
    live = set()
    peak = 0
    for event in events:
        if event["event"] == "started":
            live.add(event["pid"])
            peak = max(peak, len(live))
        else:
            live.discard(event["pid"])

    client_lines = (root / "client_calls.log").read_text().splitlines()
    clients = [dict(part.split("=", 1) for part in line.split()) for line in client_lines]
    return {
        "frust_revision": submission["frust_revision"],
        "job_id": submission["job_ids"][0],
        "submission_hostname": submission["submission_host"],
        "oet_runtime": submission["oet_runtime"],
        "stages": stages,
        "server_events": events,
        "peak_live_servers": peak,
        "servers_remaining_after_job": sorted(live),
        "client_call_count": len(clients),
        "client_hostnames": sorted({client["host"] for client in clients}),
        "client_binds": sorted({client["bind"] for client in clients}),
        "client_pids": sorted({int(client["pid"]) for client in clients}),
    }


def main() -> None:
    """Write the audit summary and fail when lifecycle invariants are broken."""
    root = Path(sys.argv[1])
    summary = summarize(root)
    (root / "lifecycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    stages = list(summary["stages"].values())
    server_pid = stages[0]["server_pid"]
    server_host = stages[0]["server_hostname"]
    server_bind = stages[0]["server_bind"]
    if not all(stage["normal_termination"] for stage in stages):
        raise RuntimeError("At least one UMA calculation failed")
    if any(stage["server_pid"] != server_pid for stage in stages):
        raise RuntimeError("UMA stages used different server processes")
    if summary["peak_live_servers"] != 1 or summary["servers_remaining_after_job"]:
        raise RuntimeError("UMA server start/stop events do not balance")
    if summary["client_hostnames"] != [server_host]:
        raise RuntimeError("UMA client and server hostnames differ")
    if summary["client_binds"] != [server_bind] or not server_bind.startswith("127.0.0.1:"):
        raise RuntimeError("UMA clients did not use the server's loopback bind")
    if summary["client_call_count"] < 3:
        raise RuntimeError("Expected client calls across SP, optimization, and NumFreq")
    if not summary["stages"]["uma_numfreq"]["frequencies_cm1"]:
        raise RuntimeError("NumFreq returned no vibrational frequencies")


if __name__ == "__main__":
    main()
