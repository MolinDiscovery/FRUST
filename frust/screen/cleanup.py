"""Safe lifecycle management for screening-mode Submitit artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

_OWNER = "frust-screening"


def initialize_submitit_directory(run_dir: str | Path) -> Path:
    """Create and mark the managed Submitit directory for a screening run."""
    root = Path(run_dir).expanduser().resolve()
    submitit_dir = root / ".submitit"
    if submitit_dir.is_symlink():
        raise ValueError("Refusing to manage a symlinked .submitit directory")
    if submitit_dir.exists() and any(submitit_dir.iterdir()):
        marker = submitit_dir / "ownership.json"
        if not marker.exists():
            raise FileExistsError(
                f"Existing Submitit directory has no FRUST ownership marker: {submitit_dir}"
            )
        _validate_owned_submitit(root)
    jobs = submitit_dir / "jobs"
    control = submitit_dir / "control"
    jobs.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": 1,
        "managed_by": _OWNER,
        "run_dir": str(root),
        "submitit_dir": str(submitit_dir),
    }
    (submitit_dir / "ownership.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    return submitit_dir


def cleanup_submitit(
    run_dir: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Remove a FRUST-owned Submitit directory after a screening run.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Catalyst-screen run containing ``.submitit/ownership.json``.
    allow_incomplete : bool, optional
        Permit deliberate removal after a failed or partial run. The ownership
        and path-safety checks always remain active.

    Returns
    -------
    dict
        Removed path and byte count.
    """
    root, submitit_dir = _validate_owned_submitit(run_dir)
    report_path = root / "run_report.json"
    status = None
    if report_path.exists():
        status = json.loads(report_path.read_text()).get("overall_status")
    if not allow_incomplete and status != "success":
        raise ValueError(
            "Submitit cleanup requires a successful run report; pass "
            "allow_incomplete=True only after debugging an incomplete run"
        )
    removed_bytes = _directory_size(submitit_dir)
    shutil.rmtree(submitit_dir)
    return {"path": str(submitit_dir), "removed_bytes": removed_bytes}


def cleanup_submitit_jobs(run_dir: str | Path) -> dict[str, Any]:
    """Remove only bulk job files after verified successful finalization."""
    _, submitit_dir = _validate_owned_submitit(run_dir)
    jobs = submitit_dir / "jobs"
    if jobs.is_symlink():
        raise ValueError("Refusing to clean a symlinked Submitit jobs directory")
    removed_bytes = _directory_size(jobs)
    if jobs.exists():
        shutil.rmtree(jobs)
    return {"path": str(jobs), "removed_bytes": removed_bytes}


def _validate_owned_submitit(run_dir: str | Path) -> tuple[Path, Path]:
    root = Path(run_dir).expanduser().resolve()
    submitit_path = root / ".submitit"
    if submitit_path.is_symlink():
        raise ValueError("Refusing to clean a symlinked .submitit directory")
    submitit_dir = submitit_path.resolve()
    if submitit_dir.parent != root:
        raise ValueError("Refusing to clean a Submitit directory outside the run")
    marker_path = submitit_dir / "ownership.json"
    if not marker_path.is_file():
        raise ValueError("Refusing to clean an unowned Submitit directory")
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("managed_by") != _OWNER
        or marker.get("run_dir") != str(root)
        or marker.get("submitit_dir") != str(submitit_dir)
    ):
        raise ValueError("Submitit ownership marker does not match this run")
    return root, submitit_dir


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
