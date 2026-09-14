#!/usr/bin/env python
"""Inspect or repair stale cached-reference labels in a catalyst-screen run."""

from __future__ import annotations

import argparse
from pathlib import Path

import frust as ft


def main() -> None:
    """Run the reference-binding migration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Portable result directory containing manifest.json.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the repair; omission performs a read-only dry run.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Optional explicit backup directory used with --apply.",
    )
    args = parser.parse_args()

    report = ft.screen.repair_reference_bindings(
        args.run_dir,
        apply=args.apply,
        backup_dir=args.backup_dir,
    )
    columns = [
        "file",
        "reference_id",
        "state_id",
        "old_substrate_name",
        "substrate_name",
        "old_catalyst_name",
        "catalyst_name",
    ]
    if report.empty:
        print("No reused reference rows require binding.")
    else:
        print(report[columns].to_string(index=False))
    if args.apply and not report.empty:
        print(f"Applied repair. Backup: {report.attrs.get('backup_dir')}")
    elif not args.apply:
        print("Dry run only. Re-run with --apply to write these changes.")


if __name__ == "__main__":
    main()
