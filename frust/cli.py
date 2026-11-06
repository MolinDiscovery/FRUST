"""Command-line entry point for FRUST."""

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run a FRUST command.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments after the command name. Use the process arguments if omitted.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(prog="frust")
    commands = parser.add_subparsers(dest="command", required=True)
    progress = commands.add_parser("progress", help="Follow a tracked Slurm submission")
    progress.add_argument("directory", nargs="?", type=Path, default=Path("."),
                          help="project or results directory (default: current directory)")
    progress.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args(argv)

    from frust.cluster.progress import _show_progress

    try:
        _show_progress(args.directory, once=args.once)
    except (OSError, ValueError) as error:
        progress.exit(1, f"frust progress: {error}\n")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
