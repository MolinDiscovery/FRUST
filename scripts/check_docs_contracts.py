"""Check documentation statements that are coupled to runtime defaults.

Run from the repository root:

    python scripts/check_docs_contracts.py
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterator
from pathlib import Path

from frust.screen import create_ts_guesses
from frust.tsguess2.specs import BUILTIN_TS_SPECS_V2
from frust.utils.pruning import DEFAULT_PRUNING_OPTIONS
from frust.workflows.factories import screen_ts


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Return one repository text file."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _python_blocks(path: Path) -> Iterator[tuple[int, str]]:
    """Yield the start line and source of fenced Python blocks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        if stripped.startswith("```python") or stripped.startswith("```py"):
            start_line = index + 1
            opening_indent = len(lines[index]) - len(stripped)
            index += 1
            block: list[str] = []
            while index < len(lines):
                current = lines[index]
                current_stripped = current.lstrip()
                current_indent = len(current) - len(current_stripped)
                if current_stripped.startswith("```") and current_indent <= opening_indent:
                    break
                if current.startswith(" " * opening_indent):
                    current = current[opening_indent:]
                block.append(current)
                index += 1
            yield start_line, textwrap.dedent("\n".join(block))
        index += 1


def main() -> None:
    """Validate documentation contracts against current public defaults.

    Raises
    ------
    SystemExit
        If a runtime default and its user-facing documentation disagree.
    """
    failures: list[str] = []

    screen_backend = inspect.signature(create_ts_guesses).parameters["backend"].default
    workflow_backend = inspect.signature(screen_ts).parameters["ts_backend"].default
    if screen_backend != "tsguess2":
        failures.append(f"screen.create_ts_guesses backend is {screen_backend!r}, expected 'tsguess2'")
    if workflow_backend != "tsguess2":
        failures.append(f"workflows.screen_ts backend is {workflow_backend!r}, expected 'tsguess2'")

    spec_ids = {name: spec.spec_id for name, spec in BUILTIN_TS_SPECS_V2.items()}
    stale_ids = {
        name: spec_id
        for name, spec_id in spec_ids.items()
        if "::tsguess2-v2::" not in spec_id
    }
    if stale_ids:
        failures.append(f"tsguess2 specification ids are not method-aware v2 ids: {stale_ids}")

    pruning_rmsd = DEFAULT_PRUNING_OPTIONS["rmsd_max_rmsd"]

    ts_docs = _read("docs/catalyst-screens/ts-guesses.md")
    for required in ("tsguess2", "::tsguess2-v2::", "B_transfer_H", "N_transfer_H"):
        if required not in ts_docs:
            failures.append(f"TS guess documentation is missing {required!r}")
    if "::builtin::" in ts_docs:
        failures.append("TS guess documentation still contains a legacy built-in specification id")

    expected_pruning = f"rmsd_max_rmsd={pruning_rmsd}"
    for relative_path in (
        "docs/catalyst-screens/running.md",
        "docs/tutorials/workflow-method-plans.md",
    ):
        if expected_pruning not in _read(relative_path):
            failures.append(f"{relative_path} does not show the current pruning default")

    for relative_path in (
        "docs/examples/screen.csv",
        "docs/examples/substrates.csv",
        "docs/examples/molecules.csv",
        "docs/examples/raw-dimers.csv",
    ):
        if not (ROOT / relative_path).is_file():
            failures.append(f"documented example input is missing: {relative_path}")

    if (ROOT / "docs/cluster/deferred-stepper-plan.md").exists():
        failures.append("obsolete deferred Stepper proposal is still published under docs/")

    markdown_paths = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    for path in markdown_paths:
        for start_line, source in _python_blocks(path):
            try:
                ast.parse(source)
            except SyntaxError as exc:
                relative = path.relative_to(ROOT)
                failures.append(
                    f"invalid Python block at {relative}:{start_line}: {exc.msg}"
                )

    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"Documentation contract checks failed:\n{formatted}")

    print("Documentation contract checks passed.")


if __name__ == "__main__":
    main()
