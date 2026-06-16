from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import frust as ft
from frust.workflows.diagnostics import inspect_failures


class WorkflowDiagnosticsTests(unittest.TestCase):
    def test_inspect_failures_from_dataframe_extracts_failed_stage(self):
        df = pd.DataFrame(
            {
                "ts_type": ["TS2"],
                "substrate_name": ["C6_wb97"],
                "catalyst_name": ["NEt"],
                "rpos": [2],
                "cid": [42],
                "Hess-NT": [True],
                "OptTS-NT": [False],
                "OptTS-error": ["RuntimeError: Orca calculation did not terminate normally"],
                "OptTS-orca.out": [
                    "Geometry cycle 70\nORCA finished by error termination in Startup\n"
                ],
                "Freq-NT": [True],
            }
        )
        df.attrs["frust_workflow"] = {"target": "TS2__substrate_001__catalyst_001__r2"}

        failures = ft.workflows.inspect_failures(df)

        self.assertEqual(len(failures), 1)
        row = failures.iloc[0]
        self.assertEqual(row["target"], "TS2__substrate_001__catalyst_001__r2")
        self.assertEqual(row["ts_type"], "TS2")
        self.assertEqual(row["substrate_name"], "C6_wb97")
        self.assertEqual(row["catalyst_name"], "NEt")
        self.assertEqual(row["rpos"], 2)
        self.assertEqual(row["cid"], 42)
        self.assertEqual(row["failed_stage"], "OptTS")
        self.assertEqual(row["failed_nt_cols"], ["OptTS-NT"])
        self.assertEqual(row["problem"], "failed_stage")
        self.assertIn("Orca calculation", row["error"])
        self.assertIn("Startup", row["backend_hint"])

    def test_inspect_failures_reads_old_collection_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "TS2__substrate_001__catalyst_001__r2"
            target_dir.mkdir()
            failed_file = target_dir / "init.hess.optts.parquet"
            pd.DataFrame(
                {
                    "ts_type": ["TS2"],
                    "substrate_name": ["C6_wb97"],
                    "catalyst_name": ["NEt"],
                    "rpos": [2],
                    "cid": [42],
                    "OptTS-NT": [False],
                    "OptTS-error": ["RuntimeError: Orca calculation did not terminate normally"],
                }
            ).to_parquet(failed_file)
            report_path = root / "collection_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "skipped_files": [str(failed_file)],
                        "missing_files": [str(root / "missing" / "final.parquet")],
                        "errored_files": [],
                        "errors": [],
                    }
                )
            )

            from_report = inspect_failures(report_path)
            from_dir = inspect_failures(root)

        self.assertEqual(list(from_report["problem"]), ["failed_stage", "missing_output"])
        self.assertEqual(list(from_dir["problem"]), ["failed_stage", "missing_output"])
        self.assertEqual(from_report.loc[0, "target"], "TS2__substrate_001__catalyst_001__r2")
        self.assertEqual(from_report.loc[0, "failed_stage"], "OptTS")
        self.assertEqual(from_report.loc[1, "target"], "missing")

    def test_inspect_failures_uses_existing_failure_summary(self):
        report = {
            "failure_summary": [
                {
                    "target": "bad",
                    "file": "bad/final.parquet",
                    "failed_stage": "OptTS",
                    "failed_nt_cols": ["OptTS-NT"],
                    "error": "RuntimeError: failed",
                    "problem": "failed_stage",
                }
            ]
        }

        failures = inspect_failures(report)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures.loc[0, "target"], "bad")
        self.assertEqual(failures.loc[0, "failed_stage"], "OptTS")
        self.assertIsNone(failures.loc[0, "backend_hint"])

    def test_inspect_failures_full_detail_includes_status_values(self):
        df = pd.DataFrame({"OptTS-NT": [pd.NA], "Freq-NT": [False]})

        failures = inspect_failures(df, detail="full")

        self.assertEqual(failures.loc[0, "failed_stage"], "OptTS")
        self.assertEqual(failures.loc[0, "failed_nt_cols"], ["OptTS-NT", "Freq-NT"])
        self.assertEqual(
            failures.loc[0, "failed_nt_values"],
            {"OptTS-NT": None, "Freq-NT": False},
        )


if __name__ == "__main__":
    unittest.main()
