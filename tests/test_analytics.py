import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from frust.utils.analytics import inspect_ts_vibrations, summarize_ts_vibrations


class AnalyticsTests(unittest.TestCase):
    def test_summarize_ts_vibrations_auto_selects_last_non_missing_vibs(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["furan"],
                "rpos": [1],
                "Hess-vibs": [[{"frequency": -500.0}, {"frequency": -50.0}]],
                "Freq-vibs": [[{"frequency": -410.0}, {"frequency": 38.0}]],
                "DFT-solv-vibs": [None],
            }
        )

        out = io.StringIO()
        with redirect_stdout(out):
            summarize_ts_vibrations(df, max_rows=5)

        text = out.getvalue()
        self.assertIn("Using vibration column: Freq-vibs", text)
        self.assertIn("True TSs : 1", text)
        self.assertIn("-410.00", text)

    def test_summarize_ts_vibrations_auto_selects_last_vibs_by_column_order(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["furan"],
                "rpos": [1],
                "Freq-vibs": [[{"frequency": -410.0}, {"frequency": 38.0}]],
                "FinalCheck-vibs": [[{"frequency": -380.0}, {"frequency": 41.0}]],
            }
        )

        out = io.StringIO()
        with redirect_stdout(out):
            summarize_ts_vibrations(df, max_rows=5)

        text = out.getvalue()
        self.assertIn("Using vibration column: FinalCheck-vibs", text)
        self.assertIn("-380.00", text)

    def test_summarize_ts_vibrations_reports_missing_vibrations(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["ethanol", "methanol"],
                "rpos": [1, 2],
                "stage-vibs": [
                    [{"frequency": -520.0}, {"frequency": 42.0}],
                    None,
                ],
            }
        )

        out = io.StringIO()
        with redirect_stdout(out):
            summarize_ts_vibrations(df, col="stage-vibs", max_rows=5)

        text = out.getvalue()
        self.assertIn("Missing", text)
        self.assertIn("True TSs : 1", text)
        self.assertIn("Non-TSs  : 0", text)
        self.assertIn("Missing vibrations : 1", text)

    def test_summarize_ts_vibrations_missing_explicit_column_lists_available(self):
        df = pd.DataFrame({"Freq-vibs": [[{"frequency": -100.0}]]})

        with self.assertRaisesRegex(ValueError, "Available vibration columns.*Freq-vibs"):
            summarize_ts_vibrations(df, col="old-default-vibs")

    def test_summarize_ts_vibrations_errors_when_no_vibration_columns(self):
        df = pd.DataFrame({"substrate_name": ["furan"]})

        with self.assertRaisesRegex(ValueError, "No vibration columns found"):
            summarize_ts_vibrations(df)

    def test_inspect_ts_vibrations_returns_compact_dataframe_report(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["furan", "furan", "anilide", "pyrrole"],
                "structure_type": ["TS1", "TS1", "TS2", "TS3"],
                "rpos": [1, 2, 3, 4],
                "cid": [7, 8, 9, 10],
                "Freq-vibs": [
                    [{"frequency": -410.0}, {"frequency": 38.0}, {"frequency": 61.0}],
                    [{"frequency": 18.0}, {"frequency": 45.0}],
                    [{"frequency": -300.0}, {"frequency": -42.0}, {"frequency": 29.0}],
                    None,
                ],
            },
            index=[10, 20, 30, 40],
        )

        report = inspect_ts_vibrations(df)

        self.assertNotIn("source", report.columns)
        self.assertEqual(list(report["row"]), [0, 1, 2, 3])
        self.assertEqual(
            list(report["status"]),
            ["first_order", "minimum", "higher_order", "missing_vibs"],
        )
        self.assertEqual(report.loc[0, "imag_freqs"], "-410.00")
        self.assertEqual(report.loc[1, "imag_freqs"], "None")
        self.assertEqual(report.loc[2, "n_imag"], 2)
        self.assertEqual(report.loc[3, "low_pos_freqs"], "Missing")
        self.assertEqual(
            report.attrs["summary"],
            {
                "first_order": 1,
                "minimum": 1,
                "higher_order": 1,
                "missing_vibs": 1,
            },
        )
        self.assertEqual(report.attrs["vibration_columns"], {"df": "Freq-vibs"})

    def test_inspect_ts_vibrations_review_filter_includes_flagged_first_order(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["good", "weak", "minimum"],
                "rpos": [1, 2, 3],
                "Freq-vibs": [
                    [{"frequency": -250.0}, {"frequency": 40.0}],
                    [{"frequency": -35.0}, {"frequency": 20.0}],
                    [{"frequency": 15.0}, {"frequency": 42.0}],
                ],
            }
        )

        report = inspect_ts_vibrations(df, status="review")

        self.assertEqual(list(report["substrate_name"]), ["weak", "minimum"])
        self.assertEqual(list(report["status"]), ["first_order", "minimum"])
        self.assertIn("weak_imag", report.iloc[0]["flags"])
        self.assertIn("very_low_pos", report.iloc[0]["flags"])
        self.assertEqual(report.attrs["n_review_rows"], 2)

    def test_inspect_ts_vibrations_accepts_dict_inputs_and_selects_columns_per_df(self):
        ts1 = pd.DataFrame(
            {
                "substrate_name": ["furan"],
                "Hess-vibs": [[{"frequency": -500.0}]],
                "Freq-vibs": [[{"frequency": -120.0}, {"frequency": 31.0}]],
            }
        )
        ts2 = pd.DataFrame(
            {
                "substrate_name": ["furan"],
                "Freq-vibs": [[{"frequency": 22.0}, {"frequency": 38.0}]],
                "FinalCheck-vibs": [[{"frequency": -90.0}, {"frequency": 41.0}]],
            }
        )

        report = inspect_ts_vibrations({"TS1": ts1, "TS2": ts2})

        self.assertEqual(list(report["source"]), ["TS1", "TS2"])
        self.assertEqual(list(report["status"]), ["first_order", "first_order"])
        self.assertEqual(
            report.attrs["vibration_columns"],
            {"TS1": "Freq-vibs", "TS2": "FinalCheck-vibs"},
        )

    def test_inspect_ts_vibrations_status_filters_and_validation(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["a", "b", "c"],
                "Freq-vibs": [
                    [{"frequency": -100.0}, {"frequency": 30.0}],
                    [{"frequency": 30.0}],
                    [{"frequency": -100.0}, {"frequency": -20.0}, {"frequency": 30.0}],
                ],
            }
        )

        minimum = inspect_ts_vibrations(df, status="minimum")
        higher_order = inspect_ts_vibrations(df, status="higher_order")
        problems = inspect_ts_vibrations(df, status="problems")

        self.assertEqual(list(minimum["substrate_name"]), ["b"])
        self.assertEqual(list(higher_order["substrate_name"]), ["c"])
        self.assertEqual(list(problems["substrate_name"]), ["b", "c"])
        with self.assertRaisesRegex(ValueError, "status"):
            inspect_ts_vibrations(df, status="bad")

    def test_inspect_ts_vibrations_rejects_bad_inputs(self):
        with self.assertRaisesRegex(TypeError, "DataFrame"):
            inspect_ts_vibrations({"bad": object()})


if __name__ == "__main__":
    unittest.main()
