import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from frust.utils.analytics import summarize_ts_vibrations


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


if __name__ == "__main__":
    unittest.main()
