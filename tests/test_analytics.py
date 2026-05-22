import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from frust.utils.analytics import summarize_ts_vibrations


class AnalyticsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
