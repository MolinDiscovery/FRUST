import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import same_color
from scipy.stats import linregress

import frust.vis as vis
from frust.vis import MolTo3DGrid, RxnTo3DGrid, plot_energy_profile, plot_mols
from frust.vis.regression import _round_to_sig_figs
from frust.vis.energy_profile.layout import _compute_x_single
from frust.vis.energy_profile.parsing import _parse_entries, _parse_placement


class PlotEnergyProfileTests(unittest.TestCase):
    def test_public_vis_imports_remain_compatible(self):
        self.assertIs(vis.plot_energy_profile, plot_energy_profile)
        self.assertIs(vis.plot_mols, plot_mols)
        self.assertIs(vis.MolTo3DGrid, MolTo3DGrid)
        self.assertIs(vis.RxnTo3DGrid, RxnTo3DGrid)

    def test_side_reaction_parsing_extracts_anchor_rise_and_legend(self):
        entries, seg_ids, anchor, rise, legend = _parse_entries(
            [
                ("A", 0.0),
                "side-rxn@A@0.6#Side pathway",
                ("B", 2.0, "tr"),
            ],
        )

        self.assertEqual(entries, [("A", 0.0, None), ("B", 2.0, "tr")])
        self.assertEqual(seg_ids, [0, 1])
        self.assertEqual(anchor, "A")
        self.assertEqual(rise, 0.6)
        self.assertEqual(legend, "Side pathway")

    def test_label_placement_short_tokens_expand_to_counts(self):
        self.assertEqual(
            _parse_placement("ttr"),
            {"top": 2, "bottom": 0, "left": 0, "right": 1},
        )

    def test_product_layout_uses_configured_offset(self):
        x = _compute_x_single(
            [
                ("Reactant", 0.0, None),
                ("Product", -1.0, None),
                ("Product + Reactant", -2.0, None),
            ],
            product_x_offset=0.5,
        )

        self.assertEqual(x.tolist(), [0.0, 0.75, 1.25])

    def test_main_product_label_keeps_main_color_after_side_reaction(self):
        profiles = {
            "main": [
                ("A", 0.0),
                ("TS", 10.0),
                "side-rxn@A@0.5#Side",
                ("side TS", 8.0),
                ("Product", -1.0, "t"),
                ("Product + A", -2.0),
            ],
            "overlay": [
                ("A", 0.0),
                ("TS", 11.0),
                "side-rxn@A@0.5#Side overlay",
                ("side TS", 9.0),
                ("Product", -3.0, "t"),
                ("Product + A", -4.0),
            ],
        }

        fig, ax = plot_energy_profile(profiles, overlay_alpha=1.0)
        self.addCleanup(lambda: plt.close(fig))

        main_line_color = ax.lines[0].get_color()
        side_line_color = ax.lines[1].get_color()

        text_colors = {
            text.get_text(): text.get_color()
            for text in ax.texts
        }

        self.assertTrue(same_color(text_colors["-1.0"], main_line_color))
        self.assertTrue(same_color(text_colors["-2.0"], side_line_color))


class PlotRegressionOutliersTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_scaled_mode_uses_dataset_fit_to_transform_x_axis(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [1.0, 3.0, 5.0, 7.0],
                "substrate_name": ["a", "b", "c", "d"],
                "rpos": [1, 2, 3, 4],
            }
        )

        with patch("matplotlib.pyplot.show"):
            vis.plot_regression_outliers(
                df,
                x_col="x",
                y_col="y",
                xlabel="x",
                ylabel="y",
                num_outliers=0,
                regression_text="legend",
                scaled=True,
            )

        ax = plt.gcf().axes[0]
        scatter = ax.collections[0].get_offsets()
        expected_x = np.array([1.0, 3.0, 5.0, 7.0])

        np.testing.assert_allclose(scatter[:, 0], expected_x)
        np.testing.assert_allclose(scatter[:, 1], df["y"].to_numpy())
        np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_x)

    def test_scaled_integer_rounds_fit_coefficients_before_transform(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [56.7, 69.1, 81.5, 93.9],
                "substrate_name": ["a", "b", "c", "d"],
                "rpos": [1, 2, 3, 4],
            }
        )

        scale_lr = linregress(df["x"], df["y"])
        full_slope = float(scale_lr.slope)
        full_intercept = float(scale_lr.intercept)
        rounded_slope = _round_to_sig_figs(full_slope, 2)
        rounded_intercept = _round_to_sig_figs(full_intercept, 2)

        with patch("matplotlib.pyplot.show"):
            with redirect_stdout(StringIO()) as buf:
                vis.plot_regression_outliers(
                    df,
                    x_col="x",
                    y_col="y",
                    xlabel="x",
                    ylabel="y",
                    num_outliers=0,
                    regression_text="none",
                    scaled=2,
                )

        ax = plt.gcf().axes[0]
        scatter = ax.collections[0].get_offsets()
        expected_x = rounded_slope * df["x"].to_numpy() + rounded_intercept
        full_precision_x = full_slope * df["x"].to_numpy() + full_intercept

        np.testing.assert_allclose(scatter[:, 0], expected_x)
        self.assertFalse(np.allclose(scatter[:, 0], full_precision_x))
        self.assertIn("Scaling relation:", buf.getvalue())
        self.assertIn("y = 12x + 57", buf.getvalue())

    def test_regression_label_reports_direct_rmsd(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [1.0, 3.0, 5.0, 7.0],
                "substrate_name": ["a", "b", "c", "d"],
                "rpos": [1, 2, 3, 4],
            }
        )
        direct_rmsd = np.sqrt(np.mean((df["y"] - df["x"]) ** 2))

        with patch("matplotlib.pyplot.show"):
            vis.plot_regression_outliers(
                df,
                x_col="x",
                y_col="y",
                xlabel="x",
                ylabel="y",
                num_outliers=0,
                regression_text="plot",
            )

        ax = plt.gcf().axes[0]
        text = "\n".join(item.get_text() for item in ax.texts)

        self.assertIn(f"RMSD$_{{direct}}$={direct_rmsd:.3f} kcal/mol", text)
        self.assertNotIn("RMSD$_{direct}$=0.000 kcal/mol", text)

    def test_regression_drops_missing_xy_pairs_before_metrics(self):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, np.nan, 3.0],
                "y": [1.0, 3.0, 5.0, 7.0],
                "substrate_name": ["a", "b", "c", "d"],
                "rpos": [1, 2, 3, 4],
            }
        )

        with patch("matplotlib.pyplot.show"):
            vis.plot_regression_outliers(
                df,
                x_col="x",
                y_col="y",
                xlabel="x",
                ylabel="y",
                num_outliers=0,
                regression_text="plot",
            )

        ax = plt.gcf().axes[0]
        scatter = ax.collections[0].get_offsets()

        self.assertEqual(len(scatter), 3)
        self.assertFalse(np.isnan(scatter).any())


if __name__ == "__main__":
    unittest.main()
