import inspect
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import same_color
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import linregress
from tooltoad.scene3d import Py3DmolGridRenderer as TooltoadPy3DmolGridRenderer

import frust.vis as vis
import frust.vis.conformers as conformers_module
import frust.vis.molecules as molecules_module
import frust.vis.structure_comparison as structure_comparison_module
from frust.vis import (
    MolTo3DGrid,
    RxnTo3DGrid,
    conformer_ensemble_grid_scene_from_dataframe,
    conformer_ensemble_scene_from_dataframe,
    plot_conformers,
    plot_energy_profile,
    plot_mols,
)
from frust.vis import (
    ArrowOverlay,
    GridScene,
    MoleculeModel,
    SceneCell,
    ScreenLabelOverlay,
)
from frust.vis.regression import _round_to_sig_figs
from frust.vis.vibrations import _select_vibration_column, _select_vibration_coords_column
from frust.vis.energy_profile.layout import _compute_x_single
from frust.vis.energy_profile.parsing import _parse_entries, _parse_placement
from frust.vis.scenes import (
    molecule_scene_from_dataframe,
    select_vibration_column,
    select_vibration_coords_column,
    ts_guess_scene_from_dataframe,
    vibration_scene_from_dataframe,
)
from frust.vis.structure_comparison import (
    PROBE_STYLE,
    REFERENCE_STYLE,
    compare_rmsd,
)


class PlotVibsSelectionTests(unittest.TestCase):
    def test_selects_last_non_missing_vibs_and_preceding_coords(self):
        df = pd.DataFrame(
            {
                "atoms": [["H", "H"]],
                "coords_embedded": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]],
                "DFT-pre-Opt-oc": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]]],
                "Hess-vibs": [[{"frequency": -200.0}]],
                "OptTS-oc": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]],
                "Freq-vibs": [[{"frequency": -100.0}]],
                "DFT-solv-EE": [-1.0],
            }
        )

        vibs_col = _select_vibration_column(df)
        coords_col = _select_vibration_coords_column(df, vibs_col)

        self.assertEqual(vibs_col, "Freq-vibs")
        self.assertEqual(coords_col, "OptTS-oc")

    def test_selects_latest_named_vibs_by_dataframe_order(self):
        df = pd.DataFrame(
            {
                "atoms": [["H", "H"]],
                "OptTS-oc": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]],
                "Freq-vibs": [[{"frequency": -100.0}]],
                "FinalCheck-vibs": [[{"frequency": -90.0}]],
            }
        )

        self.assertEqual(_select_vibration_column(df), "FinalCheck-vibs")

    def test_ignores_trailing_missing_vibration_column(self):
        df = pd.DataFrame(
            {
                "atoms": [["H", "H"]],
                "OptTS-oc": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]],
                "Freq-vibs": [[{"frequency": -100.0}]],
                "DFT-solv-vibs": [None],
            }
        )

        self.assertEqual(_select_vibration_column(df), "Freq-vibs")

    def test_prefers_matching_coordinate_column_when_available(self):
        df = pd.DataFrame(
            {
                "coords_embedded": [[[0.0, 0.0, 0.0]]],
                "Freq-oc": [[[1.0, 0.0, 0.0]]],
                "Freq-vibs": [[{"frequency": -100.0}]],
                "Later-oc": [[[2.0, 0.0, 0.0]]],
            }
        )

        self.assertEqual(_select_vibration_coords_column(df, "Freq-vibs"), "Freq-oc")

    def test_missing_custom_coordinate_column_reports_available(self):
        df = pd.DataFrame(
            {
                "OptTS-oc": [[[0.0, 0.0, 0.0]]],
                "Freq-vibs": [[{"frequency": -100.0}]],
            }
        )

        with self.assertRaisesRegex(KeyError, "Available coordinate columns.*OptTS-oc"):
            _select_vibration_coords_column(
                df,
                "Freq-vibs",
                custom_coords_col_name="missing-oc",
            )


class PlotEnergyProfileTests(unittest.TestCase):
    def test_public_vis_imports_remain_compatible(self):
        self.assertIs(vis.plot_energy_profile, plot_energy_profile)
        self.assertIs(vis.plot_mols, plot_mols)
        self.assertIs(vis.plot_conformers, plot_conformers)
        self.assertIs(vis.MolTo3DGrid, MolTo3DGrid)
        self.assertIs(vis.RxnTo3DGrid, RxnTo3DGrid)
        self.assertTrue(callable(vis.reaction_scene_cells))
        self.assertIs(vis.ArrowOverlay, ArrowOverlay)
        self.assertIs(vis.ScreenLabelOverlay, ScreenLabelOverlay)
        self.assertTrue(callable(vis.show_scene))
        self.assertTrue(callable(vis.show_conformer_ensemble_scene))
        self.assertTrue(callable(vis.molecule_scene_from_dataframe))
        self.assertTrue(callable(vis.conformer_ensemble_scene_from_dataframe))
        self.assertTrue(callable(vis.vibration_scene_from_dataframe))
        self.assertTrue(callable(vis.ts_guess_scene))

    def test_mol_to_3d_grid_accepts_measurement_decimals(self):
        signature = inspect.signature(MolTo3DGrid)

        self.assertIn("decimals_of_measure", signature.parameters)
        self.assertEqual(signature.parameters["decimals_of_measure"].default, 3)

    def test_mol_to_3d_grid_applies_distance_and_angle_measurement_decimals(self):
        class FakeRenderer:
            _CLICK_HANDLER = TooltoadPy3DmolGridRenderer._CLICK_HANDLER
            instances = []

            def __init__(self, scene):
                self.scene = scene
                self._CLICK_HANDLER = self.__class__._CLICK_HANDLER
                self.show_calls = 0
                FakeRenderer.instances.append(self)

            def show(self):
                self.show_calls += 1
                return "viewer"

            def write_html(self, path):
                self.export_path = path

        with patch.object(molecules_module, "Py3DmolGridRenderer", FakeRenderer):
            result = MolTo3DGrid("CC", decimals_of_measure=1)

        self.assertIsNone(result)
        self.assertEqual(FakeRenderer.instances[-1].show_calls, 1)
        click_handler = FakeRenderer.instances[-1]._CLICK_HANDLER
        self.assertEqual(click_handler.count(".toFixed(1)"), 2)
        self.assertNotIn(".toFixed(3)", click_handler)
        self.assertNotIn(".toFixed(2)", click_handler)

    def test_mol_to_3d_grid_rejects_invalid_measurement_decimals(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            MolTo3DGrid("CC", decimals_of_measure=-1)

        with self.assertRaisesRegex(TypeError, "non-negative integer"):
            MolTo3DGrid("CC", decimals_of_measure=1.5)

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

    def test_overlay_legend_uses_profile_colors_for_energy_labels(self):
        profiles = {
            "ref": [("A", 0.0), ("TS", 1.0)],
            "first": [("A", 0.0), ("TS", 2.0)],
            "second": [("A", 0.0), ("TS", 3.0)],
            "third": [("A", 0.0), ("TS", 4.0)],
        }

        fig, ax = plot_energy_profile(profiles, overlay_alpha=1.0)
        self.addCleanup(lambda: plt.close(fig))

        legend = ax.get_legend()
        legend_colors = {
            label.get_text(): handle.get_color()
            for label, handle in zip(legend.get_texts(), legend.legend_handles)
        }
        text_colors = {text.get_text(): text.get_color() for text in ax.texts}

        expected = {
            "ref": ("1.0", "C0"),
            "first": ("2.0", "C1"),
            "second": ("3.0", "C2"),
            "third": ("4.0", "C3"),
        }
        for profile_name, (energy_label, expected_color) in expected.items():
            self.assertTrue(same_color(legend_colors[profile_name], expected_color))
            self.assertTrue(
                same_color(legend_colors[profile_name], text_colors[energy_label])
            )

    def test_same_energy_show_keeps_matching_overlay_labels(self):
        profiles = {
            "ref": [("A", 0.0), ("TS", 1.0)],
            "overlay": [("A", 0.0), ("TS", 1.0, "r")],
        }

        fig, ax = plot_energy_profile(
            profiles,
            overlay_alpha=1.0,
            same_energy_mode="show",
        )
        self.addCleanup(lambda: plt.close(fig))

        matching_labels = [text for text in ax.texts if text.get_text() == "1.0"]

        self.assertEqual(len(matching_labels), 2)
        self.assertTrue(same_color(matching_labels[0].get_color(), "C0"))
        self.assertTrue(same_color(matching_labels[1].get_color(), "C1"))

    def test_same_energy_hide_suppresses_matching_overlay_labels(self):
        profiles = {
            "ref": [("A", 0.0), ("TS", 1.0)],
            "overlay": [("A", 0.0), ("TS", 1.0, "r")],
        }

        fig, ax = plot_energy_profile(
            profiles,
            overlay_alpha=1.0,
            same_energy_mode="hide",
        )
        self.addCleanup(lambda: plt.close(fig))

        matching_labels = [text for text in ax.texts if text.get_text() == "1.0"]

        self.assertEqual(len(matching_labels), 1)
        self.assertTrue(same_color(matching_labels[0].get_color(), "C0"))

    def test_product_reference_adds_catalyst_relative_product_energy(self):
        profiles = {
            "DFT": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("Product", 3.3, "b"),
                ("Product + int2", -10.3),
            ],
            "Constrained-xTB/SP": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("Product", 3.3, "b"),
                ("Product + int2", -10.3),
            ],
        }

        fig, ax = plot_energy_profile(
            profiles,
            product_reference="Cat",
            same_energy_mode="hide",
            overlay_alpha=1.0,
        )
        self.addCleanup(lambda: plt.close(fig))

        labels = [text.get_text() for text in ax.texts]

        self.assertEqual(labels.count("3.3\n(-2.3)"), 1)
        self.assertEqual(labels.count("-10.3"), 1)

    def test_product_reference_works_for_single_profile(self):
        states = [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("Product", 3.3),
        ]

        fig, ax = plot_energy_profile(states, product_reference="Cat")
        self.addCleanup(lambda: plt.close(fig))

        product_annotation = next(
            text
            for text in ax.texts
            if text.get_text() == "Product\n3.3\n(-2.3)"
        )

        self.assertEqual(product_annotation.xy[1], 3.3)

    def test_product_reference_connector_draws_dotted_line(self):
        states = [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("Product", 3.3),
        ]

        fig, ax = plot_energy_profile(
            states,
            product_reference=("Cat", "connector"),
        )
        self.addCleanup(lambda: plt.close(fig))

        connector = next(line for line in ax.lines if line.get_linestyle() == ":")
        relative_annotation = next(
            text
            for text in ax.texts
            if text.get_text() == "(-2.3)" and text.xy[1] == -2.3
        )

        self.assertEqual(connector.get_xdata().tolist(), [2.0, 2.0])
        self.assertEqual(connector.get_ydata().tolist(), [3.3, -2.3])
        self.assertEqual(relative_annotation.xy, (2.0, -2.3))
        self.assertIn("Product\n3.3", [text.get_text() for text in ax.texts])

    def test_product_reference_connector_hides_matching_overlay_line(self):
        profiles = {
            "first": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("Product", 3.3),
            ],
            "second": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("Product", 3.3),
            ],
        }

        fig, ax = plot_energy_profile(
            profiles,
            product_reference=("Cat", "connector"),
            same_energy_mode="hide",
            overlay_alpha=1.0,
        )
        self.addCleanup(lambda: plt.close(fig))

        connectors = [line for line in ax.lines if line.get_linestyle() == ":"]
        relative_labels = [text for text in ax.texts if text.get_text() == "(-2.3)"]

        self.assertEqual(len(connectors), 1)
        self.assertEqual(len(relative_labels), 1)

    def test_product_reference_rejects_other_display_names(self):
        states = [("Dimer", 0.0), ("Cat", 5.6), ("Product", 3.3)]

        for display in ("Show", "Compact", "Expanded"):
            with self.subTest(display=display), self.assertRaisesRegex(
                ValueError,
                "display must be 'compact' or 'connector'",
            ):
                plot_energy_profile(
                    states,
                    product_reference=("Cat", display),
                )

    def test_product_reference_keeps_distinct_overlay_reference_values(self):
        profiles = {
            "first": [
                ("Dimer", 0.0),
                ("Cat", 5.0),
                ("Product", 3.0),
            ],
            "second": [
                ("Dimer", 0.0),
                ("Cat", 6.0),
                ("Product", 3.0),
            ],
        }

        fig, ax = plot_energy_profile(
            profiles,
            product_reference="Cat",
            same_energy_mode="hide",
            overlay_alpha=1.0,
        )
        self.addCleanup(lambda: plt.close(fig))

        labels = [text.get_text() for text in ax.texts]

        self.assertIn("3.0\n(-2.0)", labels)
        self.assertIn("3.0\n(-3.0)", labels)

    def test_product_reference_requires_named_state_in_each_profile(self):
        with self.assertRaisesRegex(
            ValueError,
            "product_reference='Cat' was not found",
        ):
            plot_energy_profile(
                [("Dimer", 0.0), ("Product", -1.0)],
                product_reference="Cat",
            )

    def test_main_to_product_marker_draws_local_fraction_connector(self):
        states = [
            ("Dimer", 0.0),
            ("Cat", 5.6),
            ("TS4", 14.8),
            "main-to-product@0.8",
            ("Product", 3.3),
        ]

        fig, ax = plot_energy_profile(
            states,
            main_to_product_drop_frac=0.2,
            show_state_labels=True,
        )
        self.addCleanup(lambda: plt.close(fig))

        solid_line = next(line for line in ax.lines if line.get_linestyle() == "-")
        connector = next(line for line in ax.lines if line.get_linestyle() == ":")
        connector_x = np.asarray(connector.get_xdata(), dtype=float)
        connector_y = np.asarray(connector.get_ydata(), dtype=float)

        self.assertEqual(float(np.max(solid_line.get_xdata())), 2.0)
        self.assertTrue(np.allclose(connector_y[connector_x <= 2.79], 14.8))
        self.assertEqual(connector_x[-1], 3.0)
        self.assertAlmostEqual(connector_y[-1], 3.3)
        self.assertEqual(
            [label.get_text() for label in ax.get_xticklabels()],
            ["Dimer", "Cat", "TS4", "Product"],
        )
        self.assertIsNone(ax.get_legend())

    def test_main_to_product_marker_supports_overlay_without_side_path(self):
        profiles = {
            "DFT": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("TS4", 20.8),
                "side-rxn@Cat@0.8#Bisarylation",
                ("TS5", 47.6),
                ("Product", 3.3),
                ("Product + int2", -10.3),
            ],
            "Reference": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("TS4", 14.8),
                "main-to-product@0.8",
                ("Product", 3.3),
            ],
        }

        fig, ax = plot_energy_profile(
            profiles,
            overlay_alpha=1.0,
            overlay_colors={"Reference": "#C6C5C5"},
        )
        self.addCleanup(lambda: plt.close(fig))

        grey_connectors = [
            line
            for line in ax.lines
            if line.get_linestyle() == ":"
            and same_color(line.get_color(), "#C6C5C5")
        ]
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]

        self.assertEqual(len(grey_connectors), 1)
        self.assertEqual(legend_labels.count("Bisarylation"), 1)

    def test_main_to_product_marker_requires_final_product(self):
        with self.assertRaisesRegex(ValueError, "followed directly by a product"):
            plot_energy_profile(
                [
                    ("Dimer", 0.0),
                    ("TS4", 14.8),
                    "main-to-product@0.8",
                    ("int4", 3.3),
                ]
            )

    def test_no_product_marker_ends_overlay_without_product_connector(self):
        profiles = {
            "DFT": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("TS4", 20.8),
                ("Product", 3.3),
            ],
            "Reference": [
                ("Dimer", 0.0),
                ("Cat", 5.6),
                ("TS4", 14.8),
                "no-product",
            ],
        }

        fig, ax = plot_energy_profile(
            profiles,
            overlay_alpha=1.0,
            overlay_colors={"Reference": "#C6C5C5"},
            product_reference=("Cat", "compact"),
        )
        self.addCleanup(lambda: plt.close(fig))

        grey_lines = [
            line for line in ax.lines if same_color(line.get_color(), "#C6C5C5")
        ]
        grey_text = [
            text.get_text()
            for text in ax.texts
            if same_color(text.get_color(), "#C6C5C5")
        ]

        self.assertEqual(len(grey_lines), 1)
        self.assertEqual(grey_lines[0].get_linestyle(), "-")
        self.assertEqual(float(np.max(grey_lines[0].get_xdata())), 2.0)
        self.assertFalse(any(text.startswith("3.3") for text in grey_text))
        self.assertFalse(any("(" in text for text in grey_text))

    def test_no_product_marker_must_be_final_and_have_no_product(self):
        with self.assertRaisesRegex(ValueError, "must be the final entry"):
            plot_energy_profile(
                [("Dimer", 0.0), "no-product", ("TS4", 14.8)]
            )

        with self.assertRaisesRegex(ValueError, "already contains a product"):
            plot_energy_profile(
                [("Dimer", 0.0), ("Product", 3.3), "no-product"]
            )

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


class SceneAdapterTests(unittest.TestCase):
    def small_molecule_df(self):
        return pd.DataFrame(
            {
                "substrate_name": ["furan", "pyrrole"],
                "rpos": [0, 1],
                "atoms": [["C", "H"], ["N", "H"]],
                "coords_embedded": [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                "connectivity_bonds": [[(0, 1)], [(0, 1)]],
            }
        )

    def small_vib_df(self):
        df = self.small_molecule_df()
        df["OptTS-oc"] = df["coords_embedded"]
        df["Freq-vibs"] = [
            [{"frequency": -427.1, "mode": [[0, 0, 0.1], [0, 0, -0.1]]}],
            [{"frequency": -388.4, "mode": [[0, 0.1, 0], [0, -0.1, 0]]}],
        ]
        return df

    def test_molecule_scene_from_dataframe_uses_atoms_coords_and_bonds(self):
        scene = molecule_scene_from_dataframe(
            self.small_molecule_df(),
            row_indices=[0],
            coord_indices=slice(-1, None),
        )

        self.assertEqual(len(scene.cells), 1)
        model = scene.cells[0].models[0]
        self.assertEqual(model.atoms, ["C", "H"])
        self.assertEqual(model.bonds, [(0, 1)])
        self.assertIn("furan", scene.cells[0].title)

    def test_molecule_scene_preserves_numpy_array_connectivity(self):
        df = self.small_molecule_df()
        df.at[0, "connectivity_bonds"] = np.array([[0, 1]], dtype=object)

        scene = molecule_scene_from_dataframe(
            df,
            row_indices=[0],
            coord_indices=slice(-1, None),
        )

        self.assertEqual(scene.cells[0].models[0].bonds, [(0, 1)])

    def test_vibration_scene_from_dataframe_supports_all_rows_and_columns(self):
        scene = vibration_scene_from_dataframe(
            self.small_vib_df(),
            row_indices="all",
            max_rows=2,
            columns=2,
            vId=0,
        )

        self.assertEqual(len(scene.cells), 2)
        self.assertEqual(scene.columns, 2)
        self.assertEqual(scene.cells[0].animations[0].frequency, -427.1)
        self.assertEqual(scene.background_color, ("blue", 0.1))
        self.assertFalse(scene.transparent)
        self.assertEqual(scene.cell_size, (400, 400))
        self.assertEqual(scene.cells[0].models[0].style["sphere"]["radius"], 0.3)

    def test_vibration_scene_from_dataframe_defaults_to_all_rows(self):
        scene = vibration_scene_from_dataframe(self.small_vib_df(), vId=0)

        self.assertEqual(len(scene.cells), 2)
        self.assertEqual([cell.animations[0].frequency for cell in scene.cells], [-427.1, -388.4])

    def test_vibration_scene_from_dataframe_explicit_row_index_is_single_row(self):
        scene = vibration_scene_from_dataframe(self.small_vib_df(), row_index=1, vId=0)

        self.assertEqual(len(scene.cells), 1)
        self.assertEqual(scene.cells[0].animations[0].frequency, -388.4)

    def test_vibration_scene_preserves_numpy_array_connectivity(self):
        df = self.small_vib_df()
        df.at[0, "connectivity_bonds"] = np.array([[0, 1]], dtype=object)

        scene = vibration_scene_from_dataframe(
            df,
            row_indices=[0],
            vId=0,
        )

        self.assertEqual(scene.cells[0].models[0].bonds, [(0, 1)])

    def test_vibration_column_selection_prefers_latest_non_missing(self):
        df = self.small_vib_df()
        df["Later-vibs"] = [None, None]

        self.assertEqual(select_vibration_column(df), "Freq-vibs")
        self.assertEqual(select_vibration_coords_column(df, "Freq-vibs"), "OptTS-oc")

    def test_plot_mols_renders_scene(self):
        with patch("frust.vis.molecules.Py3DmolGridRenderer.show") as show:
            viewer = plot_mols(self.small_molecule_df(), row_indices=[0])

        show.assert_called_once()
        self.assertIsNone(viewer)

    def test_plot_vibs_returns_viewer_without_explicit_show(self):
        with (
            patch("frust.vis.vibrations.Py3DmolGridRenderer.render", return_value="viewer") as render,
            patch("frust.vis.vibrations.Py3DmolGridRenderer.show") as show,
        ):
            viewer = vis.plot_vibs(
                self.small_vib_df(),
                row_indices="all",
                max_rows=2,
                columns=2,
                vId=0,
            )

        render.assert_called_once()
        show.assert_not_called()
        self.assertEqual(viewer, "viewer")

    def test_show_scene_returns_viewer_without_explicit_show(self):
        scene = molecule_scene_from_dataframe(self.small_molecule_df(), row_indices=[0])
        with (
            patch("frust.vis.scenes.Py3DmolGridRenderer.render", return_value="viewer") as render,
            patch("frust.vis.scenes.Py3DmolGridRenderer.show") as show,
        ):
            viewer = vis.show_scene(scene)

        render.assert_called_once()
        show.assert_not_called()
        self.assertEqual(viewer, "viewer")

    def test_manual_scene_accepts_dataframe_numpy_bonds(self):
        row = self.small_molecule_df().iloc[0].copy()
        row["connectivity_bonds"] = np.array([[0, 1]], dtype=object)
        scene = GridScene(
            cells=[
                SceneCell(
                    title="manual",
                    models=[
                        MoleculeModel(
                            atoms=row["atoms"],
                            coords=row["coords_embedded"],
                            bonds=row["connectivity_bonds"],
                        )
                    ],
                )
            ]
        )

        with patch("frust.vis.scenes.Py3DmolGridRenderer.render", return_value="viewer"):
            viewer = vis.show_scene(scene)

        self.assertEqual(scene.cells[0].models[0].bonds, [(0, 1)])
        self.assertEqual(viewer, "viewer")

    def test_ts_guess_scene_adds_role_distance_and_angle_overlays(self):
        df = self.small_molecule_df().iloc[[0]].copy()
        df["atoms"] = [["C", "H", "B"]]
        df["coords_embedded"] = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        ]
        df["connectivity_bonds"] = [[(0, 1), (1, 2)]]
        df["constraint_roles"] = [{"cat_B": 0, "transfer_H": 1, "pin_B": 2}]
        df["constraint_spec"] = [
            [
                {"kind": "distance", "roles": ["cat_B", "transfer_H"], "value": 1.2},
                {"kind": "angle", "roles": ["cat_B", "transfer_H", "pin_B"], "value": 89.48},
            ]
        ]

        scene = ts_guess_scene_from_dataframe(
            df,
            row_indices=[0],
            show_roles=True,
            show_constraint_distances=True,
            show_constraint_angles=True,
        )

        overlay_types = {type(overlay).__name__ for overlay in scene.cells[0].overlays}
        self.assertIn("AtomLabel", overlay_types)
        self.assertIn("AtomHighlight", overlay_types)
        self.assertIn("DistanceOverlay", overlay_types)
        self.assertIn("AngleOverlay", overlay_types)

        angle_overlay = next(
            overlay
            for overlay in scene.cells[0].overlays
            if type(overlay).__name__ == "AngleOverlay"
        )
        self.assertEqual(
            (angle_overlay.atom1, angle_overlay.atom2, angle_overlay.atom3),
            (0, 1, 2),
        )
        self.assertEqual(angle_overlay.label, "89.5 deg")

    def test_ts_guess_scene_can_show_angles_without_distances(self):
        df = self.small_molecule_df().iloc[[0]].copy()
        df["atoms"] = [["C", "H", "B"]]
        df["coords_embedded"] = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        ]
        df["connectivity_bonds"] = [[(0, 1), (1, 2)]]
        df["constraint_roles"] = [{"cat_B": 0, "transfer_H": 1, "pin_B": 2}]
        df["constraint_spec"] = [
            [
                {"kind": "distance", "roles": ["cat_B", "transfer_H"], "value": 1.2},
                {"kind": "angle", "roles": ["cat_B", "transfer_H", "pin_B"], "value": 89.48},
            ]
        ]

        scene = ts_guess_scene_from_dataframe(
            df,
            row_indices=[0],
            show_roles=False,
            show_constraint_distances=False,
            show_constraint_angles=True,
        )

        overlay_types = {type(overlay).__name__ for overlay in scene.cells[0].overlays}
        self.assertEqual(overlay_types, {"AngleOverlay"})
class ConformerEnsembleSceneTests(unittest.TestCase):
    @staticmethod
    def _transform(coords):
        rot = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        shift = np.array([4.0, -2.0, 1.5])
        return np.asarray(coords, dtype=float) @ rot.T + shift

    def conformer_df(self):
        base = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        rows = []
        energies = [0.0, 2.0, 6.0, 1.0]
        for cid, energy in enumerate(energies):
            coords = base.copy()
            coords[3] += np.array([0.0, 0.15 * cid, 0.0])
            coords[4] += np.array([0.0, 0.15 * cid, 0.2 * cid])
            if cid == 1:
                coords = self._transform(coords)
            rows.append(
                {
                    "system_name": "sys",
                    "substrate_name": "anisole",
                    "catalyst_name": "cat",
                    "structure_type": "TS1",
                    "molecule_role": "ts",
                    "rpos": 4,
                    "cid": cid,
                    "atoms": ["B", "N", "C", "H", "O"],
                    "connectivity_bonds": [(0, 1), (1, 2), (2, 3), (3, 4)],
                    "coords_embedded": coords.tolist(),
                    "constraint_roles": {"cat_B": 0, "cat_N": 1, "substrate_C": 2},
                    "energy_uff": energy,
                }
            )
        return pd.DataFrame(rows)

    def multi_rpos_conformer_df(self):
        frames = []
        for offset, rpos in enumerate([3, 4, 5]):
            frame = self.conformer_df()
            frame["rpos"] = rpos
            frame["structure_id"] = f"TS1:sys:r{rpos}"
            frame["custom_name"] = f"TS1(sys_rpos({rpos}))"
            frame["coords_embedded"] = frame["coords_embedded"].map(
                lambda coords, offset=offset: (np.asarray(coords, dtype=float) + offset).tolist()
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_conformer_scene_filters_and_renders_core_once_with_mobile_cloud(self):
        scene = conformer_ensemble_scene_from_dataframe(
            self.conformer_df(),
            row_index=0,
            mode="representatives+cloud",
            top_n=3,
            energy_window_kcal=3.0,
        )

        self.assertEqual(len(scene.cells), 1)
        models = scene.cells[0].models
        self.assertEqual(models[0].atoms, ["B", "N", "C"])
        self.assertEqual(models[0].bonds, [(0, 1), (1, 2)])
        mobile_models = [model for model in models if model.atoms == ["H", "O"]]
        connector_models = [model for model in models if model.atoms == ["C", "H"]]
        self.assertEqual(len(mobile_models), 4)
        self.assertEqual(len(connector_models), 4)
        cloud_models = [
            model
            for model in mobile_models
            if model.style["stick"].get("opacity") < 0.9
        ]
        representative_models = [
            model
            for model in mobile_models
            if model.style["stick"].get("opacity") >= 0.9
        ]
        self.assertEqual(len(cloud_models), 3)
        self.assertEqual(len(representative_models), 1)
        for model in cloud_models:
            self.assertEqual(model.bonds, [(0, 1)])
            self.assertIn("sphere", model.style)
            self.assertGreaterEqual(model.style["stick"]["opacity"], 0.5)
            self.assertAlmostEqual(model.style["stick"]["radius"], 0.075)
        for model in connector_models:
            self.assertEqual(model.bonds, [(0, 1)])
            self.assertNotIn("sphere", model.style)
            self.assertLess(model.style["stick"]["radius"], models[-2].style["stick"]["radius"])
        self.assertEqual(models[0].style["stick"]["color"], "black")
        self.assertIn("3 conformers", scene.cells[0].title)

    def test_conformer_grid_scene_uses_one_cell_per_rpos_by_default(self):
        scene = conformer_ensemble_grid_scene_from_dataframe(
            self.multi_rpos_conformer_df(),
            mode="representatives+cloud",
            top_n=2,
        )

        self.assertEqual(len(scene.cells), 3)
        self.assertEqual(scene.columns, 3)
        self.assertIn("anisole r3", scene.cells[0].title)
        self.assertIn("anisole r4", scene.cells[1].title)
        self.assertIn("anisole r5", scene.cells[2].title)
        for cell in scene.cells:
            self.assertIn("2 conformers", cell.title)

    def test_conformer_grid_scene_can_select_families_by_row_indices(self):
        scene = conformer_ensemble_grid_scene_from_dataframe(
            self.multi_rpos_conformer_df(),
            row_indices=[0, 8],
            mode="single",
        )

        self.assertEqual(len(scene.cells), 2)
        self.assertEqual(scene.columns, 2)
        self.assertIn("anisole r3", scene.cells[0].title)
        self.assertIn("anisole r5", scene.cells[1].title)

    def test_conformer_grid_scene_accepts_parquet_object_array_coordinates(self):
        df = self.multi_rpos_conformer_df()

        def object_array(coords):
            rows = np.empty(len(coords), dtype=object)
            rows[:] = [np.asarray(row, dtype=float) for row in coords]
            return rows

        df["coords_embedded"] = df["coords_embedded"].map(object_array)

        scene = conformer_ensemble_grid_scene_from_dataframe(df, mode="single")

        self.assertEqual(len(scene.cells), 3)
        self.assertIn("anisole r4", scene.cells[1].title)

    def test_conformer_alignment_uses_core_atoms(self):
        df = self.conformer_df()
        ref = np.asarray(df.iloc[0]["coords_embedded"], dtype=float)
        moved = np.asarray(df.iloc[1]["coords_embedded"], dtype=float)

        aligned = conformers_module._align_coords_to_reference(moved, ref, [0, 1, 2])

        np.testing.assert_allclose(aligned[[0, 1, 2]], ref[[0, 1, 2]], atol=1e-10)

    def test_conformer_scene_supports_role_based_core_and_cluster_mode(self):
        df = self.conformer_df()

        scene = conformer_ensemble_scene_from_dataframe(
            df,
            row_index=0,
            mode="cluster",
            color_by="cluster",
            n_clusters=2,
            top_n=4,
        )

        models = scene.cells[0].models
        self.assertGreaterEqual(len(models), 6)
        cluster_colors = {
            model.style["stick"]["color"]
            for model in models[1:]
            if model.style["stick"]["color"] != "black"
        }
        self.assertGreaterEqual(len(cluster_colors), 1)

    def test_plot_conformers_applies_model_specific_styles_before_export(self):
        class FakeViewer:
            def __init__(self):
                self.styles = []

            def setStyle(self, selector, style, viewer=None):
                self.styles.append((selector, style, viewer))

        class FakeRenderer:
            instances = []

            def __init__(self, scene):
                self.scene = scene
                self.viewer = FakeViewer()
                self.styles_at_export = None
                self.export_path = None
                FakeRenderer.instances.append(self)

            def render(self):
                return self.viewer

            def write_html(self, path):
                self.export_path = path
                self.styles_at_export = list(self.viewer.styles)

        with patch.object(conformers_module, "Py3DmolGridRenderer", FakeRenderer):
            viewer = plot_conformers(
                self.conformer_df(),
                mode="representatives+cloud",
                color_by="uniform",
                top_n=2,
                cloud_opacity=0.7,
                cloud_radius=0.08,
                cloud_color="#445566",
                export_HTML="conformers.html",
            )

        renderer = FakeRenderer.instances[-1]
        self.assertEqual(renderer.export_path, "conformers.html")
        self.assertEqual(renderer.styles_at_export, viewer.styles)
        self.assertEqual([style[0] for style in viewer.styles], [{"model": i} for i in range(7)])
        self.assertEqual(viewer.styles[0][1]["stick"]["color"], "black")
        self.assertEqual(viewer.styles[1][1]["stick"]["color"], "#445566")
        self.assertEqual(viewer.styles[1][1]["stick"]["opacity"], 0.7)
        self.assertEqual(viewer.styles[1][1]["stick"]["radius"], 0.08)
        self.assertEqual(viewer.styles[-2][1]["stick"]["color"], "#445566")

    def test_plot_conformers_defaults_to_all_conformer_families(self):
        class FakeViewer:
            def __init__(self):
                self.styles = []

            def setStyle(self, selector, style, viewer=None):
                self.styles.append((selector, style, viewer))

        class FakeRenderer:
            instances = []

            def __init__(self, scene):
                self.scene = scene
                self.viewer = FakeViewer()
                FakeRenderer.instances.append(self)

            def render(self):
                return self.viewer

            def write_html(self, path):
                pass

        with patch.object(conformers_module, "Py3DmolGridRenderer", FakeRenderer):
            plot_conformers(self.multi_rpos_conformer_df(), mode="single")

        renderer = FakeRenderer.instances[-1]
        self.assertEqual(len(renderer.scene.cells), 3)
        self.assertEqual(renderer.scene.columns, 3)
        self.assertEqual(
            {style[2] for style in renderer.viewer.styles},
            {(0, 0), (0, 1), (0, 2)},
        )

    def test_plot_conformers_row_index_selects_one_conformer_family(self):
        class FakeRenderer:
            instances = []

            def __init__(self, scene):
                self.scene = scene
                self.viewer = type(
                    "FakeViewer",
                    (),
                    {"setStyle": lambda *args, **kwargs: None},
                )()
                FakeRenderer.instances.append(self)

            def render(self):
                return self.viewer

            def write_html(self, path):
                pass

        with patch.object(conformers_module, "Py3DmolGridRenderer", FakeRenderer):
            plot_conformers(self.multi_rpos_conformer_df(), row_index=4, mode="single")

        renderer = FakeRenderer.instances[-1]
        self.assertEqual(len(renderer.scene.cells), 1)
        self.assertIn("anisole r4", renderer.scene.cells[0].title)


class StructureComparisonTests(unittest.TestCase):
    @staticmethod
    def _embedded_structure(smiles="CCO"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=17)
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        return atoms, coords

    @staticmethod
    def _write_xyz(path: Path, atoms, coords):
        lines = [str(len(atoms)), "test structure"]
        for atom, (x, y, z) in zip(atoms, coords):
            lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_compare_rmsd_from_xyz_paths_adds_deviation_overlays(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[1] += np.array([0.25, -0.10, 0.05])

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.xyz"
            probe_path = Path(tmpdir) / "probe.xyz"
            self._write_xyz(ref_path, atoms, ref_coords)
            self._write_xyz(probe_path, atoms, probe_coords)

            result = compare_rmsd(
                str(probe_path),
                str(ref_path),
                render=False,
                print_summary=False,
                top_n=2,
            )
            scene = result["scene"]

        self.assertEqual(len(scene.cells), 1)
        self.assertEqual(len(scene.cells[0].models), 2)
        overlay_types = [type(overlay).__name__ for overlay in scene.cells[0].overlays]
        self.assertEqual(overlay_types.count("DistanceOverlay"), 2)
        self.assertEqual(overlay_types.count("ScreenLabelOverlay"), 2)
        self.assertEqual(overlay_types.count("AtomHighlight"), 4)
        distance_overlays = [
            overlay
            for overlay in scene.cells[0].overlays
            if type(overlay).__name__ == "DistanceOverlay"
        ]
        screen_labels = [
            overlay
            for overlay in scene.cells[0].overlays
            if type(overlay).__name__ == "ScreenLabelOverlay"
        ]
        self.assertTrue(all(overlay.atom1 >= len(atoms) for overlay in distance_overlays))
        self.assertTrue(all(overlay.atom2 < len(atoms) for overlay in distance_overlays))
        self.assertTrue(all(overlay.label is None for overlay in distance_overlays))
        self.assertEqual(screen_labels[0].screen_offset, {"x": 10, "y": 34})
        self.assertEqual(screen_labels[1].screen_offset, {"x": 10, "y": 58})

    def test_compare_rmsd_render_false_returns_scene_without_viewer(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[2] += np.array([0.10, 0.05, -0.20])

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.xyz"
            probe_path = Path(tmpdir) / "probe.xyz"
            self._write_xyz(ref_path, atoms, ref_coords)
            self._write_xyz(probe_path, atoms, probe_coords)

            result = vis.compare_rmsd(
                str(probe_path),
                str(ref_path),
                render=False,
                print_summary=False,
            )

        self.assertIsNotNone(result["scene"])
        self.assertIsNone(result["viewer"])
        self.assertGreater(result["rmsd"], 0.0)
        rmsd_from_table = np.sqrt(np.mean(result["df_dev"]["distance_A"].to_numpy() ** 2))
        self.assertAlmostEqual(result["rmsd"], rmsd_from_table)

    def test_compare_rmsd_uses_dataframe_coordinate_columns(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[0] += np.array([0.20, 0.00, 0.00])
        df = pd.DataFrame(
            {
                "substrate_name": ["ethanol"],
                "rpos": [1],
                "atoms": [atoms],
                "gxtb-oc": [probe_coords],
                "orca-oc": [ref_coords],
            }
        )

        result = compare_rmsd(
            {"df": df, "coords_col": "gxtb-oc"},
            {"df": df, "coords_col": "orca-oc"},
            render=False,
            print_summary=False,
            top_n=1,
        )
        scene = result["scene"]

        self.assertEqual(result["probe_label"], "ethanol r1 gxtb-oc")
        self.assertEqual(result["ref_label"], "ethanol r1 orca-oc")
        self.assertGreater(result["rmsd"], 0.0)
        self.assertIn("ethanol r1", scene.cells[0].title)
        overlay_types = [type(overlay).__name__ for overlay in scene.cells[0].overlays]
        self.assertEqual(overlay_types.count("DistanceOverlay"), 1)
        self.assertEqual(overlay_types.count("ScreenLabelOverlay"), 1)

    def test_compare_rmsd_mixes_xyz_file_and_dataframe_row(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[1] += np.array([0.15, -0.05, 0.10])
        df = pd.DataFrame(
            {
                "substrate_name": ["ethanol"],
                "atoms": [atoms],
                "wb97-oc": [ref_coords],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            probe_path = Path(tmpdir) / "font2017.xyz"
            self._write_xyz(probe_path, atoms, probe_coords)
            result = compare_rmsd(
                {"path": probe_path, "label": "Font 2017"},
                {
                    "df": df,
                    "row_index": 0,
                    "coords_col": "wb97-oc",
                    "label": "FRUST wb97",
                },
                render=False,
                print_summary=False,
            )

        self.assertEqual(result["probe_label"], "Font 2017")
        self.assertEqual(result["ref_label"], "FRUST wb97")
        self.assertEqual(result["probe_source"], "xyz_path")
        self.assertEqual(result["ref_source"], "dataframe")
        self.assertGreater(result["rmsd"], 0.0)

    def test_compare_rmsd_show_none_skips_scene(self):
        atoms, coords = self._embedded_structure()
        df = pd.DataFrame(
            {
                "atoms": [atoms],
                "probe-oc": [coords],
                "ref-oc": [coords],
            }
        )

        result = compare_rmsd(
            {"df": df, "coords_col": "probe-oc"},
            {"df": df, "coords_col": "ref-oc"},
            show="none",
            render=False,
            print_summary=False,
        )

        self.assertIsNone(result["scene"])
        self.assertIsNone(result["viewer"])
        self.assertAlmostEqual(result["rmsd"], 0.0)

    def test_structure_comparison_rejects_invalid_show_mode(self):
        atoms, coords = self._embedded_structure()
        df = pd.DataFrame(
            {
                "atoms": [atoms],
                "probe-oc": [coords],
                "ref-oc": [coords],
            }
        )

        with self.assertRaisesRegex(ValueError, "Invalid show mode"):
            compare_rmsd(
                {"df": df, "coords_col": "probe-oc"},
                {"df": df, "coords_col": "ref-oc"},
                show="bad",
                render=False,
                print_summary=False,
            )

    def test_compare_rmsd_accepts_xyz_block_and_atoms_coords_tuple(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[0] += np.array([0.10, 0.00, 0.00])
        xyz_block = "\n".join(
            [
                str(len(atoms)),
                "probe",
                *[
                    f"{atom} {x:.10f} {y:.10f} {z:.10f}"
                    for atom, (x, y, z) in zip(atoms, probe_coords)
                ],
            ]
        )

        result = compare_rmsd(
            {"xyz": xyz_block, "label": "block"},
            (atoms, ref_coords),
            mapping="index",
            render=False,
            print_summary=False,
        )

        self.assertEqual(result["mapping"], "index")
        self.assertEqual(result["probe_label"], "block")
        self.assertEqual(result["ref_label"], "reference")
        self.assertGreater(result["rmsd"], 0.0)

    def test_compare_rmsd_accepts_object_array_coordinate_rows(self):
        atoms, coords = self._embedded_structure()
        object_coords = np.empty(len(coords), dtype=object)
        for idx, row in enumerate(coords):
            object_coords[idx] = np.asarray(row)

        result = compare_rmsd(
            {"atoms": atoms, "coords": object_coords},
            {"atoms": atoms, "coords": coords},
            render=False,
            print_summary=False,
        )

        self.assertAlmostEqual(result["rmsd"], 0.0)

    def test_compare_rmsd_bare_xyz_block_string_is_rejected(self):
        atoms, coords = self._embedded_structure()
        xyz_block = "\n".join(
            [
                str(len(atoms)),
                "probe",
                *[
                    f"{atom} {x:.10f} {y:.10f} {z:.10f}"
                    for atom, (x, y, z) in zip(atoms, coords)
                ],
            ]
        )

        with self.assertRaisesRegex(ValueError, r"\{'xyz': xyz_block\}"):
            compare_rmsd(
                xyz_block,
                (atoms, coords),
                render=False,
                print_summary=False,
            )

    def test_compare_rmsd_index_mapping_tolerates_display_bond_failure(self):
        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[1] += np.array([0.05, 0.00, 0.00])

        with patch(
            "frust.utils.RMSD.rdDetermineBonds.DetermineBonds",
            side_effect=RuntimeError("bond perception failed"),
        ):
            result = compare_rmsd(
                {"atoms": atoms, "coords": probe_coords},
                {"atoms": atoms, "coords": ref_coords},
                mapping="index",
                render=False,
                print_summary=False,
            )

        self.assertEqual(result["mapping"], "index")
        self.assertEqual(result["probe_display_bonds"], "none")
        self.assertEqual(result["ref_display_bonds"], "none")
        self.assertGreater(result["rmsd"], 0.0)

    def test_compare_rmsd_geometry_mapping_handles_shuffled_atom_order(self):
        atoms, coords = self._embedded_structure()
        permutation = [2, 0, 1, *range(3, len(atoms))]
        probe_atoms = [atoms[idx] for idx in permutation]
        probe_coords = coords[permutation]

        result = compare_rmsd(
            {"atoms": probe_atoms, "coords": probe_coords},
            {"atoms": atoms, "coords": coords},
            mapping="geometry",
            render=False,
            print_summary=False,
        )

        self.assertEqual(result["mapping"], "geometry")
        self.assertAlmostEqual(result["rmsd"], 0.0)
        self.assertIn((0, 2), result["atom_map"])
        self.assertIn((1, 0), result["atom_map"])
        self.assertIn((2, 1), result["atom_map"])

    def test_compare_rmsd_geometry_mapping_preserves_display_bonds(self):
        atoms, coords = self._embedded_structure()
        permutation = [2, 0, 1, *range(3, len(atoms))]
        probe_atoms = [atoms[idx] for idx in permutation]
        probe_coords = coords[permutation]

        result = compare_rmsd(
            {"atoms": probe_atoms, "coords": probe_coords},
            {"atoms": atoms, "coords": coords},
            mapping="geometry",
            render=False,
            print_summary=False,
        )

        self.assertEqual(result["probe_display_bonds"], "perceived")
        self.assertEqual(result["ref_display_bonds"], "perceived")
        self.assertGreater(result["probe_mol_aligned"].GetNumBonds(), 0)
        self.assertGreater(result["ref_mol"].GetNumBonds(), 0)

    def test_compare_rmsd_geometry_mapping_uses_dataframe_connectivity_bonds_for_display(self):
        atoms, coords = self._embedded_structure()
        probe_coords = coords.copy()
        probe_coords[1] += np.array([0.02, 0.00, 0.00])
        bonds = np.array([[0, 1], [1, 2]], dtype=object)
        df = pd.DataFrame(
            {
                "atoms": [atoms],
                "probe-oc": [probe_coords],
                "ref-oc": [coords],
                "connectivity_bonds": [bonds],
            }
        )

        with patch(
            "frust.utils.RMSD.rdDetermineBonds.DetermineBonds",
            side_effect=RuntimeError("bond perception failed"),
        ):
            result = compare_rmsd(
                {"df": df, "coords_col": "probe-oc"},
                {"df": df, "coords_col": "ref-oc"},
                mapping="geometry",
                render=False,
                print_summary=False,
            )

        self.assertEqual(result["probe_display_bonds"], "input")
        self.assertEqual(result["ref_display_bonds"], "input")
        self.assertEqual(result["probe_mol_aligned"].GetNumBonds(), 2)
        self.assertEqual(result["ref_mol"].GetNumBonds(), 2)

    def test_compare_rmsd_connectivity_mapping_uses_dataframe_bonds_for_reordered_atoms(self):
        old_atoms = ["N", "C", "C", "C", "C", "C"]
        old_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        old_bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)]
        new_atoms = ["N", "C", "C", "C", "C", "C"]
        new_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        new_bonds = [
            (0, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 1),
            (0, 3),
        ]
        old_df = pd.DataFrame(
            {
                "atoms": [old_atoms],
                "OptTS-oc": [old_coords],
                "connectivity_bonds": [old_bonds],
            }
        )
        new_df = pd.DataFrame(
            {
                "atoms": [new_atoms],
                "OptTS-oc": [new_coords],
                "connectivity_bonds": [new_bonds],
            }
        )

        result = compare_rmsd(
            {"df": old_df, "coords_col": "OptTS-oc"},
            {"df": new_df, "coords_col": "OptTS-oc"},
            mapping="connectivity",
            render=False,
            print_summary=False,
        )
        reverse = compare_rmsd(
            {"df": new_df, "coords_col": "OptTS-oc"},
            {"df": old_df, "coords_col": "OptTS-oc"},
            mapping="connectivity",
            render=False,
            print_summary=False,
        )

        self.assertEqual(result["mapping"], "connectivity")
        self.assertIn((5, 1), result["atom_map"])
        self.assertNotIn((5, 3), result["atom_map"])
        self.assertIn((1, 5), reverse["atom_map"])
        self.assertEqual(result["probe_display_bonds"], "input")
        self.assertEqual(result["ref_display_bonds"], "input")

    def test_compare_rmsd_connectivity_mapping_requires_bonds(self):
        atoms, coords = self._embedded_structure()

        with self.assertRaisesRegex(ValueError, "requires bonds for both structures"):
            compare_rmsd(
                {"atoms": atoms, "coords": coords},
                {"atoms": atoms, "coords": coords},
                mapping="connectivity",
                render=False,
                print_summary=False,
            )

    def test_probe_style_keeps_hetero_atom_colors(self):
        self.assertEqual(PROBE_STYLE["stick"]["colorscheme"], "orangeCarbon")
        self.assertEqual(PROBE_STYLE["sphere"]["colorscheme"], "orangeCarbon")
        self.assertNotIn("color", PROBE_STYLE["stick"])
        self.assertNotIn("color", PROBE_STYLE["sphere"])

    def test_structure_comparison_scene_accepts_molto3d_style_options(self):
        atoms, coords = self._embedded_structure()
        df = pd.DataFrame(
            {
                "atoms": [atoms],
                "probe-oc": [coords],
                "ref-oc": [coords],
            }
        )

        result = compare_rmsd(
            {"df": df, "coords_col": "probe-oc"},
            {"df": df, "coords_col": "ref-oc"},
            show="overlay",
            render=False,
            print_summary=False,
            background_color=("white", 1.0),
            show_labels=True,
            show_charges=False,
            kekulize=False,
        )
        scene = result["scene"]

        self.assertEqual(scene.background_color, ("white", 1.0))
        for model in scene.cells[0].models:
            self.assertTrue(model.show_atom_labels)
            self.assertFalse(model.show_charges)
            self.assertFalse(model.kekulize)

    def test_compare_rmsd_applies_model_specific_styles_before_export(self):
        class FakeViewer:
            def __init__(self):
                self.styles = []
                self.show_calls = 0

            def setStyle(self, selector, style, viewer=None):
                self.styles.append((selector, style, viewer))

            def show(self):
                self.show_calls += 1

        class FakeRenderer:
            instances = []

            def __init__(self, scene):
                self.scene = scene
                self.viewer = FakeViewer()
                self.styles_at_export = None
                self.export_path = None
                FakeRenderer.instances.append(self)

            def render(self):
                return self.viewer

            def write_html(self, path):
                self.export_path = path
                self.styles_at_export = list(self.viewer.styles)

        atoms, ref_coords = self._embedded_structure()
        probe_coords = ref_coords.copy()
        probe_coords[1] += np.array([0.20, 0.00, 0.00])

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "ref.xyz"
            probe_path = Path(tmpdir) / "probe.xyz"
            self._write_xyz(ref_path, atoms, ref_coords)
            self._write_xyz(probe_path, atoms, probe_coords)
            with patch.object(
                structure_comparison_module,
                "Py3DmolGridRenderer",
                FakeRenderer,
            ):
                result = vis.compare_rmsd(
                    str(probe_path),
                    str(ref_path),
                    render=False,
                    export_HTML="comparison.html",
                    print_summary=False,
                )

        renderer = FakeRenderer.instances[-1]
        expected_styles = [
            ({"model": 0}, REFERENCE_STYLE, (0, 0)),
            ({"model": 1}, PROBE_STYLE, (0, 0)),
            ({"elem": "H"}, {}, (0, 0)),
        ]
        self.assertEqual(result["viewer"].styles, expected_styles)
        self.assertEqual(renderer.styles_at_export, expected_styles)
        self.assertEqual(renderer.export_path, "comparison.html")
        self.assertEqual(result["viewer"].show_calls, 0)
if __name__ == "__main__":
    unittest.main()
