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
from frust.vis import GridScene, MoleculeModel, SceneCell
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
        self.assertIs(vis.MolTo3DGrid, MolTo3DGrid)
        self.assertIs(vis.RxnTo3DGrid, RxnTo3DGrid)
        self.assertTrue(callable(vis.show_scene))
        self.assertTrue(callable(vis.molecule_scene_from_dataframe))
        self.assertTrue(callable(vis.vibration_scene_from_dataframe))
        self.assertTrue(callable(vis.ts_guess_scene))

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

    def test_ts_guess_scene_adds_role_and_distance_overlays(self):
        df = self.small_molecule_df().iloc[[0]].copy()
        df["constraint_roles"] = [{"cat_B": 0, "transfer_H": 1}]
        df["constraint_spec"] = [
            [{"kind": "distance", "roles": ["cat_B", "transfer_H"], "value": 1.2}]
        ]

        scene = ts_guess_scene_from_dataframe(
            df,
            row_indices=[0],
            show_roles=True,
            show_constraint_distances=True,
        )

        overlay_types = {type(overlay).__name__ for overlay in scene.cells[0].overlays}
        self.assertIn("AtomLabel", overlay_types)
        self.assertIn("AtomHighlight", overlay_types)
        self.assertIn("DistanceOverlay", overlay_types)


if __name__ == "__main__":
    unittest.main()
