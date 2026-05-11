import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import same_color

import frust.vis as vis
from frust.vis import MolTo3DGrid, RxnTo3DGrid, plot_energy_profile, plot_mols
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


if __name__ == "__main__":
    unittest.main()
