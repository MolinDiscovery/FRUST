import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import frust as ft
from frust.stepper import Stepper
from frust.utils.pruning import prune_conformers


def _pruning_df() -> pd.DataFrame:
    atoms = ["C", "H", "H"]
    coords = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-1.0, 0.0, 0.0]],
    ]
    return pd.DataFrame(
        {
            "system_name": ["sys", "sys", "sys", "sys"],
            "substrate_name": ["sub", "sub", "sub", "sub"],
            "catalyst_name": ["cat", "cat", "cat", "cat"],
            "structure_type": ["TS1", "TS1", "TS1", "TS1"],
            "molecule_role": ["ts", "ts", "ts", "ts"],
            "rpos": [4, 4, 4, 4],
            "cid": [0, 1, 2, 3],
            "atoms": [atoms, atoms, atoms, atoms],
            "coords_embedded": coords,
            "xtb_opt-oc": [[[x + 9.0, y, z] for x, y, z in conf] for conf in coords],
            "energy": [10.0, 0.0, 5.0, 2.0],
        }
    )


class PruneConformersTests(unittest.TestCase):
    def test_moi_and_rmsd_masks_compose_with_grouping_and_attrs(self):
        calls = []

        def fake_moi(coords, atoms, **kwargs):
            calls.append(("moi", len(coords), list(atoms)))
            if len(coords) == 4:
                mask = np.array([True, False, True, True])
            else:
                mask = np.ones(len(coords), dtype=bool)
            return coords[mask], mask

        def fake_rmsd(coords, atoms, **kwargs):
            calls.append(("rmsd", len(coords), kwargs["max_rmsd"]))
            mask = np.array([False, True, True]) if len(coords) == 3 else np.ones(len(coords), dtype=bool)
            return coords[mask], mask

        df = _pruning_df()
        df.attrs["source"] = "kept"

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, fake_rmsd, None, None),
        ):
            out = prune_conformers(df, modes=("moi", "rmsd"), rmsd_max_rmsd=0.4)

        self.assertEqual(list(out["cid"]), [2, 3])
        self.assertEqual(out.attrs["source"], "kept")
        self.assertEqual(calls, [("moi", 4, ["C", "H", "H"]), ("rmsd", 3, 0.4)])
        step = out.attrs["frust_steps"]["initial_prune"]
        self.assertEqual(step["engine"], "prism_pruner")
        self.assertEqual(step["row_counts"]["input_rows"], 4)
        self.assertEqual(step["row_counts"]["output_rows"], 2)
        shown = ft.show_steps(out)
        self.assertEqual(shown.loc["initial_prune", "engine"], "prism_pruner")
        self.assertEqual(shown.loc["initial_prune", "input_rows"], 4)
        self.assertEqual(shown.loc["initial_prune", "output_rows"], 2)

    def test_pruning_does_not_cross_structure_groups(self):
        group_a = _pruning_df().iloc[:2].copy()
        group_b = _pruning_df().iloc[:2].copy()
        group_b["structure_type"] = "TS2"
        df = pd.concat([group_a, group_b], ignore_index=True)
        group_sizes = []

        def fake_moi(coords, atoms, **kwargs):
            group_sizes.append(len(coords))
            mask = np.array([True, False])
            return coords[mask], mask

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, None, None, None),
        ):
            out = prune_conformers(df, modes=("moi",))

        self.assertEqual(group_sizes, [2, 2])
        self.assertEqual(list(out["structure_type"]), ["TS1", "TS2"])

    def test_energy_col_sorting_makes_lower_energy_row_survive(self):
        df = _pruning_df().iloc[:2].copy()

        def fake_moi(coords, atoms, **kwargs):
            mask = np.array([True, False])
            return coords[mask], mask

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, None, None, None),
        ):
            out = prune_conformers(df, modes=("moi",), energy_col="energy")

        self.assertEqual(list(out["cid"]), [1])

    def test_single_string_mode_is_treated_as_one_mode(self):
        def fake_moi(coords, atoms, **kwargs):
            mask = np.array([True, False, True, True])
            return coords[mask], mask

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, None, None, None),
        ):
            out = prune_conformers(_pruning_df(), modes="moi")

        self.assertEqual(list(out["cid"]), [0, 2, 3])

    def test_atom_order_mismatch_raises(self):
        df = _pruning_df().iloc[:2].copy()
        df.at[1, "atoms"] = ["H", "C", "H"]

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(lambda coords, atoms, **kwargs: (coords, np.ones(len(coords), dtype=bool)), None, None, None),
        ):
            with self.assertRaisesRegex(ValueError, "same atom order"):
                prune_conformers(df, modes=("moi",))

    def test_rot_corr_uses_connectivity_bonds_when_present(self):
        df = _pruning_df().iloc[:2].copy()
        df["connectivity_bonds"] = [[(0, 1), (0, 2)], [(0, 1), (0, 2)]]
        captured = {}

        def fake_rot(coords, atoms, graph, **kwargs):
            captured["graph"] = graph
            mask = np.array([True, False])
            return coords[mask], mask

        fake_graph = SimpleNamespace(edges={(0, 1), (0, 2)})

        with (
            patch(
                "frust.utils.pruning._load_prism_functions",
                return_value=(None, None, fake_rot, lambda atoms, coords: SimpleNamespace(edges=set())),
            ),
            patch("frust.utils.pruning._graph_from_bonds", return_value=fake_graph) as graph_from_bonds,
        ):
            out = prune_conformers(df, modes=("rot_corr_rmsd",))

        self.assertEqual(list(out["cid"]), [0])
        graph_from_bonds.assert_called_once()
        self.assertIs(captured["graph"], fake_graph)

    def test_missing_prism_dependency_raises_clear_error(self):
        with patch(
            "frust.utils.pruning.importlib.import_module",
            side_effect=ModuleNotFoundError("missing", name="prism_pruner"),
        ):
            with self.assertRaisesRegex(ImportError, "optional `prism_pruner` package"):
                prune_conformers(_pruning_df(), modes=("moi",))

    def test_stepper_prune_conformers_resolves_latest_coordinate_column(self):
        recorded = {}

        def fake_moi(coords, atoms, **kwargs):
            recorded["first_x"] = float(coords[0, 0, 0])
            mask = np.array([True, False, True, True])
            return coords[mask], mask

        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, None, None, None),
        ):
            out = Stepper(save_output_dir=False).prune_conformers(
                _pruning_df(),
                modes=("moi",),
            )

        self.assertEqual(recorded["first_x"], 9.0)
        self.assertIn("initial_prune", out.attrs["frust_steps"])

    def test_stepper_prune_conformers_logs_summary(self):
        messages = []

        def fake_moi(coords, atoms, **kwargs):
            mask = np.array([True, False, True, True])
            return coords[mask], mask

        def fake_rmsd(coords, atoms, **kwargs):
            mask = np.ones(len(coords), dtype=bool)
            return coords[mask], mask

        step = Stepper(save_output_dir=False)
        step.logger = SimpleNamespace(info=messages.append)
        with patch(
            "frust.utils.pruning._load_prism_functions",
            return_value=(fake_moi, fake_rmsd, None, None),
        ):
            step.prune_conformers(
                _pruning_df(),
                modes=("moi", "rmsd"),
                moi_max_deviation=0.03,
                rmsd_max_rmsd=0.4,
                rmsd_max_dev=0.8,
            )

        self.assertIn(
            "[initial_prune] kept 3/4 row(s); dropped=1; modes=moi,rmsd; "
            "moi_max_deviation=0.03; rmsd_max_rmsd=0.4; rmsd_max_dev=0.8",
            messages,
        )


if __name__ == "__main__":
    unittest.main()
