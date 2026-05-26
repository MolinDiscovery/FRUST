import unittest
from unittest.mock import patch

import pandas as pd
from tooltoad.chemutils import ac2mol

import frust as ft
from frust.constraints import render_orca_constraints, render_xtb_constraints
from frust.stepper import Stepper
from frust.tsguess.matching import match_catalyst_roles, mol_from_smiles
from frust.vis.molecules import _row_to_mol


CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"


class ScreenWorkflowTests(unittest.TestCase):
    def test_read_normalizes_roles_names_and_catalyst_rpos(self):
        raw = pd.DataFrame(
            {
                "role": ["sub", "cat"],
                "smiles": ["CN1C=CC=C1", CATALYST],
                "rpos": ["2,3", "4"],
                "batch": ["a", "b"],
            }
        )

        with self.assertWarnsRegex(UserWarning, "Catalyst rows"):
            components = ft.screen.read(raw)

        self.assertEqual(list(components["role"]), ["substrate", "catalyst"])
        self.assertEqual(list(components["compound_name"]), ["substrate_000", "catalyst_000"])
        self.assertTrue(pd.isna(components.loc[1, "rpos"]))
        self.assertEqual(list(components["batch"]), ["a", "b"])

        with self.assertRaisesRegex(ValueError, "Catalyst rows"):
            ft.screen.read(raw, strict=True)

    def test_expand_preserves_substrate_rpos_and_metadata(self):
        components = ft.screen.read(
            pd.DataFrame(
                {
                    "role": ["substrate", "substrate", "catalyst", "catalyst", "catalyst"],
                    "smiles": ["CN1C=CC=C1", "COc1ccccc1", CATALYST, CATALYST, CATALYST],
                    "compound_name": ["pyrrole", "anisole", "cat_a", "cat_b", "cat_c"],
                    "rpos": ["2", "2;3", None, None, None],
                    "tag": ["s1", "s2", "c1", "c2", "c3"],
                }
            )
        )

        systems = ft.screen.expand(components)

        self.assertEqual(len(systems), 6)
        self.assertEqual(systems.iloc[0]["system_name"], "pyrrole__cat_a")
        self.assertEqual(systems.iloc[0]["rpos"], "2")
        self.assertEqual(systems.iloc[0]["substrate_tag"], "s1")
        self.assertEqual(systems.iloc[0]["catalyst_tag"], "c1")

    def test_strict_catalyst_match(self):
        catalyst = mol_from_smiles(CATALYST, label="cat")
        roles = match_catalyst_roles(catalyst, catalyst_name="cat")

        self.assertEqual(set(roles), {"cat_B", "cat_N"})

        bad = mol_from_smiles("c1ccccc1", label="benzene")
        with self.assertRaisesRegex(ValueError, "B-aryl-N"):
            match_catalyst_roles(bad, catalyst_name="benzene")

    def test_create_ts_guesses_returns_grouped_dataframes(self):
        systems = ft.screen.expand(
            ft.screen.read(
                pd.DataFrame(
                    {
                        "role": ["substrate", "catalyst"],
                        "smiles": ["CN1C=CC=C1", CATALYST],
                        "compound_name": ["pyrrole", "cat_a"],
                        "rpos": ["2", None],
                    }
                )
            )
        )

        guesses = ft.screen.create_ts_guesses(
            systems,
            ts_types=["TS1", "TS2", "TS3", "TS4"],
            n_confs=1,
        )

        self.assertEqual(set(guesses), {"TS1", "TS2", "TS3", "TS4"})
        for ts_type, df in guesses.items():
            self.assertEqual(len(df), 1)
            self.assertEqual(df["structure_type"].iloc[0], ts_type)
            self.assertEqual(df["system_name"].iloc[0], "pyrrole__cat_a")
            self.assertIn("constraint_roles", df.columns)
            self.assertIn("constraint_spec", df.columns)
            self.assertEqual(len(df["atoms"].iloc[0]), len(df["coords_embedded"].iloc[0]))

    def test_multifragment_ts_guess_does_not_collapse_fragments(self):
        systems = ft.screen.expand(
            ft.screen.read(
                pd.DataFrame(
                    {
                        "role": ["substrate", "catalyst"],
                        "smiles": ["N1(CC2=CC=CC=C2)C=CC=C1", CATALYST],
                    }
                )
            )
        )

        guesses = ft.screen.create_ts_guesses(systems, ts_types=["TS4"], n_confs=1)
        row = guesses["TS4"].loc[guesses["TS4"]["rpos"].eq(8)].iloc[0]
        mol = ac2mol(row["atoms"], row["coords_embedded"])
        distance_deltas = [
            abs(metric["delta"])
            for metric in row["ts_core_metrics"]
            if metric["kind"] == "distance"
        ]

        self.assertLess(mol.GetNumBonds(), 120)
        self.assertLess(max(distance_deltas), 0.16)

    def test_ts_guesses_are_dehydrogenated_and_store_plot_connectivity(self):
        systems = ft.screen.expand(
            ft.screen.read(
                pd.DataFrame(
                    {
                        "role": ["substrate", "catalyst"],
                        "smiles": ["CN1C=CC=C1", CATALYST],
                        "rpos": ["2", None],
                    }
                )
            )
        )

        guesses = ft.screen.create_ts_guesses(
            systems,
            ts_types=["TS1", "TS2", "TS3", "TS4"],
            n_confs=1,
        )

        for df in guesses.values():
            row = df.iloc[0]
            mol = _row_to_mol(row, row["atoms"], row["coords_embedded"])
            substrate_c = row["constraint_roles"]["substrate_C"]
            substrate_c_neighbors = mol.GetAtomWithIdx(substrate_c).GetNeighbors()
            substrate_c_hydrogens = [
                neighbor.GetIdx()
                for neighbor in substrate_c_neighbors
                if neighbor.GetAtomicNum() == 1
            ]

            self.assertEqual(mol.GetNumBonds(), len(row["connectivity_bonds"]))
            self.assertEqual(substrate_c_hydrogens, [])

    def test_plot_connectivity_avoids_distance_perception_artifacts(self):
        systems = ft.screen.expand(
            ft.screen.read(
                pd.DataFrame(
                    {
                        "role": ["substrate", "catalyst"],
                        "smiles": ["CN1C=CC=C1", CATALYST],
                        "rpos": ["2", None],
                    }
                )
            )
        )

        row = ft.screen.create_ts_guesses(systems, ts_types=["TS2"], n_confs=1)["TS2"].iloc[0]
        inferred = ac2mol(row["atoms"], row["coords_embedded"])
        plotted = _row_to_mol(row, row["atoms"], row["coords_embedded"])

        self.assertGreater(inferred.GetNumBonds(), len(row["connectivity_bonds"]))
        self.assertEqual(plotted.GetNumBonds(), len(row["connectivity_bonds"]))

    def test_constraint_renderers_and_stepper_row_first(self):
        row = pd.Series(
            {
                "constraint_roles": {"cat_B": 0, "transfer_H": 1, "substrate_C": 2},
                "constraint_spec": [
                    {"kind": "distance", "roles": ["cat_B", "transfer_H"], "value": 1.5},
                    {
                        "kind": "angle",
                        "roles": ["cat_B", "transfer_H", "substrate_C"],
                        "value": 90.0,
                    },
                ],
            }
        )

        self.assertIn("distance: 1, 2, 1.5", render_xtb_constraints(row))
        self.assertIn("{A 0 1 2 90 C}", render_orca_constraints(row))

        df = pd.DataFrame(
            {
                "custom_name": ["row_ts"],
                "substrate_name": ["sub"],
                "system_name": ["sub__cat"],
                "structure_type": ["TS1"],
                "molecule_role": ["ts"],
                "rpos": [2],
                "cid": [0],
                "atoms": [["H", "H", "H"]],
                "coords_embedded": [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
                "constraint_roles": [row["constraint_roles"]],
                "constraint_spec": [row["constraint_spec"]],
            }
        )
        captured = []

        def fake_xtb_calculate(atoms, coords, options, detailed_input_str=None):
            captured.append(detailed_input_str)
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        step = Stepper(debug=True, save_output_dir=False)
        step.xtb_fn = fake_xtb_calculate
        out = step.xtb(df, name="probe", options={"gfnff": None}, constraint=True)

        self.assertTrue(out["probe-NT"].iloc[0])
        self.assertIn("$constrain", captured[0])
        self.assertIn("distance: 1, 2, 1.5", captured[0])


if __name__ == "__main__":
    unittest.main()
