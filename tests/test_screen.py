import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from tooltoad.chemutils import ac2mol

import frust as ft
from frust.constraints import render_orca_constraints, render_xtb_constraints
from frust.stepper import Stepper
from frust.tsguess.matching import match_catalyst_roles, mol_from_smiles
from frust.vis.molecules import _row_to_mol


CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"
TS2_TEMPLATE_SUBSTRATE_FRAME = np.array(
    [
        (4.032469, 5.165170, -0.184499),
        (3.638645, 3.836721, -0.134114),
        (2.362445, 3.792621, -0.684305),
    ],
    dtype=float,
)
TS3_TEMPLATE_SUBSTRATE_FRAME = np.array(
    [
        (3.160721, 1.507283, 2.235734),
        (1.976672, 0.248906, 2.068494),
        (1.648714, -0.239019, 3.315702),
    ],
    dtype=float,
)
TS4_TEMPLATE_SUBSTRATE_FRAME = np.array(
    [
        (-0.719273, 1.363805, 4.969583),
        (0.013065, 0.446161, 3.676874),
        (0.189969, -0.867008, 4.103431),
    ],
    dtype=float,
)
SCREEN_PANEL_SUBSTRATES = [
    "CN1C=CC=C1",
    "C1=CC=CO1",
    "COC1=CC=CO1",
    "CC([Si](N1C=CC=C1)(C(C)C)C(C)C)C",
]


def _unit_normal(points):
    normal = np.cross(points[0] - points[1], points[2] - points[1])
    return normal / np.linalg.norm(normal)


def _excluded_atoms(excluded):
    if isinstance(excluded, int):
        return {excluded}
    return set(excluded)


def _ordered_substrate_frame_neighbors(mol, substrate_c, excluded):
    excluded = _excluded_atoms(excluded)
    neighbors = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(substrate_c).GetNeighbors()
        if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in excluded
    ]
    return sorted(
        neighbors,
        key=lambda idx: (
            mol.GetAtomWithIdx(idx).GetAtomicNum() == 6,
            idx,
        ),
    )


def _substrate_heavy_component(mol, substrate_c, excluded):
    excluded = _excluded_atoms(excluded)
    visited = {substrate_c}
    frontier = [substrate_c]
    while frontier:
        atom_idx = frontier.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in excluded or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor_idx in visited:
                continue
            visited.add(neighbor_idx)
            frontier.append(neighbor_idx)
    return visited


def _role_bond_exists(mol, roles, left, right):
    return mol.GetBondBetweenAtoms(roles[left], roles[right]) is not None


def _has_distance_constraint(row, left, right):
    target = {left, right}
    return any(
        constraint["kind"] == "distance" and set(constraint["roles"]) == target
        for constraint in row["constraint_spec"]
    )


class ScreenWorkflowTests(unittest.TestCase):
    def assert_core_metrics_close(self, row, *, distance_tol=0.16, angle_tol=4.0):
        for metric in row["ts_core_metrics"]:
            delta = abs(float(metric["delta"]))
            if metric["kind"] == "distance":
                self.assertLess(delta, distance_tol, metric)
            elif metric["kind"] == "angle":
                self.assertLess(delta, angle_tol, metric)

    def assert_sane_ts2_row(self, row):
        roles = row["constraint_roles"]
        coords = np.array(row["coords_embedded"], dtype=float)
        mol = _row_to_mol(row, row["atoms"], row["coords_embedded"])
        cat_b = roles["cat_B"]
        cat_h = roles["cat_H"]
        substrate_c = roles["substrate_C"]
        h2_atoms = [roles["transfer_H"], roles["n_transfer_H"]]

        b_hydrogens = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(cat_b).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]
        self.assertEqual(b_hydrogens, [cat_h])
        self.assertIsNotNone(mol.GetBondBetweenAtoms(cat_b, substrate_c))

        frame_neighbors = _ordered_substrate_frame_neighbors(mol, substrate_c, cat_b)
        self.assertEqual(len(frame_neighbors), 2)
        frame_points = coords[[frame_neighbors[0], substrate_c, frame_neighbors[1]]]
        template_normal = _unit_normal(TS2_TEMPLATE_SUBSTRATE_FRAME)
        actual_normal = _unit_normal(frame_points)
        self.assertGreater(float(np.dot(actual_normal, template_normal)), 0.95)

        substrate_heavy = _substrate_heavy_component(mol, substrate_c, cat_b)
        h2_to_substrate = [
            float(np.linalg.norm(coords[h_atom] - coords[substrate_atom]))
            for h_atom in h2_atoms
            for substrate_atom in substrate_heavy
        ]
        self.assertGreater(min(h2_to_substrate), 1.8)

    def assert_sane_ts3_row(self, row):
        roles = row["constraint_roles"]
        coords = np.array(row["coords_embedded"], dtype=float)
        mol = _row_to_mol(row, row["atoms"], row["coords_embedded"])
        cat_b = roles["cat_B"]
        pin_b = roles["pin_B"]
        substrate_c = roles["substrate_C"]
        transfer_h = roles["transfer_H"]
        cat_h = roles["cat_H"]

        cat_b_hydrogens = sorted(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(cat_b).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        pin_b_hydrogens = sorted(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(pin_b).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        self.assertEqual(cat_b_hydrogens, [cat_h])
        self.assertEqual(pin_b_hydrogens, [transfer_h])
        self.assertTrue(_role_bond_exists(mol, roles, "transfer_H", "pin_B"))
        self.assertFalse(_role_bond_exists(mol, roles, "cat_B", "pin_B"))
        self.assertTrue(_role_bond_exists(mol, roles, "cat_B", "substrate_C"))
        self.assertFalse(_role_bond_exists(mol, roles, "pin_B", "substrate_C"))
        self.assertTrue(_has_distance_constraint(row, "cat_B", "pin_B"))

        frame_neighbors = _ordered_substrate_frame_neighbors(
            mol,
            substrate_c,
            {cat_b, pin_b},
        )
        frame_points = coords[[frame_neighbors[0], substrate_c, frame_neighbors[1]]]
        template_normal = _unit_normal(TS3_TEMPLATE_SUBSTRATE_FRAME)
        actual_normal = _unit_normal(frame_points)
        self.assertGreater(float(np.dot(actual_normal, template_normal)), 0.95)
        self.assert_core_metrics_close(row)

    def assert_sane_ts4_row(self, row):
        roles = row["constraint_roles"]
        coords = np.array(row["coords_embedded"], dtype=float)
        mol = _row_to_mol(row, row["atoms"], row["coords_embedded"])
        cat_b = roles["cat_B"]
        pin_b = roles["pin_B"]
        substrate_c = roles["substrate_C"]
        cat_h = roles["cat_H"]
        transfer_h = roles["transfer_H"]

        cat_b_hydrogens = sorted(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(cat_b).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        pin_b_hydrogens = sorted(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(pin_b).GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        )
        self.assertEqual(cat_b_hydrogens, sorted([cat_h, transfer_h]))
        self.assertEqual(pin_b_hydrogens, [])
        self.assertTrue(_role_bond_exists(mol, roles, "cat_B", "transfer_H"))
        self.assertFalse(_role_bond_exists(mol, roles, "cat_B", "pin_B"))
        self.assertTrue(_role_bond_exists(mol, roles, "pin_B", "substrate_C"))
        self.assertFalse(_role_bond_exists(mol, roles, "cat_B", "substrate_C"))
        self.assertTrue(_has_distance_constraint(row, "cat_B", "pin_B"))

        frame_neighbors = _ordered_substrate_frame_neighbors(
            mol,
            substrate_c,
            {cat_b, pin_b},
        )
        frame_points = coords[[frame_neighbors[0], substrate_c, frame_neighbors[1]]]
        template_normal = _unit_normal(TS4_TEMPLATE_SUBSTRATE_FRAME)
        actual_normal = _unit_normal(frame_points)
        self.assertGreater(float(np.dot(actual_normal, template_normal)), 0.95)
        self.assert_core_metrics_close(row)

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

        self.assertGreaterEqual(inferred.GetNumBonds(), len(row["connectivity_bonds"]))
        self.assertEqual(plotted.GetNumBonds(), len(row["connectivity_bonds"]))

    def test_ts2_uses_template_substrate_frame_without_extra_boron_hydride(self):
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
        self.assert_sane_ts2_row(row)

    def test_ts2_screen_panel_has_sane_core_graphs(self):
        components = ft.screen.read(
            pd.DataFrame(
                {
                    "role": ["substrate", "substrate", "substrate", "substrate", "catalyst"],
                    "smiles": SCREEN_PANEL_SUBSTRATES + [CATALYST],
                }
            )
        )
        rows = ft.screen.create_ts_guesses(
            ft.screen.expand(components),
            ts_types=["TS2"],
            n_confs=1,
        )["TS2"]

        self.assertEqual(len(rows), 9)
        for _, row in rows.iterrows():
            self.assert_sane_ts2_row(row)

    def test_ts3_uses_tmp_hydride_topology_and_substrate_frame(self):
        components = ft.screen.read(
            pd.DataFrame(
                {
                    "role": ["substrate", "substrate", "substrate", "substrate", "catalyst"],
                    "smiles": SCREEN_PANEL_SUBSTRATES + [CATALYST],
                }
            )
        )
        rows = ft.screen.create_ts_guesses(
            ft.screen.expand(components),
            ts_types=["TS3"],
            n_confs=1,
        )["TS3"]

        self.assertEqual(len(rows), 9)
        for _, row in rows.iterrows():
            self.assert_sane_ts3_row(row)

    def test_ts4_uses_tmp_hydride_topology_and_substrate_frame(self):
        components = ft.screen.read(
            pd.DataFrame(
                {
                    "role": ["substrate", "substrate", "substrate", "substrate", "catalyst"],
                    "smiles": SCREEN_PANEL_SUBSTRATES + [CATALYST],
                }
            )
        )
        rows = ft.screen.create_ts_guesses(
            ft.screen.expand(components),
            ts_types=["TS4"],
            n_confs=1,
        )["TS4"]

        self.assertEqual(len(rows), 9)
        for _, row in rows.iterrows():
            self.assert_sane_ts4_row(row)

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
