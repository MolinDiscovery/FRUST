import unittest
from unittest.mock import patch

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from frust.stepper import Stepper


def _mol_with_conformer(smiles: str = "CCO") -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cid = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    if cid < 0:
        raise RuntimeError(f"Could not embed test molecule {smiles!r}")
    return mol


def _metadata(name: str, smiles: str = "CCO") -> dict:
    return {
        "structure_id": f"MOL:{name}:structure",
        "custom_name": name,
        "substrate_name": name,
        "structure_type": "MOL",
        "molecule_role": "structure",
        "rpos": pd.NA,
        "smiles": smiles,
        "input_smiles": smiles,
    }


class StepperBuildInitialDfTests(unittest.TestCase):
    def test_single_smiles_builds_plain_molecule_dataframe(self):
        step = Stepper(debug=True, save_output_dir=False)

        df = step.build_initial_df("CCO", name="ethanol")

        self.assertEqual(len(df), 1)
        self.assertEqual(df["substrate_name"].iloc[0], "ethanol")
        self.assertEqual(df["custom_name"].iloc[0], "ethanol")
        self.assertEqual(df["structure_type"].iloc[0], "MOL")
        self.assertEqual(df["molecule_role"].iloc[0], "structure")
        self.assertEqual(df["smiles"].iloc[0], "CCO")
        self.assertIn("O", df["atoms"].iloc[0])
        self.assertEqual(step.step_type, None)
        self.assertEqual(
            df.attrs["frust_initial_df"],
            {
                "input_kind": "smiles",
                "workflow": None,
                "n_confs": 1,
                "n_cores": 8,
                "optimization": "none",
                "max_iters": 100,
                "select_mols": None,
                "ts_type": None,
                "ts_optimize": None,
                "step_type": None,
                "resolved_step_type": None,
            },
        )

    def test_batch_smiles_inputs_use_expected_labels(self):
        step = Stepper(debug=True, save_output_dir=False)

        listed = step.build_initial_df(
            ["CCO", "CCN"],
            names=["ethanol", "ethylamine"],
        )
        single_list = step.build_initial_df(["CCO"], names=["ethanol"])
        named = step.build_initial_df({"ethanol": "CCO", "ethylamine": "CCN"})

        self.assertEqual(list(listed["substrate_name"]), ["ethanol", "ethylamine"])
        self.assertEqual(list(single_list["substrate_name"]), ["ethanol"])
        self.assertEqual(list(named["substrate_name"]), ["ethanol", "ethylamine"])

    def test_dataframe_smiles_input_uses_substrate_name_column(self):
        step = Stepper(debug=True, save_output_dir=False)
        inputs = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN"],
                "substrate_name": ["ethanol", "ethylamine"],
            }
        )

        df = step.build_initial_df(inputs)

        self.assertEqual(list(df["substrate_name"]), ["ethanol", "ethylamine"])
        self.assertEqual(list(df["input_smiles"]), ["CCO", "CCN"])

    def test_existing_embedded_dictionary_keeps_current_behavior(self):
        step = Stepper(debug=True, save_output_dir=False)
        mol = _mol_with_conformer()

        df = step.build_initial_df({"ethanol": (mol, [0])})

        self.assertEqual(len(df), 1)
        self.assertEqual(df["substrate_name"].iloc[0], "ethanol")
        self.assertEqual(df["structure_type"].iloc[0], "MOL")
        self.assertIsNone(df["smiles"].iloc[0])

    def test_raw_molecule_dictionary_is_embedded(self):
        step = Stepper(debug=True, save_output_dir=False)
        raw = {"ethanol": (Chem.MolFromSmiles("CCO"), _metadata("ethanol"))}

        df = step.build_initial_df(raw, n_confs=1)

        self.assertEqual(len(df), 1)
        self.assertEqual(df["substrate_name"].iloc[0], "ethanol")
        self.assertEqual(df["smiles"].iloc[0], "CCO")
        self.assertIn("coords_embedded", df.columns)

    def test_workflow_mols_expands_smiles_before_embedding(self):
        step = Stepper(debug=True, save_output_dir=False)
        raw = {"ethanol": (Chem.MolFromSmiles("CCO"), _metadata("ethanol"))}

        with patch("frust.utils.mols.create_mol_per_rpos", return_value=raw) as create:
            df = step.build_initial_df(
                "CCO",
                workflow="mols",
                select_mols="uniques",
                n_confs=1,
            )

        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["select_mols"], "uniques")
        self.assertEqual(df["substrate_name"].iloc[0], "ethanol")
        self.assertEqual(df.attrs["frust_initial_df"]["input_kind"], "workflow_mols")
        self.assertEqual(df.attrs["frust_initial_df"]["workflow"], "mols")
        self.assertEqual(df.attrs["frust_initial_df"]["select_mols"], "uniques")

    def test_raw_ts_dictionary_embeds_and_resolves_auto_step_type(self):
        step = Stepper(step_type="auto", debug=True, save_output_dir=False)
        raw_ts = {"TS1(phenol_rpos(2))": (Chem.MolFromSmiles("CCO"), [0, 1, 2, 3, 4, 5], "CCO")}
        embedded = {
            "TS1(phenol_rpos(2))": (
                _mol_with_conformer(),
                [0],
                [0, 1, 2, 3, 4, 5],
                "CCO",
                [],
            )
        }

        with patch("frust.embedder.embed_ts", return_value=embedded) as embed:
            df = step.build_initial_df(raw_ts, n_confs=1, ts_optimize=True)

        self.assertEqual(embed.call_args.kwargs["ts_type"], "TS1")
        self.assertTrue(embed.call_args.kwargs["optimize"])
        self.assertEqual(step.step_type, "TS1")
        self.assertEqual(df["structure_type"].iloc[0], "TS1")
        self.assertEqual(df.attrs["frust_initial_df"]["input_kind"], "raw_ts_dict")
        self.assertEqual(df.attrs["frust_initial_df"]["ts_type"], "TS1")
        self.assertTrue(df.attrs["frust_initial_df"]["ts_optimize"])
        self.assertEqual(df.attrs["frust_initial_df"]["step_type"], "auto")
        self.assertEqual(df.attrs["frust_initial_df"]["resolved_step_type"], "TS1")

    def test_existing_embedded_dictionary_records_initial_attrs(self):
        step = Stepper(debug=True, save_output_dir=False)
        mol = _mol_with_conformer()

        df = step.build_initial_df({"ethanol": (mol, [0])})

        self.assertEqual(df.attrs["frust_initial_df"]["input_kind"], "embedded_dict")
        self.assertIsNone(df.attrs["frust_initial_df"]["n_confs"])

    def test_explicit_step_type_mismatch_fails(self):
        step = Stepper(step_type="TS2", debug=True, save_output_dir=False)
        mol = _mol_with_conformer()
        embedded = {
            "TS1(phenol_rpos(2))": (
                mol,
                [0],
                [0, 1, 2, 3, 4, 5],
                "CCO",
                [],
            )
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            step.build_initial_df(embedded)

    def test_invalid_and_ambiguous_inputs_fail_clearly(self):
        step = Stepper(debug=True, save_output_dir=False)

        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            step.build_initial_df("not a smiles")

        with self.assertRaisesRegex(ValueError, "Mixed"):
            step.build_initial_df(
                {
                    "raw": Chem.MolFromSmiles("CCO"),
                    "embedded": (_mol_with_conformer(), [0]),
                }
            )

        with self.assertRaisesRegex(ValueError, "`name=`"):
            step.build_initial_df(["CCO", "CCN"], name="ethanol")


if __name__ == "__main__":
    unittest.main()
