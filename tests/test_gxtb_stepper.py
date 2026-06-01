import os
import subprocess
import unittest

import pandas as pd

from frust.stepper import Stepper
from frust.utils import show_steps


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atoms": [["H", "H"], ["H", "H"]],
            "coords_embedded": [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
            ],
            "substrate_name": ["mol", "mol"],
            "cid": [10, 11],
            "prev-EE": [-2.0, -1.0],
        }
    )


class StepperGxtbTests(unittest.TestCase):
    def test_gxtb_uses_dedicated_backend_and_shared_inputs(self):
        calls = []

        def fake_gxtb(atoms, coords, n_cores, scr, data2file, options, detailed_input_str):
            kwargs = {
                "atoms": atoms,
                "coords": coords,
                "n_cores": n_cores,
                "scr": scr,
                "data2file": data2file,
                "options": options,
                "detailed_input_str": detailed_input_str,
            }
            calls.append(kwargs)
            return {
                "normal_termination": True,
                "electronic_energy": -1.23,
                "opt_coords": coords,
            }

        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False, n_cores=8)
        step.gxtb_fn = fake_gxtb

        out = step.gxtb(
            _df().head(1),
            options={"opt": None},
            detailed_inp_str="$test\n$end",
            n_cores=3,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["options"], {"opt": None})
        self.assertEqual(calls[0]["n_cores"], 3)
        self.assertIn("$test", calls[0]["detailed_input_str"])
        self.assertIn("gxtb-opt-NT", out.columns)
        self.assertIn("gxtb-opt-EE", out.columns)
        self.assertIn("gxtb-opt-oc", out.columns)
        self.assertEqual(out.attrs["frust_steps"]["gxtb-opt"]["engine"], "gxtb")

    def test_gxtb_lowest_and_failure_columns(self):
        calls = []

        def fake_gxtb(atoms, coords, n_cores, scr, data2file, options):
            calls.append(
                {
                    "atoms": atoms,
                    "coords": coords,
                    "n_cores": n_cores,
                    "scr": scr,
                    "data2file": data2file,
                    "options": options,
                }
            )
            raise RuntimeError("boom")

        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
        step.gxtb_fn = fake_gxtb

        out = step.gxtb(_df(), options={"grad": None}, lowest=1)

        self.assertEqual(len(calls), 1)
        self.assertFalse(out["gxtb-NT"].iloc[0])
        self.assertIn("RuntimeError: boom", out["gxtb-error"].iloc[0])
        row_counts = out.attrs["frust_steps"]["gxtb"]["row_counts"]
        self.assertEqual(row_counts["input_rows"], 2)
        self.assertEqual(row_counts["output_rows"], 1)
        self.assertEqual(row_counts["dropped_rows"], 1)
        filtering = out.attrs["frust_steps"]["gxtb"]["filtering"]
        self.assertEqual(filtering["lowest"], 1)
        self.assertEqual(filtering["energy_col"], "prev-EE")
        self.assertEqual(filtering["input_rows"], 2)
        self.assertEqual(filtering["output_rows"], 1)
        self.assertEqual(filtering["dropped_rows"], 1)
        self.assertEqual(filtering["groups"][0]["selected_cids"], [10])
        steps = show_steps(out)
        self.assertEqual(steps.loc["gxtb", "lowest"], 1)
        self.assertEqual(steps.loc["gxtb", "filter_energy_col"], "prev-EE")
        self.assertEqual(steps.loc["gxtb", "dropped_rows"], 1)

    def test_gxtb_constraints_are_forwarded_as_detailed_input(self):
        calls = []

        def fake_gxtb(atoms, coords, n_cores, scr, data2file, options, detailed_input_str):
            calls.append(
                {
                    "atoms": atoms,
                    "coords": coords,
                    "n_cores": n_cores,
                    "scr": scr,
                    "data2file": data2file,
                    "options": options,
                    "detailed_input_str": detailed_input_str,
                }
            )
            return {"normal_termination": True, "electronic_energy": -1.0}

        df = _df().head(1).copy()
        df["constraint_atoms"] = [[0, 1, 0, 1, 0, 1]]
        step = Stepper(step_type="TS1", debug=True, save_output_dir=False)
        step.gxtb_fn = fake_gxtb

        step.gxtb(df, options={"grad": None}, constraint=True)

        self.assertIn("$constrain", calls[0]["detailed_input_str"])
        self.assertIn("force constant=50", calls[0]["detailed_input_str"])

    def test_gxtb_smoke_is_skipped_without_gxtb_exe(self):
        gxtb_exe = os.environ.get("GXTB_EXE")
        if not gxtb_exe:
            self.skipTest("GXTB_EXE is not configured")
        help_out = subprocess.run(
            [gxtb_exe, "--help"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "--gxtb" not in help_out:
            self.skipTest("Configured GXTB_EXE does not advertise --gxtb")


if __name__ == "__main__":
    unittest.main()
