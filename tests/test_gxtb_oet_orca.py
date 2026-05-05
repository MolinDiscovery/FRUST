import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from frust.stepper import Stepper
from frust.utils.gxtb import gxtb_orca_block


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atoms": [["H", "H"]],
            "coords_embedded": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]],
            "substrate_name": ["h2"],
        }
    )


def _fake_paths(root: Path) -> tuple[Path, Path]:
    oet = root / "oet"
    bin_dir = oet / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "oet_gxtb"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)

    gxtb = root / "gxtb-xtb"
    gxtb.write_text("#!/bin/sh\n")
    gxtb.chmod(0o755)
    return oet, gxtb


class GxtbOetOrcaTests(unittest.TestCase):
    def test_gxtb_orca_block_uses_oet_gxtb_and_exe(self):
        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"UMA_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                block = gxtb_orca_block()

        self.assertIn('ProgExt "', block)
        self.assertIn("/bin/oet_gxtb", block)
        self.assertIn(f"Ext_Params \"--exe {gxtb}\"", block)
        self.assertIn("Print[P_EXT_GRAD] 1", block)

    def test_orca_gxtb_injects_extopt_and_method_block(self):
        calls = []

        def fake_orca(
            atoms,
            coords,
            n_cores,
            scr,
            data2file,
            options,
            xtra_inp_str,
            memory,
            read_files,
        ):
            calls.append(
                {
                    "atoms": atoms,
                    "coords": coords,
                    "n_cores": n_cores,
                    "scr": scr,
                    "data2file": data2file,
                    "options": options,
                    "xtra_inp_str": xtra_inp_str,
                    "memory": memory,
                    "read_files": read_files,
                }
            )
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"UMA_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(_df(), options={"OptTS": None}, gxtb=True)

        self.assertEqual(calls[0]["options"], {"ExtOpt": None, "OptTS": None})
        self.assertIn("/bin/oet_gxtb", calls[0]["xtra_inp_str"])
        self.assertIn(f"--exe {gxtb}", calls[0]["xtra_inp_str"])
        self.assertIn("orca-ExtOpt-OptTS-NT", out.columns)
        self.assertTrue(out.attrs["frust_steps"]["orca-ExtOpt-OptTS"]["gxtb"])

    def test_orca_rejects_uma_and_gxtb_together(self):
        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
        with self.assertRaisesRegex(ValueError, "both UMA and g-xTB"):
            step.orca(_df(), options={"OptTS": None}, uma="omol", gxtb=True)

    def test_orca_gxtb_rejects_analytic_freq(self):
        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
        with self.assertRaisesRegex(ValueError, "Use NumFreq"):
            step.orca(_df(), options={"OptTS": None, "Freq": None}, gxtb=True)

    def test_orca_gxtb_rejects_calc_hess_block(self):
        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
        with self.assertRaisesRegex(ValueError, "Calc_Hess"):
            step.orca(
                _df(),
                options={"OptTS": None},
                gxtb=True,
                xtra_inp_str="""
%geom
  Calc_Hess true
end
""",
            )

    def test_orca_gxtb_allows_numfreq(self):
        calls = []

        def fake_orca(
            atoms,
            coords,
            n_cores,
            scr,
            data2file,
            options,
            xtra_inp_str,
            memory,
            read_files,
        ):
            calls.append(options)
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"UMA_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(_df(), options={"OptTS": None, "NumFreq": None}, gxtb=True)

        self.assertEqual(calls[0], {"ExtOpt": None, "OptTS": None, "NumFreq": None})
        self.assertIn("orca-ExtOpt-OptTS-NumFreq-NT", out.columns)


if __name__ == "__main__":
    unittest.main()
