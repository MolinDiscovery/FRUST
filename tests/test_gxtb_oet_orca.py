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


def _df_with_hessian() -> pd.DataFrame:
    df = _df()
    df["gxtb-hess-input.hess"] = ["$orca_hessian_file\n$end\n"]
    return df


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
            with patch.dict(os.environ, {"OET_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                block = gxtb_orca_block()

        self.assertIn('ProgExt "', block)
        self.assertIn("/bin/oet_gxtb", block)
        self.assertIn(f"Ext_Params \"--exe {gxtb.resolve()}\"", block)
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
            with patch.dict(os.environ, {"OET_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(_df(), options={"OptTS": None}, gxtb=True)

        self.assertEqual(calls[0]["options"], {"ExtOpt": None, "OptTS": None})
        self.assertIn("/bin/oet_gxtb", calls[0]["xtra_inp_str"])
        self.assertIn(f"--exe {gxtb.resolve()}", calls[0]["xtra_inp_str"])
        self.assertIn("orca-ExtOpt-OptTS-NT", out.columns)
        meta = out.attrs["frust_steps"]["orca-ExtOpt-OptTS"]
        self.assertTrue(meta["gxtb"])
        self.assertEqual(meta["gxtb_exe"], str(gxtb.resolve()))
        self.assertEqual(meta["gxtb_exe_source"], "GXTB_EXE")
        calc = meta["calculator"]
        self.assertEqual(calc["name"], "orca")
        self.assertEqual(calc["mode"], "orca_external_gxtb")
        self.assertEqual(calc["executables"]["gxtb"]["path"], str(gxtb.resolve()))
        self.assertEqual(calc["executables"]["gxtb"]["source"], "GXTB_EXE")
        self.assertEqual(calc["executables"]["oet_gxtb"]["path"], str((oet / "bin" / "oet_gxtb").resolve()))

    def test_orca_gxtb_attrs_record_explicit_exe_source(self):
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
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"OET_TOOLS": str(oet)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(
                    _df(),
                    options={"OptTS": None},
                    gxtb=True,
                    gxtb_exe=str(gxtb),
                )

        meta = out.attrs["frust_steps"]["orca-ExtOpt-OptTS"]
        self.assertEqual(meta["gxtb_exe"], str(gxtb.resolve()))
        self.assertEqual(meta["gxtb_exe_source"], "gxtb_exe")
        self.assertEqual(meta["calculator"]["executables"]["gxtb"]["path"], str(gxtb.resolve()))
        self.assertEqual(meta["calculator"]["executables"]["gxtb"]["source"], "gxtb_exe")

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

    def test_orca_gxtb_rejects_calc_hess_flag(self):
        step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
        with self.assertRaisesRegex(ValueError, "Calc_Hess"):
            step.orca(
                _df(),
                options={"OptTS": None},
                gxtb=True,
                calc_hess=True,
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
            with patch.dict(os.environ, {"OET_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(_df(), options={"OptTS": None, "NumFreq": None}, gxtb=True)

        self.assertEqual(calls[0], {"ExtOpt": None, "OptTS": None, "NumFreq": None})
        self.assertIn("orca-ExtOpt-OptTS-NumFreq-NT", out.columns)

    def test_save_step_preserves_engine_outputs_when_calc_dir_is_save_dir(self):
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
            calc_dir=None,
            save_dir=None,
        ):
            self.assertEqual(calc_dir, save_dir)
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"OET_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(
                    step_type="MOLS",
                    debug=True,
                    save_output_dir=True,
                    output_base=td,
                )
                step.orca_fn = fake_orca
                out = step.orca(_df(), options={"Opt": None}, gxtb=True, save_step=True)

        self.assertIn("orca-ExtOpt-Opt-EE", out.columns)
        self.assertIn("orca-ExtOpt-Opt-NT", out.columns)
        self.assertIn("orca-ExtOpt-Opt-oc", out.columns)
        self.assertTrue(out["orca-ExtOpt-Opt-NT"].iloc[0])

    def test_orca_gxtb_forwards_last_hessian_to_engine(self):
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
                    "data2file": data2file,
                    "xtra_inp_str": xtra_inp_str,
                    "options": options,
                }
            )
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            oet, gxtb = _fake_paths(Path(td))
            with patch.dict(os.environ, {"OET_TOOLS": str(oet), "GXTB_EXE": str(gxtb)}):
                step = Stepper(step_type="MOLS", debug=True, save_output_dir=False)
                step.orca_fn = fake_orca
                out = step.orca(
                    _df_with_hessian(),
                    options={"OptTS": None},
                    gxtb=True,
                    use_last_hess=True,
                )

        self.assertEqual(
            calls[0]["data2file"],
            {"private_input.hess": "$orca_hessian_file\n$end\n"},
        )
        self.assertIn("inhess Read", calls[0]["xtra_inp_str"])
        self.assertIn('InHessName "private_input.hess"', calls[0]["xtra_inp_str"])
        self.assertEqual(calls[0]["options"], {"ExtOpt": None, "OptTS": None})
        self.assertTrue(out["orca-ExtOpt-OptTS-NT"].iloc[0])


if __name__ == "__main__":
    unittest.main()
