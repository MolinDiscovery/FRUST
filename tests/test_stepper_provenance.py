import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from frust.stepper import Stepper


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atoms": [["H", "H"]],
            "coords_embedded": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]],
            "substrate_name": ["h2"],
        }
    )


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def _fake_orca(
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


def _fake_oet_root(root: Path) -> Path:
    oet = root / "oet"
    bin_dir = oet / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("oet_client", "oet_uma", "oet_server", "oet_gxtb"):
        _executable(bin_dir / name)
    return oet


class StepperProvenanceTests(unittest.TestCase):
    def test_xtb_records_calculator_metadata(self):
        def fake_xtb(atoms, coords, n_cores, scr, data2file, options):
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            xtb = _executable(Path(td) / "xtb")
            with patch.dict(os.environ, {"XTB_EXE": str(xtb)}):
                step = Stepper(debug=True, save_output_dir=False, n_cores=8)
                step.xtb_fn = fake_xtb
                out = step.xtb(_df(), name="xtb_test", options={"gfn": 2}, n_cores=3)

        calc = out.attrs["frust_steps"]["xtb_test"]["calculator"]
        self.assertEqual(calc["name"], "xtb")
        self.assertEqual(calc["mode"], "direct")
        self.assertEqual(calc["backend"], f"{__name__}.StepperProvenanceTests.test_xtb_records_calculator_metadata.<locals>.fake_xtb")
        self.assertEqual(calc["resources"], {"n_cores": 3})
        self.assertEqual(calc["executables"]["xtb"]["path"], str(xtb.resolve()))
        self.assertEqual(calc["executables"]["xtb"]["source"], "XTB_EXE")

    def test_direct_gxtb_records_calculator_metadata(self):
        def fake_gxtb(atoms, coords, n_cores, scr, data2file, options):
            return {
                "normal_termination": True,
                "electronic_energy": -1.0,
                "opt_coords": coords,
            }

        with tempfile.TemporaryDirectory() as td:
            gxtb = _executable(Path(td) / "gxtb-xtb")
            with patch.dict(os.environ, {"GXTB_EXE": str(gxtb)}):
                step = Stepper(debug=True, save_output_dir=False, n_cores=8)
                step.gxtb_fn = fake_gxtb
                out = step.gxtb(_df(), name="gxtb_test", options={"opt": None}, n_cores=4)

        calc = out.attrs["frust_steps"]["gxtb_test"]["calculator"]
        self.assertEqual(calc["name"], "gxtb")
        self.assertEqual(calc["mode"], "direct_gxtb")
        self.assertEqual(calc["resources"], {"n_cores": 4})
        self.assertEqual(calc["executables"]["gxtb"]["path"], str(gxtb.resolve()))
        self.assertEqual(calc["executables"]["gxtb"]["source"], "GXTB_EXE")

    def test_plain_orca_records_calculator_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orca = _executable(root / "orca")
            xtb = _executable(root / "xtb")
            mpi = root / "openmpi"
            xtbpath = root / "xtbpath"
            mpi.mkdir()
            xtbpath.mkdir()
            env = {
                "ORCA_EXE": str(orca),
                "XTB_EXE": str(xtb),
                "OPEN_MPI_DIR": str(mpi),
                "XTBPATH": str(xtbpath),
            }
            with patch.dict(os.environ, env):
                step = Stepper(debug=True, save_output_dir=False, n_cores=8, memory_gb=12)
                step.orca_fn = _fake_orca
                out = step.orca(
                    _df(),
                    name="orca_test",
                    options={"HF": None, "STO-3G": None, "SP": None},
                    n_cores=2,
                )

        calc = out.attrs["frust_steps"]["orca_test"]["calculator"]
        self.assertEqual(calc["name"], "orca")
        self.assertEqual(calc["mode"], "direct")
        self.assertEqual(calc["resources"], {"n_cores": 2, "memory_gb": 12})
        self.assertEqual(calc["executables"]["orca"]["path"], str(orca.resolve()))
        self.assertEqual(calc["executables"]["xtb"]["path"], str(xtb.resolve()))
        self.assertEqual(calc["environment"]["OPEN_MPI_DIR"]["path"], str(mpi.resolve()))
        self.assertEqual(calc["environment"]["XTBPATH"]["path"], str(xtbpath.resolve()))

    def test_orca_uma_standalone_records_calculator_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oet = _fake_oet_root(root)
            orca = _executable(root / "orca")
            env_file = root / "frust.env"
            env_file.write_text(f"OET_TOOLS={oet}\nORCA_EXE={orca}\n")
            with patch.dict(
                os.environ,
                {
                    "OET_TOOLS": str(oet),
                    "ORCA_EXE": str(orca),
                    "TOOLTOAD_DOTENV_PATH": str(env_file),
                },
            ):
                step = Stepper(debug=True, save_output_dir=False)
                step.orca_fn = _fake_orca
                out = step.orca(
                    _df(),
                    name="uma_test",
                    options={"ExtOpt": None, "Opt": None},
                    uma="omol@uma-s-1p1",
                    uma_server=False,
                )

        calc = out.attrs["frust_steps"]["uma_test"]["calculator"]
        self.assertEqual(calc["mode"], "orca_external_uma")
        self.assertEqual(calc["uma"]["task"], "omol")
        self.assertEqual(calc["uma"]["model"], "uma-s-1p1")
        self.assertFalse(calc["uma"]["server"])
        self.assertEqual(calc["executables"]["oet_uma"]["path"], str((oet / "bin" / "oet_uma").resolve()))

    def test_orca_uma_server_records_calculator_metadata(self):
        @contextmanager
        def fake_uma_server(**kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                bind="127.0.0.1:12345",
                preserve=lambda: "preserved.log",
            )

        calls = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oet = _fake_oet_root(root)
            orca = _executable(root / "orca")
            env_file = root / "frust.env"
            env_file.write_text(f"OET_TOOLS={oet}\nORCA_EXE={orca}\n")
            with patch.dict(
                os.environ,
                {
                    "OET_TOOLS": str(oet),
                    "ORCA_EXE": str(orca),
                    "TOOLTOAD_DOTENV_PATH": str(env_file),
                },
            ):
                with patch("frust.utils.uma.uma_server", fake_uma_server):
                    step = Stepper(debug=True, save_output_dir=False)
                    step.orca_fn = _fake_orca
                    out = step.orca(
                        _df(),
                        name="uma_server_test",
                        options={"ExtOpt": None, "Opt": None},
                        uma="omol",
                        uma_server=True,
                        uma_server_cores=5,
                        uma_memory_per_thread_mib=700,
                    )

        calc = out.attrs["frust_steps"]["uma_server_test"]["calculator"]
        self.assertEqual(calls[0]["server_cores"], 5)
        self.assertEqual(calc["mode"], "orca_external_uma")
        self.assertTrue(calc["uma"]["server"])
        self.assertEqual(calc["resources"]["uma_server_cores"], 5)
        self.assertEqual(calc["resources"]["uma_memory_per_thread_mib"], 700)
        self.assertEqual(calc["executables"]["oet_client"]["path"], str((oet / "bin" / "oet_client").resolve()))
        self.assertEqual(calc["executables"]["oet_server"]["path"], str((oet / "bin" / "oet_server").resolve()))


if __name__ == "__main__":
    unittest.main()
