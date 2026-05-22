import unittest

import pandas as pd

from frust.utils import show_steps, lowest_energy_rows


class DataFrameUtilityTests(unittest.TestCase):
    def test_show_steps_summarizes_step_attrs(self):
        df = pd.DataFrame({"substrate_name": ["ethanol"]})
        df.attrs["frust_steps"] = {
            "xtb_opt": {
                "engine": "xtb",
                "columns": ["xtb_opt-EE", "xtb_opt-NT", "xtb_opt-oc"],
                "options": {"gfn": 2, "opt": None},
                "calculator": {
                    "name": "xtb",
                    "mode": "direct",
                    "backend": "tooltoad.xtb.xtb_calculate",
                    "resources": {"n_cores": 4},
                    "executables": {
                        "xtb": {
                            "path": "/cluster/apps/xtb/bin/xtb",
                            "configured": "xtb",
                            "source": "PATH",
                            "resolved": True,
                        }
                    },
                    "environment": {
                        "XTBPATH": {
                            "path": "/cluster/apps/xtb/share/xtb",
                            "configured": "/cluster/apps/xtb/share/xtb",
                            "source": "XTBPATH",
                            "resolved": True,
                        }
                    },
                },
            },
            "gxtb_opt": {
                "engine": "orca",
                "columns": ["gxtb_opt-EE"],
                "options": {"ExtOpt": None, "Opt": None},
                "gxtb": True,
                "gxtb_exe": "/cluster/apps/g-xtb/bin/xtb",
                "gxtb_exe_source": "GXTB_EXE",
                "input": {
                    "options": {"ExtOpt": None, "Opt": None},
                    "xtra_inp_str": '%CPCM\nSMD TRUE\nSMDSOLVENT "chloroform"\nend',
                    "constraint": False,
                },
                "calculator": {
                    "name": "orca",
                    "mode": "orca_external_gxtb",
                    "backend": "tooltoad.orca.orca_calculate",
                    "resources": {"n_cores": 8, "memory_gb": 20.0},
                    "executables": {
                        "gxtb": {
                            "path": "/cluster/apps/g-xtb/bin/xtb",
                            "configured": "/cluster/apps/g-xtb/bin/xtb",
                            "source": "GXTB_EXE",
                            "resolved": True,
                        },
                        "oet_gxtb": {
                            "path": None,
                            "configured": "/missing/oet_gxtb",
                            "source": "OET_TOOLS",
                            "resolved": False,
                        },
                    },
                },
            },
        }

        out = show_steps(df)

        self.assertEqual(list(out.index), ["xtb_opt", "gxtb_opt"])
        self.assertEqual(out.index.name, "step")
        self.assertNotIn("executables", out.columns)
        self.assertNotIn("environment", out.columns)
        self.assertNotIn("gxtb_exe_source", out.columns)
        self.assertEqual(out.loc["xtb_opt", "engine"], "xtb")
        self.assertEqual(out.loc["xtb_opt", "calc_name"], "xtb")
        self.assertEqual(out.loc["xtb_opt", "mode"], "direct")
        self.assertEqual(out.loc["xtb_opt", "options"], "gfn opt")
        self.assertEqual(out.loc["xtb_opt", "columns"], "xtb_opt-EE, xtb_opt-NT, xtb_opt-oc")
        self.assertEqual(out.loc["xtb_opt", "n_columns"], 3)
        self.assertEqual(out.loc["xtb_opt", "n_cores"], 4)
        self.assertEqual(out.loc["gxtb_opt", "memory_gb"], 20.0)
        self.assertIn("xtra_inp_str", out.columns)
        self.assertIn("SMDSOLVENT", out.loc["gxtb_opt", "xtra_inp_str"])

        full = show_steps(df, detail="full")

        self.assertIn("xtb: /cluster/apps/xtb/bin/xtb", full.loc["xtb_opt", "executables"])
        self.assertIn("configured=xtb", full.loc["xtb_opt", "executables"])
        self.assertIn("XTBPATH: /cluster/apps/xtb/share/xtb", full.loc["xtb_opt", "environment"])
        self.assertTrue(full.loc["gxtb_opt", "gxtb"])
        self.assertEqual(full.loc["gxtb_opt", "gxtb_exe_source"], "GXTB_EXE")
        self.assertIn("xtra_inp_str:", full.loc["gxtb_opt", "input"])
        self.assertIn("resolved=False", full.loc["gxtb_opt", "executables"])

    def test_show_steps_rejects_unknown_detail(self):
        with self.assertRaisesRegex(ValueError, "detail"):
            show_steps(pd.DataFrame(), detail="verbose")

    def test_show_steps_returns_empty_dataframe_without_step_attrs(self):
        out = show_steps(pd.DataFrame({"x": [1]}))

        self.assertTrue(out.empty)
        self.assertEqual(out.index.name, "step")

    def test_lowest_energy_rows_matches_stepper_grouping_defaults(self):
        df = pd.DataFrame(
            {
                "ligand_name": ["anisole", "anisole", "phenol", "phenol"],
                "rpos": [None, None, None, None],
                "cid": [0, 1, 0, 1],
                "xtb-EE": [-1.0, -2.0, -10.0, -9.0],
                "dft-electronic_energy": [-100.0, -101.0, -200.0, -199.0],
            },
            index=[10, 11, 20, 21],
        )
        df.attrs["note"] = "preserved"

        out = lowest_energy_rows(df)

        self.assertEqual(list(out.index), [11, 20])
        self.assertIn("substrate_name", out.columns)
        self.assertNotIn("ligand_name", out.columns)
        self.assertEqual(list(out["substrate_name"]), ["anisole", "phenol"])
        self.assertEqual(list(out["cid"]), [1, 0])
        self.assertEqual(out.attrs["note"], "preserved")

    def test_lowest_energy_rows_can_keep_n_per_group_and_choose_energy(self):
        df = pd.DataFrame(
            {
                "substrate_name": ["a", "a", "a", "b", "b"],
                "rpos": [1, 1, 1, 1, 1],
                "cid": [0, 1, 2, 0, 1],
                "cheap-EE": [0.0, -5.0, -1.0, -2.0, -3.0],
                "final-EE": [-1.0, -2.0, -3.0, -4.0, -5.0],
            }
        )

        out = lowest_energy_rows(df, n=2, energy_col="cheap-EE")

        self.assertEqual(list(out["substrate_name"]), ["a", "a", "b", "b"])
        self.assertEqual(list(out["cid"]), [1, 2, 1, 0])

    def test_lowest_energy_rows_accepts_legacy_explicit_columns(self):
        df = pd.DataFrame(
            {
                "ligand_name": ["a", "a", "b", "b"],
                "cid": [0, 1, 0, 1],
                "stage-electronic_energy": [-1.0, -2.0, -3.0, -4.0],
            }
        )

        out = lowest_energy_rows(
            df,
            energy_col="stage-electronic_energy",
            group_cols=["ligand_name"],
        )

        self.assertEqual(list(out["substrate_name"]), ["a", "b"])
        self.assertEqual(list(out["cid"]), [1, 1])
        self.assertIn("stage-EE", out.columns)

    def test_lowest_energy_rows_validates_inputs(self):
        with self.assertRaisesRegex(ValueError, "n must be"):
            lowest_energy_rows(pd.DataFrame(), n=0)

        with self.assertRaisesRegex(ValueError, "no energy column"):
            lowest_energy_rows(pd.DataFrame({"substrate_name": ["x"]}))

        with self.assertRaisesRegex(ValueError, "missing group columns"):
            lowest_energy_rows(
                pd.DataFrame({"substrate_name": ["x"], "stage-EE": [-1.0]}),
                group_cols=["missing"],
            )


if __name__ == "__main__":
    unittest.main()
