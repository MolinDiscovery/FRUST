import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from frust.utils import show_steps, lowest_energy_rows, map_substrate_names
from frust.utils.analytics import merge_parquet_dir
from frust.utils.dataframes import merge_dataframe_attrs


class DataFrameUtilityTests(unittest.TestCase):
    def test_show_steps_summarizes_step_attrs(self):
        df = pd.DataFrame({"substrate_name": ["ethanol"]})
        df.attrs["frust_steps"] = {
            "xtb_opt": {
                "engine": "xtb",
                "columns": ["xtb_opt-EE", "xtb_opt-NT", "xtb_opt-oc"],
                "options": {"gfn": 2, "opt": None},
                "row_counts": {
                    "input_rows": 50,
                    "output_rows": 10,
                    "dropped_rows": 40,
                },
                "filtering": {
                    "lowest": 10,
                    "energy_col": "xtb_sp-EE",
                    "input_rows": 50,
                    "output_rows": 10,
                    "dropped_rows": 40,
                    "n_groups": 1,
                },
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
                "row_counts": {
                    "input_rows": 10,
                    "output_rows": 10,
                    "dropped_rows": 0,
                },
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
        self.assertEqual(out.loc["xtb_opt", "lowest"], 10)
        self.assertEqual(out.loc["xtb_opt", "filter_energy_col"], "xtb_sp-EE")
        self.assertEqual(out.loc["xtb_opt", "input_rows"], 50)
        self.assertEqual(out.loc["xtb_opt", "output_rows"], 10)
        self.assertEqual(out.loc["xtb_opt", "dropped_rows"], 40)
        self.assertEqual(out.loc["xtb_opt", "n_cores"], 4)
        self.assertEqual(out.loc["gxtb_opt", "memory_gb"], 20.0)
        self.assertEqual(out.loc["gxtb_opt", "input_rows"], 10)
        self.assertEqual(out.loc["gxtb_opt", "output_rows"], 10)
        self.assertEqual(out.loc["gxtb_opt", "dropped_rows"], 0)
        self.assertIn("xtra_inp_str", out.columns)
        self.assertEqual(out.loc["gxtb_opt", "xtra_inp_str"], "%CPCM ... (4 lines)")
        self.assertNotIn("\n", out.loc["gxtb_opt", "xtra_inp_str"])

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

    def test_show_steps_includes_initial_conformer_generation_attrs(self):
        df = pd.DataFrame({"x": [1]})
        df.attrs["frust_conformers"] = {
            "schema_version": 1,
            "source": "screen.create_ts_guesses",
            "n_cores": 8,
            "structures": [
                {
                    "structure_id": "TS1:furan__cat:r0",
                    "requested_n_confs": 5,
                    "resolved_n_confs": 5,
                    "generated_n_confs": 4,
                    "cids": [0, 1, 2, 3],
                },
                {
                    "structure_id": "TS1:furan__cat:r1",
                    "requested_n_confs": 5,
                    "resolved_n_confs": 5,
                    "generated_n_confs": 5,
                    "cids": [0, 1, 2, 3, 4],
                },
            ],
        }

        out = show_steps(df)

        self.assertEqual(list(out.index), ["initial_conformers"])
        self.assertEqual(out.loc["initial_conformers", "engine"], "rdkit")
        self.assertEqual(out.loc["initial_conformers", "calc_name"], "screen.create_ts_guesses")
        self.assertEqual(
            out.loc["initial_conformers", "options"],
            "structures=2 requested=5 resolved=5 generated=9 missing=1",
        )
        self.assertNotIn("n_structures", out.columns)
        self.assertNotIn("n_confs_requested", out.columns)
        self.assertNotIn("n_confs_resolved", out.columns)
        self.assertNotIn("n_confs_generated", out.columns)
        self.assertNotIn("n_confs_missing", out.columns)
        self.assertEqual(out.loc["initial_conformers", "output_rows"], 9)
        self.assertEqual(out.loc["initial_conformers", "n_cores"], 8)

    def test_show_steps_summary_collapses_merged_step_variants(self):
        df = pd.DataFrame({"x": [1]})
        columns = ["DFT-pre-Opt-EE", "DFT-pre-Opt-NT", "DFT-pre-Opt-oc"]
        options = {
            "r2SCAN-3c": None,
            "TightSCF": None,
            "SlowConv": None,
            "Opt": None,
            "NoSym": None,
        }

        def step_meta(label, *, source_files, input_rows=20, output_rows=1, dropped_rows=19):
            return {
                "engine": "orca",
                "columns": columns,
                "options": options,
                "row_counts": {
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "dropped_rows": dropped_rows,
                },
                "filtering": {
                    "lowest": 1,
                    "energy_col": "DFT-pre-SP-EE",
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "dropped_rows": dropped_rows,
                },
                "input": {
                    "xtra_inp_str": f"{label}\nend",
                    "constraint": True,
                },
                "calculator": {
                    "name": "orca",
                    "mode": "direct",
                    "resources": {"n_cores": 24, "memory_gb": 20},
                },
                "source_files": source_files,
            }

        df.attrs["frust_steps"] = {
            "DFT-pre-Opt": step_meta("A", source_files=["a.parquet", "b.parquet"]),
            "DFT-pre-Opt__variant_001": step_meta("B", source_files=["c.parquet"]),
            "DFT-pre-Opt__variant_002": step_meta(
                "C",
                source_files=["d.parquet"],
                input_rows=10,
                output_rows=1,
                dropped_rows=9,
            ),
        }

        summary = show_steps(df)

        self.assertEqual(list(summary.index), ["DFT-pre-Opt"])
        self.assertEqual(summary.loc["DFT-pre-Opt", "n_variants"], 3)
        self.assertEqual(summary.loc["DFT-pre-Opt", "n_sources"], 4)
        self.assertEqual(summary.loc["DFT-pre-Opt", "input_rows"], 70)
        self.assertEqual(summary.loc["DFT-pre-Opt", "output_rows"], 4)
        self.assertEqual(summary.loc["DFT-pre-Opt", "dropped_rows"], 66)
        self.assertEqual(summary.loc["DFT-pre-Opt", "n_cores"], 24)
        self.assertEqual(summary.loc["DFT-pre-Opt", "memory_gb"], 20)
        self.assertEqual(
            summary.loc["DFT-pre-Opt", "xtra_inp_str"],
            "mixed: A ... (2 lines); B ... (2 lines); C ... (2 lines)",
        )
        self.assertNotIn("\n", summary.loc["DFT-pre-Opt", "xtra_inp_str"])

        full = show_steps(df, detail="full")

        self.assertEqual(
            list(full.index),
            ["DFT-pre-Opt", "DFT-pre-Opt__variant_001", "DFT-pre-Opt__variant_002"],
        )
        self.assertEqual(full.loc["DFT-pre-Opt", "xtra_inp_str"], "A\nend")
        self.assertEqual(full.loc["DFT-pre-Opt", "source_files"], "a.parquet, b.parquet")

    def test_merge_dataframe_attrs_preserves_steps_and_namespaces_conflicts(self):
        first = pd.DataFrame({"x": [1]})
        first.attrs["frust_steps"] = {
            "DFT": {
                "engine": "orca",
                "columns": ["DFT-EE"],
                "options": {"wB97X-D3": None},
            }
        }
        first.attrs["note"] = "same"
        first.attrs["method"] = "wB97X-D3"
        first.attrs["frust_conformers"] = {
            "schema_version": 1,
            "source": "screen.create_ts_guesses",
            "requested_n_confs": 2,
            "structures": [
                {
                    "structure_id": "TS1:a:r0",
                    "generated_n_confs": 2,
                    "cids": [0, 1],
                }
            ],
        }

        second = pd.DataFrame({"x": [2]})
        second.attrs["frust_steps"] = {
            "DFT": {
                "engine": "orca",
                "columns": ["DFT-EE"],
                "options": {"r2SCAN-3c": None},
            },
            "xtb": {
                "engine": "xtb",
                "columns": ["xtb-EE"],
                "options": {"gfn": 2},
            },
        }
        second.attrs["note"] = "same"
        second.attrs["method"] = "r2SCAN-3c"
        second.attrs["frust_conformers"] = {
            "schema_version": 1,
            "source": "screen.create_ts_guesses",
            "requested_n_confs": 2,
            "structures": [
                {
                    "structure_id": "TS1:b:r0",
                    "generated_n_confs": 1,
                    "cids": [0],
                }
            ],
        }

        attrs = merge_dataframe_attrs(
            [first, second],
            source_files=["first.parquet", "second.parquet"],
        )

        self.assertIn("DFT", attrs["frust_steps"])
        self.assertIn("DFT__variant_001", attrs["frust_steps"])
        self.assertIn("xtb", attrs["frust_steps"])
        self.assertEqual(attrs["note"], "same")
        self.assertNotIn("method", attrs)
        self.assertEqual(
            attrs["frust_steps"]["DFT"]["source_files"],
            ["first.parquet"],
        )
        self.assertEqual(
            attrs["frust_steps"]["DFT__variant_001"]["source_files"],
            ["second.parquet"],
        )
        self.assertIn("DFT", attrs["frust_merge"]["step_variants"])
        self.assertIn("method", attrs["frust_merge"]["attr_conflicts"])
        self.assertEqual(attrs["frust_conformers"]["total_generated_confs"], 3)
        self.assertEqual(attrs["frust_conformers"]["n_structures"], 2)
        self.assertEqual(
            attrs["frust_conformers"]["structures"][0]["source_files"],
            ["first.parquet"],
        )

    def test_merge_dataframe_attrs_ignores_existing_step_source_files(self):
        first = pd.DataFrame({"x": [1]})
        first.attrs["frust_steps"] = {
            "DFT": {
                "engine": "orca",
                "columns": ["DFT-EE"],
                "source_files": ["original.parquet"],
            }
        }
        second = pd.DataFrame({"x": [2]})
        second.attrs["frust_steps"] = {
            "DFT": {
                "engine": "orca",
                "columns": ["DFT-EE"],
            }
        }

        attrs = merge_dataframe_attrs(
            [first, second],
            source_files=["merged.parquet", "second.parquet"],
        )

        self.assertEqual(list(attrs["frust_steps"]), ["DFT"])
        self.assertEqual(
            attrs["frust_steps"]["DFT"]["source_files"],
            ["original.parquet", "second.parquet"],
        )

    def test_merge_parquet_dir_round_trips_attrs_for_show_steps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(2):
                df = pd.DataFrame({"x": [idx], "DFT-NT": [True]})
                df.attrs["frust_steps"] = {
                    "DFT": {
                        "engine": "orca",
                        "columns": ["DFT-EE", "DFT-NT"],
                        "options": {"wB97X-D3": None},
                    }
                }
                df.attrs["note"] = "preserved"
                df.to_parquet(root / f"part_{idx}.parquet")

            output = root / "merged.parquet"
            merge_parquet_dir(root, output=output)
            merged = pd.read_parquet(output)

        steps = show_steps(merged)
        self.assertEqual(list(steps.index), ["DFT"])
        self.assertEqual(steps.loc["DFT", "engine"], "orca")
        self.assertEqual(merged.attrs["note"], "preserved")
        self.assertEqual(merged.attrs["frust_merge"]["n_merged_files"], 2)

    def test_merge_parquet_dir_records_conflicts_and_skipped_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = pd.DataFrame({"x": [1], "DFT-NT": [True]})
            first.attrs["frust_steps"] = {
                "DFT": {
                    "engine": "orca",
                    "columns": ["DFT-EE"],
                    "options": {"wB97X-D3": None},
                }
            }
            first.attrs["method"] = "wB97X-D3"
            first.to_parquet(root / "first.parquet")

            second = pd.DataFrame({"x": [2], "DFT-NT": [True]})
            second.attrs["frust_steps"] = {
                "DFT": {
                    "engine": "orca",
                    "columns": ["DFT-EE"],
                    "options": {"r2SCAN-3c": None},
                }
            }
            second.attrs["method"] = "r2SCAN-3c"
            second.to_parquet(root / "second.parquet")

            skipped = pd.DataFrame({"x": [3], "DFT-NT": [False]})
            skipped.attrs["frust_steps"] = {
                "DFT": {
                    "engine": "orca",
                    "columns": ["DFT-EE"],
                    "options": {"B3LYP": None},
                }
            }
            skipped.attrs["method"] = "B3LYP"
            skipped.to_parquet(root / "skipped.parquet")

            output = root / "merged.parquet"
            merge_parquet_dir(
                root,
                output=output,
                require_normal_termination=True,
            )
            merged = pd.read_parquet(output)

        self.assertEqual(set(merged["x"]), {1, 2})
        self.assertIn("DFT", merged.attrs["frust_steps"])
        self.assertIn("DFT__variant_001", merged.attrs["frust_steps"])
        self.assertNotIn("method", merged.attrs)
        self.assertIn("method", merged.attrs["frust_merge"]["attr_conflicts"])
        self.assertEqual(merged.attrs["frust_merge"]["n_skipped_files"], 1)
        self.assertTrue(
            any(
                str(path).endswith("skipped.parquet")
                for path in merged.attrs["frust_merge"]["skipped_files"]
            )
        )

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

    def test_map_substrate_names_replaces_names_with_lean_default(self):
        df = pd.DataFrame(
            {
                "substrate_smiles": ["CN1C=CC=C1", "CN1C=CC=C1"],
                "substrate_name": ["substrate_000", "substrate_000"],
                "rpos": [0, 1],
            }
        )
        df.attrs["note"] = "preserved"
        calls = []

        def fake_lookup(smiles):
            calls.append(smiles)
            return {
                "input_smiles": smiles,
                "canonical_smiles": "Cn1cccc1",
                "pubchem_iupac": "1 methyl pyrrole",
                "pubchem_cid": 12345,
                "lookup_status": "success",
                "lookup_error": None,
            }

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "pubchem.csv"
            with patch("frust.utils.mols.lookup_pubchem_name", side_effect=fake_lookup):
                out = map_substrate_names(df, cache_path=cache_path)

            verbose = map_substrate_names(
                df,
                cache_path=cache_path,
                add_metadata=True,
                original_col="substrate_name_original",
            )

        self.assertEqual(calls, ["CN1C=CC=C1"])
        self.assertEqual(list(out["substrate_name"]), ["1_methyl_pyrrole", "1_methyl_pyrrole"])
        self.assertNotIn("substrate_name_original", out.columns)
        self.assertNotIn("substrate_pubchem_iupac", out.columns)
        self.assertNotIn("substrate_pubchem_cid", out.columns)
        self.assertNotIn("substrate_pubchem_status", out.columns)
        self.assertEqual(out.attrs["note"], "preserved")
        self.assertEqual(list(df["substrate_name"]), ["substrate_000", "substrate_000"])
        self.assertEqual(list(verbose["substrate_name"]), ["1_methyl_pyrrole", "1_methyl_pyrrole"])
        self.assertEqual(list(verbose["substrate_name_original"]), ["substrate_000", "substrate_000"])
        self.assertEqual(list(verbose["substrate_pubchem_iupac"]), ["1 methyl pyrrole", "1 methyl pyrrole"])
        self.assertEqual(list(verbose["substrate_pubchem_cid"]), [12345, 12345])
        self.assertEqual(list(verbose["substrate_pubchem_status"]), ["success", "success"])

    def test_map_substrate_names_uses_cache_and_force_requeries(self):
        df = pd.DataFrame(
            {
                "substrate_smiles": ["C1=CC=CO1"],
                "substrate_name": ["substrate_000"],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "pubchem.csv"
            with patch(
                "frust.utils.mols.lookup_pubchem_name",
                return_value={
                    "input_smiles": "C1=CC=CO1",
                    "canonical_smiles": "c1ccoc1",
                    "pubchem_iupac": "furan",
                    "pubchem_cid": 8029,
                    "lookup_status": "success",
                    "lookup_error": None,
                },
            ) as lookup:
                first = map_substrate_names(df, cache_path=cache_path)

            with patch(
                "frust.utils.mols.lookup_pubchem_name",
                side_effect=AssertionError("cache was not used"),
            ) as lookup_cached:
                second = map_substrate_names(df, cache_path=cache_path)

            with patch(
                "frust.utils.mols.lookup_pubchem_name",
                return_value={
                    "input_smiles": "C1=CC=CO1",
                    "canonical_smiles": "c1ccoc1",
                    "pubchem_iupac": "oxole",
                    "pubchem_cid": 8029,
                    "lookup_status": "success",
                    "lookup_error": None,
                },
            ) as lookup_forced:
                forced = map_substrate_names(df, cache_path=cache_path, force=True)

        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(lookup_cached.call_count, 0)
        self.assertEqual(lookup_forced.call_count, 1)
        self.assertEqual(first["substrate_name"].iloc[0], "furan")
        self.assertEqual(second["substrate_name"].iloc[0], "furan")
        self.assertEqual(forced["substrate_name"].iloc[0], "oxole")

    def test_map_substrate_names_handles_failed_lookups_and_strict_mode(self):
        df = pd.DataFrame(
            {
                "substrate_smiles": ["not_a_smiles"],
                "substrate_name": ["substrate_000"],
            }
        )

        def fake_lookup(smiles):
            return {
                "input_smiles": smiles,
                "canonical_smiles": smiles,
                "pubchem_iupac": None,
                "pubchem_cid": None,
                "lookup_status": "not_found",
                "lookup_error": None,
            }

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "pubchem.csv"
            with patch("frust.utils.mols.lookup_pubchem_name", side_effect=fake_lookup):
                out = map_substrate_names(
                    df,
                    cache_path=cache_path,
                    add_metadata=True,
                    original_col="substrate_name_original",
                )

            with self.assertRaisesRegex(ValueError, "PubChem lookup failed"):
                map_substrate_names(df, cache_path=cache_path, strict=True)

        self.assertEqual(out["substrate_name"].iloc[0], "substrate_000")
        self.assertEqual(out["substrate_name_original"].iloc[0], "substrate_000")
        self.assertIsNone(out["substrate_pubchem_iupac"].iloc[0])
        self.assertEqual(out["substrate_pubchem_status"].iloc[0], "not_found")

    def test_map_substrate_names_can_return_mapping_table(self):
        df = pd.DataFrame(
            {
                "substrate_smiles": ["C1=CC=CO1", "CN1C=CC=C1"],
                "substrate_name": ["substrate_000", "substrate_001"],
            }
        )

        def fake_lookup(smiles):
            name = "furan" if smiles == "C1=CC=CO1" else "1-methylpyrrole"
            cid = 8029 if smiles == "C1=CC=CO1" else 11772
            return {
                "input_smiles": smiles,
                "canonical_smiles": smiles,
                "pubchem_iupac": name,
                "pubchem_cid": cid,
                "lookup_status": "success",
                "lookup_error": None,
            }

        with tempfile.TemporaryDirectory() as td:
            with patch("frust.utils.mols.lookup_pubchem_name", side_effect=fake_lookup):
                out, mapping = map_substrate_names(
                    df,
                    cache_path=Path(td) / "pubchem.csv",
                    return_mapping=True,
                )

        self.assertEqual(list(out["substrate_name"]), ["furan", "1-methylpyrrole"])
        self.assertEqual(set(mapping["substrate_smiles"]), {"C1=CC=CO1", "CN1C=CC=C1"})
        self.assertIn("cache_hit", mapping.columns)
        self.assertTrue(mapping["lookup_status"].eq("success").all())

    def test_map_substrate_names_validates_smiles_column(self):
        with self.assertRaisesRegex(ValueError, "substrate_smiles"):
            map_substrate_names(pd.DataFrame({"substrate_name": ["x"]}))


if __name__ == "__main__":
    unittest.main()
