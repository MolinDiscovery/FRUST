from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import frust as ft
from frust.cluster import ClusterConfig, Resources
from frust.workflows import core as workflow_core
from frust.workflows import methods


CATALYST = "CC1(C)CCCC(C)(C)N1C2=CC=CC=C2B"


def _mol_jobs():
    return [
        {"int2_rpos(2)": ("mol-r2", {"structure_type": "MOL", "rpos": 2})},
        {"int2_rpos(3)": ("mol-r3", {"structure_type": "MOL", "rpos": 3})},
    ]


def _screen_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["substrate", "catalyst"],
            "smiles": ["CN1C=CC=C1", CATALYST],
            "compound_name": ["pyrrole", "cat"],
            "rpos": ["2,3", None],
        }
    )


def _initial_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "structure_id": ["MOL:pyrrole:int2:r2", "MOL:pyrrole:int2:r2"],
            "custom_name": ["int2_rpos(2)", "int2_rpos(2)"],
            "substrate_name": ["pyrrole", "pyrrole"],
            "structure_type": ["MOL", "MOL"],
            "molecule_role": ["int2", "int2"],
            "rpos": [2, 2],
            "cid": [0, 1],
            "atoms": [["H"], ["H"]],
            "coords_embedded": [[(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)]],
        }
    )


def _initial_df_with_conformer_attrs() -> pd.DataFrame:
    df = _initial_df()
    df.attrs["frust_conformers"] = {
        "schema_version": 1,
        "source": "screen.create_ts_guesses",
        "backend": "tsguess2",
        "requested_n_confs": 2,
        "n_cores": 2,
        "n_structures": 1,
        "total_generated_confs": 2,
        "structures": [
            {
                "structure_id": "TS1:pyrrole:cat:r2",
                "requested_n_confs": 2,
                "resolved_n_confs": 2,
                "generated_n_confs": 2,
            }
        ],
    }
    return df


class FakeStepper:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeStepper.calls.append(("init", kwargs))

    def build_initial_df(self, *args, **kwargs):
        FakeStepper.calls.append(("build_initial_df", args, kwargs))
        return _initial_df()

    def xtb(self, df, name, options, lowest=None, constraint=False, **kwargs):
        FakeStepper.calls.append(("xtb", name, options, lowest, constraint, kwargs))
        out = df.copy()
        if lowest:
            out = out.sort_values("cid").head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = [float(value) for value in range(len(out), 0, -1)]
        if "opt" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def gxtb(self, df, name, options, lowest=None, constraint=False, **kwargs):
        FakeStepper.calls.append(("gxtb", name, options, lowest, constraint, kwargs))
        out = df.copy()
        if lowest:
            out = out.sort_values("cid").head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = [float(value) for value in range(len(out), 0, -1)]
        if "opt" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def orca(self, df, name, options, lowest=None, **kwargs):
        FakeStepper.calls.append(("orca", name, options, lowest, kwargs))
        out = df.copy()
        if lowest:
            out = out.sort_values("cid").head(lowest).copy()
        out[f"{name}-NT"] = True
        out[f"{name}-EE"] = [float(value) for value in range(len(out), 0, -1)]
        if "Opt" in options or "OptTS" in options:
            out[f"{name}-oc"] = out["coords_embedded"]
        return out

    def prune_conformers(self, df, name="initial_prune", **kwargs):
        FakeStepper.calls.append(("prune_conformers", name, kwargs))
        out = df.sort_values("cid").head(1).copy()
        out.attrs.update(getattr(df, "attrs", {}))
        out.attrs.setdefault("frust_steps", {})[name] = {
            "engine": "prism_pruner",
            "columns": [],
            "options": kwargs,
            "row_counts": {
                "input_rows": len(df),
                "output_rows": len(out),
                "dropped_rows": len(df) - len(out),
            },
        }
        return out


class FakeExecutor:
    def __init__(self):
        self.parameters = []
        self.submissions = []

    def update_parameters(self, **kwargs):
        self.parameters.append(kwargs)

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        return SimpleNamespace(job_id=f"job-{len(self.submissions)}")


class WorkflowTargetTests(unittest.TestCase):
    def test_mols_per_rpos_targets_are_lightweight_structure_plans(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2,3"]})
        with patch("frust.workflows.factories.create_mol_per_rpos", return_value=_mol_jobs()) as create:
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", select_mols="int2")
            targets = wf.targets()

        create.assert_not_called()
        self.assertEqual(
            [target.tag for target in targets],
            [
                "int2__substrate_000__frust_catalyst__r2",
                "int2__substrate_000__frust_catalyst__r3",
            ],
        )
        self.assertEqual([target.state_id for target in targets], ["int2", "int2"])
        self.assertEqual([target.rpos for target in targets], [2, 3])

    def test_raw_mols_targets_use_exact_smiles_payloads(self):
        df = pd.DataFrame(
            {
                "compound_name": ["cat A", "cat_A", "cat A"],
                "smiles": ["CCO", "CCN", "CCC"],
            }
        )
        with patch("frust.workflows.factories.create_mol_per_rpos") as create:
            wf = ft.workflows.raw_mols(dataframe=df)
            targets = wf.targets()

        create.assert_not_called()
        self.assertEqual([target.tag for target in targets], ["cat_A", "cat_A_001", "cat_A_002"])
        self.assertEqual([target.payload.loc[0, "smiles"] for target in targets], ["CCO", "CCN", "CCC"])
        self.assertEqual(
            [target.payload.loc[0, "substrate_name"] for target in targets],
            ["cat A", "cat_A", "cat A"],
        )

    def test_screen_ts_targets_expand_ts_type_system_and_rpos(self):
        wf = ft.workflows.screen_ts(dataframe=_screen_df(), ts_types=["TS1", "TS4"])
        targets = wf.targets()

        self.assertEqual(
            [target.tag for target in targets],
            [
                "TS1__pyrrole__cat__r2",
                "TS1__pyrrole__cat__r3",
                "TS4__pyrrole__cat__r2",
                "TS4__pyrrole__cat__r3",
            ],
        )

    def test_screen_ts_rejects_removed_tsguess3_backend(self):
        wf = ft.workflows.screen_ts(
            dataframe=_screen_df(),
            ts_types=["TS3"],
            ts_backend="tsguess3",
        )

        with self.assertRaisesRegex(ValueError, "'tsguess2' or 'tsguess'"):
            wf.targets()


class WorkflowExecutionTests(unittest.TestCase):
    def setUp(self):
        FakeStepper.calls = []

    def test_raw_mols_show_stages_lists_active_molecule_stages(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(dataframe=df, method="r2scan-3c", dft=True)

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"]),
            [
                "prepare",
                "xtb_preopt",
                "xtb_sp",
                "xtb_opt",
                "dft_rank_sp",
                "dft_opt",
                "dft_freq",
                "dft_solv_sp",
            ],
        )
        self.assertEqual(
            list(stages["group"]),
            ["init", "init", "init", "init", "init", "dft_opt", "dft_freq", "dft_solv_sp"],
        )
        self.assertNotIn("dft_hessian", stages["stage"].tolist())
        self.assertNotIn("dft_ts_opt", stages["stage"].tolist())
        xtb_sp = stages.loc[stages["stage"].eq("xtb_sp")].iloc[0]
        self.assertEqual(xtb_sp["engine"], "gxtb")
        self.assertIsNone(xtb_sp["options"])
        xtb_opt = stages.loc[stages["stage"].eq("xtb_opt")].iloc[0]
        self.assertEqual(xtb_opt["engine"], "gxtb")
        self.assertEqual(xtb_opt["options"], "opt")

    def test_screen_ts_show_stages_lists_ts_dft_stages(self):
        wf = ft.workflows.screen_ts(
            dataframe=_screen_df(),
            ts_types=["TS1"],
            method="r2scan-3c",
            dft=True,
        )

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"]),
            [
                "prepare",
                "initial_prune",
                "xtb_preopt",
                "xtb_sp",
                "xtb_opt",
                "dft_rank_sp",
                "dft_preopt",
                "dft_hessian",
                "dft_ts_opt",
                "dft_freq",
                "dft_solv_sp",
            ],
        )
        self.assertEqual(
            list(stages["group"]),
            ["init", "init", "init", "init", "init", "init", "init", "dft_hessian", "dft_ts_opt", "dft_freq", "dft_solv_sp"],
        )
        optts = stages.loc[stages["stage"].eq("dft_ts_opt")].iloc[0]
        self.assertEqual(optts["calculation"], "DFT transition-state optimization")
        self.assertEqual(optts["method_key"], "dft_ts_opt")

    def test_solvent_inclusive_preset_omits_terminal_solvent_stage_for_mols(self):
        wf = ft.workflows.mols(
            dataframe=pd.DataFrame({"smiles": ["CCO"]}),
            method="r2scan-3c-solv",
            dft=True,
        )

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"])[-3:],
            ["dft_rank_sp", "dft_opt", "dft_freq"],
        )
        self.assertNotIn("dft_solv_sp", stages["stage"].tolist())

    def test_solvent_inclusive_preset_omits_terminal_solvent_stage_for_screen_ts(self):
        wf = ft.workflows.screen_ts(
            dataframe=_screen_df(),
            ts_types=["TS1"],
            method="r2scan-3c-solv",
            dft=True,
        )

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"])[-4:],
            ["dft_preopt", "dft_hessian", "dft_ts_opt", "dft_freq"],
        )
        self.assertNotIn("dft_solv_sp", stages["stage"].tolist())
        dft_stages = stages.loc[stages["stage"].str.startswith("dft_")]
        self.assertEqual(set(dft_stages["solvent"]), {"SMD(chloroform)"})

    def test_solvent_inclusive_preset_omits_terminal_solvent_stage_for_int3(self):
        wf = ft.workflows.int3(
            dataframe=_screen_df(),
            method="r2scan-3c-solv",
            dft=True,
        )

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"])[-3:],
            ["dft_preopt", "dft_opt", "dft_freq"],
        )
        self.assertNotIn("dft_solv_sp", stages["stage"].tolist())

    def test_solvent_inclusive_preset_uses_frequency_energy_as_final_result(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        with (
            patch("frust.workflows.factories.Stepper", FakeStepper),
            patch("frust.workflows.core.Stepper", FakeStepper),
        ):
            wf = ft.workflows.raw_mols(
                dataframe=df,
                method="r2scan-3c-solv",
                dft=True,
            )
            out = wf.run(targets=[0], n_cores=2, mem_gb=4)

        orca_calls = [call for call in FakeStepper.calls if call[0] == "orca"]
        self.assertEqual(
            [call[1] for call in orca_calls],
            ["dft_rank_sp", "dft_opt", "dft_freq"],
        )
        self.assertEqual(ft.result_column(out), "dft_freq-EE")

    def test_show_stages_full_exposes_planned_inputs_and_stage_controls(self):
        method = methods.preset("r2scan-3c-solv").replace(
            dft_ts_opt=methods.orca_composite(
                "r2SCAN-3c",
                job="optts",
                solvent="chloroform",
                uma="omol",
            )
        )
        wf = ft.workflows.screen_ts(
            dataframe=_screen_df(),
            ts_types=["TS1"],
            method=method,
            dft=True,
        )

        stages = wf.show_stages(detail="full")

        self.assertIn("xtra_inp_str", stages.columns)
        self.assertIn("calculator_kwargs", stages.columns)
        rank_sp = stages.loc[stages["stage"].eq("dft_rank_sp")].iloc[0]
        self.assertEqual(
            rank_sp["xtra_inp_str"],
            '%CPCM\\nSMD TRUE\\nSMDSOLVENT "chloroform"\\nend',
        )
        hessian = stages.loc[stages["stage"].eq("dft_hessian")].iloc[0]
        self.assertEqual(hessian["read_files"], '["input.hess"]')
        optts = stages.loc[stages["stage"].eq("dft_ts_opt")].iloc[0]
        self.assertTrue(optts["use_last_hess"])
        self.assertEqual(optts["calculator_kwargs"], '{"uma": "omol"}')
        prune = stages.loc[stages["stage"].eq("initial_prune")].iloc[0]
        self.assertIn('"modes": ["moi", "rmsd"]', prune["prune_options"])

    def test_show_stages_rejects_unknown_detail(self):
        wf = ft.workflows.raw_mols(smiles=["CCO"])

        with self.assertRaisesRegex(ValueError, "detail must be 'summary' or 'full'"):
            wf.show_stages(detail="verbose")

    def test_screen_ts_dft_false_stops_after_dft_pre_sp_filter(self):
        wf = ft.workflows.screen_ts(
            dataframe=_screen_df(),
            ts_types=["TS1"],
            method="r2scan-3c",
            dft=False,
        )

        stages = wf.show_stages()

        self.assertEqual(
            list(stages["stage"]),
            [
                "prepare",
                "initial_prune",
                "xtb_preopt",
                "xtb_sp",
                "xtb_opt",
                "dft_rank_sp",
                "filter",
            ],
        )
        self.assertNotIn("dft_preopt", stages["stage"].tolist())
        self.assertNotIn("dft_hessian", stages["stage"].tolist())
        self.assertEqual(stages.loc[stages["stage"].eq("filter"), "rank_by"].item(), "dft_rank_sp")
        self.assertEqual(stages.loc[stages["stage"].eq("filter"), "kind"].item(), "filter")

    def test_show_stages_execution_grouping_modes(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(dataframe=df, dft=True)
        non_dft = ft.workflows.raw_mols(dataframe=df, dft=False)

        single_job = wf.show_stages(execution="single_job")
        fully_staged = wf.show_stages(execution="fully_staged")
        non_dft_default = non_dft.show_stages()

        self.assertEqual(single_job["group"].unique().tolist(), ["single_job"])
        self.assertEqual(
            list(fully_staged["group"]),
            ["init", "xtb_preopt", "xtb_sp", "xtb_opt", "dft_rank_sp", "dft_opt", "dft_freq", "dft_solv_sp"],
        )
        self.assertEqual(non_dft_default["group"].unique().tolist(), ["single_job"])
        self.assertIn("filter", non_dft_default["stage"].tolist())

    def test_show_stages_reflects_replaced_method_stage(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        method = methods.preset("r2scan-3c").replace(xtb_sp=methods.xtb(gfn=1))
        wf = ft.workflows.raw_mols(dataframe=df, method=method, dft=False)

        stages = wf.show_stages()

        xtb_sp = stages.loc[stages["stage"].eq("xtb_sp")].iloc[0]
        self.assertEqual(xtb_sp["engine"], "xtb")
        self.assertEqual(xtb_sp["options"], "gfn=1")

    def test_prune_initial_adds_pruning_stage_after_prepare(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(dataframe=df, dft=True, prune_initial=True)

        stages = wf.show_stages()

        self.assertEqual(list(stages["stage"])[:3], ["prepare", "initial_prune", "xtb_preopt"])
        self.assertEqual(stages.loc[stages["stage"].eq("initial_prune"), "group"].item(), "init")
        self.assertEqual(
            stages.loc[stages["stage"].eq("initial_prune"), "engine"].item(),
            "prism_pruner",
        )
        self.assertIn("modes", stages.loc[stages["stage"].eq("initial_prune"), "options"].item())

    def test_prune_initial_false_preserves_stage_graph(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(dataframe=df, dft=True, prune_initial=False)

        stages = wf.show_stages()

        self.assertNotIn("initial_prune", stages["stage"].tolist())

    def test_prune_initial_dict_overrides_defaults(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(
            dataframe=df,
            dft=True,
            prune_initial={"modes": ("moi",), "moi_max_deviation": 0.03},
        )

        stage = wf.show_stages().loc[lambda table: table["stage"].eq("initial_prune")].iloc[0]

        self.assertIn("moi_max_deviation=0.03", stage["options"])
        self.assertNotIn("rmsd_max_rmsd=0.25", stage["options"])

    def test_show_stages_does_not_build_targets(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        wf = ft.workflows.raw_mols(dataframe=df, dft=True)

        with patch.object(wf, "_build_targets", side_effect=AssertionError("targets built")):
            stages = wf.show_stages()

        self.assertEqual(stages["stage"].iloc[0], "prepare")

    def test_show_stages_missing_method_stage_raises_key_error(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        method = methods.MethodPlan(
            name="missing",
            stages={"xtb_preopt": methods.xtb(gfn=2)},
        )
        wf = ft.workflows.raw_mols(dataframe=df, method=method, dft=True)

        with self.assertRaisesRegex(KeyError, "xtb_sp"):
            wf.show_stages()

    def test_local_run_prune_initial_calls_stepper_pruning(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        with (
            patch("frust.workflows.factories.Stepper", FakeStepper),
            patch("frust.workflows.core.Stepper", FakeStepper),
        ):
            wf = ft.workflows.raw_mols(
                dataframe=df,
                dft=False,
                prune_initial={"modes": ("moi",), "moi_max_deviation": 0.03},
            )
            out = wf.run(targets=[0], n_cores=2, mem_gb=4)

        prune_calls = [call for call in FakeStepper.calls if call[0] == "prune_conformers"]
        self.assertEqual(len(prune_calls), 1)
        self.assertEqual(prune_calls[0][1], "initial_prune")
        self.assertEqual(prune_calls[0][2]["moi_max_deviation"], 0.03)
        self.assertEqual(len(out), 1)

    def test_screen_ts_dft_false_run_calls_only_dft_pre_sp_before_filter(self):
        with (
            patch("frust.workflows.factories.create_ts_guesses", return_value={"TS1": _initial_df()}),
            patch("frust.workflows.factories.Stepper", FakeStepper),
            patch("frust.workflows.core.Stepper", FakeStepper),
        ):
            wf = ft.workflows.screen_ts(
                dataframe=_screen_df(),
                ts_types=["TS1"],
                method="r2scan-3c",
                dft=False,
                prune_initial=False,
            )
            out = wf.run(targets=[0], n_cores=2, mem_gb=4)

        orca_calls = [call for call in FakeStepper.calls if call[0] == "orca"]
        self.assertEqual([call[1] for call in orca_calls], ["dft_rank_sp"])
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out["state_id"]), {"TS1"})
        self.assertEqual(set(out["state_kind"]), {"transition_state"})
        self.assertEqual(ft.result_column(out), "dft_rank_sp-EE")

    def test_workflow_logs_prepare_and_filter_summaries_without_duplicate_prune(self):
        messages = []
        logger = SimpleNamespace(info=messages.append)
        with (
            patch(
                "frust.workflows.factories.create_ts_guesses",
                return_value={"TS1": _initial_df_with_conformer_attrs()},
            ),
            patch("frust.workflows.factories.Stepper", FakeStepper),
            patch("frust.workflows.core.Stepper", FakeStepper),
            patch("frust.workflows.core.make_stepper_logger", return_value=logger),
        ):
            wf = ft.workflows.screen_ts(
                dataframe=_screen_df(),
                ts_types=["TS1"],
                method="r2scan-3c",
                dft=False,
                prune_initial=True,
            )
            wf.run(targets=[0], n_cores=2, mem_gb=4)

        self.assertIn(
            "[prepare] generated 2 conformer row(s) from 1 structure(s); "
            "requested=2; resolved=2; missing=0; backend=tsguess2",
            messages,
        )
        self.assertIn("[filter] kept 1/1 row(s); dropped=0", messages)
        self.assertFalse(any(message.startswith("[initial_prune]") for message in messages))

    def test_local_run_dispatches_gxtb_stage_and_compacts_successful_target(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2"]})
        method = methods.preset("r2scan-3c")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("frust.workflows.factories.create_mol_per_rpos", return_value=[_mol_jobs()[0]]),
                patch("frust.workflows.factories.Stepper", FakeStepper),
                patch("frust.workflows.core.Stepper", FakeStepper),
            ):
                wf = ft.workflows.mols(
                    dataframe=df,
                    split="per_rpos",
                    select_mols="int2",
                    method=method,
                    dft=True,
                    prune_initial=False,
                )
                out = wf.run(
                    targets=[0],
                    out_dir=tmp,
                    execution="dft_staged",
                    n_cores=3,
                    mem_gb=9,
                )

            target_dir = Path(tmp) / "int2__substrate_000__frust_catalyst__r2"
            self.assertFalse((target_dir / "init.parquet").exists())
            self.assertFalse((target_dir / "init.dft_opt.parquet").exists())
            self.assertTrue(
                (target_dir / "init.dft_opt.dft_freq.dft_solv_sp.parquet").exists()
            )
            self.assertTrue((target_dir / "tier_low_cost.parquet").exists())
            self.assertTrue((target_dir / "tier_dft_ranked.parquet").exists())
            low_cost_tier = pd.read_parquet(target_dir / "tier_low_cost.parquet")
            ranked_tier = pd.read_parquet(target_dir / "tier_dft_ranked.parquet")
            final_tier = pd.read_parquet(
                target_dir / "init.dft_opt.dft_freq.dft_solv_sp.parquet"
            )
            self.assertEqual(low_cost_tier["cid"].tolist(), [1])
            self.assertEqual(ranked_tier["cid"].tolist(), [1])
            self.assertEqual(final_tier["cid"].tolist(), [0])
            self.assertEqual(ft.result_column(low_cost_tier), "xtb_opt-EE")
            self.assertEqual(ft.result_column(ranked_tier), "dft_rank_sp-EE")
            self.assertTrue((target_dir / "timing.json").exists())
            self.assertFalse((target_dir / "init.timing.json").exists())
            timing_payload = json.loads((target_dir / "timing.json").read_text())
            self.assertEqual(
                timing_payload["target"], "int2__substrate_000__frust_catalyst__r2"
            )
            self.assertEqual(
                [record["group"] for record in timing_payload["groups"]],
                ["init", "dft_opt", "dft_freq", "dft_solv_sp"],
            )
            self.assertIn("prepare", [record["stage"] for record in timing_payload["stages"]])
            collected = wf.collect(tmp)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(collected), 1)
        self.assertEqual(ft.result_column(out), "dft_solv_sp-EE")
        self.assertIn("dft_solv_sp-EE", out.columns)
        self.assertEqual(out.attrs["frust_results"]["profile"], "minimum")
        workflow_timing = ft.show_timing(out, detail="workflow")
        self.assertIn("group", workflow_timing["kind"].tolist())
        self.assertIn("prepare", workflow_timing["kind"].tolist())
        engines = [call[0] for call in FakeStepper.calls]
        self.assertIn("gxtb", engines)
        gxtb_calls = [call for call in FakeStepper.calls if call[0] == "gxtb"]
        self.assertEqual(gxtb_calls[0][2], {})
        self.assertEqual(gxtb_calls[1][2], {"opt": None})

    def test_raw_mols_local_run_can_keep_all_staged_parquets(self):
        df = pd.DataFrame({"compound_name": ["raw dimer"], "smiles": ["CCO"]})
        method = methods.preset("r2scan-3c")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("frust.workflows.factories.create_mol_per_rpos") as create,
                patch("frust.workflows.factories.Stepper", FakeStepper),
                patch("frust.workflows.core.Stepper", FakeStepper),
            ):
                wf = ft.workflows.raw_mols(dataframe=df, method=method, dft=True)
                out = wf.run(
                    targets=[0],
                    out_dir=tmp,
                    execution="dft_staged",
                    n_cores=3,
                    mem_gb=9,
                    target_retention="all",
                )

            create.assert_not_called()
            target_dir = Path(tmp) / "raw_dimer"
            self.assertTrue((target_dir / "init.parquet").exists())
            self.assertTrue((target_dir / "init.dft_opt.parquet").exists())
            self.assertTrue(
                (target_dir / "init.dft_opt.dft_freq.dft_solv_sp.parquet").exists()
            )
            self.assertTrue((target_dir / "timing.json").exists())

        self.assertEqual(len(out), 1)
        engines = [call[0] for call in FakeStepper.calls]
        self.assertIn("gxtb", engines)
        build_calls = [call for call in FakeStepper.calls if call[0] == "build_initial_df"]
        payload = build_calls[0][1][0]
        self.assertEqual(payload.loc[0, "substrate_name"], "raw dimer")

    def test_failed_local_run_keeps_intermediate_parquets(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})

        class FailingStepper(FakeStepper):
            def orca(self, df, name, options, lowest=None, **kwargs):
                if name == "dft_opt":
                    raise RuntimeError("DFT failed")
                return super().orca(df, name, options, lowest=lowest, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("frust.workflows.factories.Stepper", FailingStepper),
                patch("frust.workflows.core.Stepper", FailingStepper),
            ):
                wf = ft.workflows.raw_mols(dataframe=df, dft=True)
                with self.assertRaisesRegex(RuntimeError, "DFT failed"):
                    wf.run(targets=[0], out_dir=tmp, execution="dft_staged")

            target_dir = Path(tmp) / "raw"
            self.assertTrue((target_dir / "init.parquet").exists())
            self.assertTrue((target_dir / "timing.json").exists())
            self.assertFalse(
                (target_dir / "init.dft_opt.dft_freq.dft_solv_sp.parquet").exists()
            )

    def test_submit_dft_staged_submits_dependent_groups(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.factories.create_mol_per_rpos", return_value=[_mol_jobs()[0]]),
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.mols(
                dataframe=df, split="per_rpos", select_mols="int2", dft=True
            )
            result = wf.submit(
                out_dir=tmp,
                cluster=cluster,
                execution="dft_staged",
                collect=False,
                stage_resources={
                    "init": Resources(cpus=5, mem_gb=11, timeout_min=120),
                    "dft_opt": Resources(cpus=7, mem_gb=13, timeout_min=240),
                    "dft_freq": Resources(cpus=3, mem_gb=8, timeout_min=180),
                    "dft_solv_sp": Resources(cpus=3, mem_gb=6, timeout_min=60),
                },
            )

        self.assertEqual(result.mode, "mols:dft_staged")
        self.assertEqual(
            result.tags, ["int2__substrate_000__frust_catalyst__r2"]
        )
        self.assertEqual(len(fake.submissions), 4)
        dependencies = [
            params.get("slurm_additional_parameters", {}).get("dependency")
            for params in fake.parameters
        ]
        self.assertEqual(dependencies[0], None)
        self.assertEqual(dependencies[1], "afterok:job-1")
        self.assertEqual(dependencies[2], "afterok:job-2")
        self.assertEqual(dependencies[3], "afterok:job-3")
        self.assertEqual(
            [params["mem_gb"] for params in fake.parameters],
            [11, 13, 8, 6],
        )
        self.assertEqual(
            [round(submission[1][6].mem_gb, 2) for submission in fake.submissions],
            [8.8, 10.4, 6.4, 4.8],
        )
        self.assertIsNone(result.collection_job_id)

    def test_raw_mols_submit_dft_staged_submits_dependent_groups(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.raw_mols(dataframe=df, dft=True)
            result = wf.submit(
                out_dir=tmp,
                cluster=cluster,
                execution="dft_staged",
                collect=False,
                stage_resources={
                    "init": Resources(cpus=5, mem_gb=11, timeout_min=120),
                    "dft_opt": Resources(cpus=7, mem_gb=13, timeout_min=240),
                    "dft_freq": Resources(cpus=3, mem_gb=8, timeout_min=180),
                    "dft_solv_sp": Resources(cpus=3, mem_gb=6, timeout_min=60),
                },
            )

        self.assertEqual(result.mode, "raw_mols:dft_staged")
        self.assertEqual(result.tags, ["raw"])
        self.assertEqual(len(fake.submissions), 4)
        dependencies = [
            params.get("slurm_additional_parameters", {}).get("dependency")
            for params in fake.parameters
        ]
        self.assertEqual(dependencies[0], None)
        self.assertEqual(dependencies[1], "afterok:job-1")
        self.assertEqual(dependencies[2], "afterok:job-2")
        self.assertEqual(dependencies[3], "afterok:job-3")
        self.assertIsNone(result.collection_job_id)

    def test_submit_defaults_to_collection_job_with_afterany_dependency(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2,3"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.factories.create_mol_per_rpos", return_value=_mol_jobs()),
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.mols(
                dataframe=df, split="per_rpos", select_mols="int2", dft=True
            )
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="dft_staged")

        self.assertEqual(len(fake.submissions), 9)
        self.assertEqual(len(result.job_ids), 8)
        self.assertEqual(result.collection_job_id, "job-9")
        self.assertEqual(result.collection_output, str(Path(tmp) / "merged.parquet"))
        self.assertEqual(result.collection_report, str(Path(tmp) / "collection_report.json"))
        dependencies = [
            params.get("slurm_additional_parameters", {}).get("dependency")
            for params in fake.parameters
        ]
        self.assertEqual(dependencies[-1], "afterany:job-4:job-8")
        collector_fn, collector_args, _ = fake.submissions[-1]
        self.assertEqual(collector_fn.__name__, "_collect_expected_outputs")
        self.assertEqual(
            collector_args[3],
            {
                "int2__substrate_000__frust_catalyst__r2": "init.dft_opt.dft_freq.dft_solv_sp.parquet",
                "int2__substrate_000__frust_catalyst__r3": "init.dft_opt.dft_freq.dft_solv_sp.parquet",
            },
        )

    def test_submit_dft_staged_uses_default_resources_when_stage_resources_omitted(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.factories.create_mol_per_rpos", return_value=[_mol_jobs()[0]]),
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.mols(
                dataframe=df, split="per_rpos", select_mols="int2", dft=True
            )
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="dft_staged", collect=False)

        self.assertEqual(result.mode, "mols:dft_staged")
        self.assertEqual(len(fake.submissions), 4)
        resource_params = [
            (params["cpus_per_task"], params["mem_gb"], params["timeout_min"])
            for params in fake.parameters
        ]
        self.assertEqual(
            resource_params,
            [(4, 20, 720), (4, 20, 720), (4, 20, 720), (4, 20, 720)],
        )

    def test_raw_mols_submit_uses_default_resources_when_stage_resources_omitted(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.raw_mols(dataframe=df, dft=True)
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="dft_staged", collect=False)

        self.assertEqual(result.mode, "raw_mols:dft_staged")
        self.assertEqual(len(fake.submissions), 4)
        resource_params = [
            (params["cpus_per_task"], params["mem_gb"], params["timeout_min"])
            for params in fake.parameters
        ]
        self.assertEqual(
            resource_params,
            [(4, 20, 720), (4, 20, 720), (4, 20, 720), (4, 20, 720)],
        )

    def test_submit_single_job_submits_one_job_per_target(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2,3"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.factories.create_mol_per_rpos", return_value=_mol_jobs()),
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.mols(
                dataframe=df, split="per_rpos", select_mols="int2", dft=True
            )
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="single_job", collect=False)

        self.assertEqual(result.mode, "mols:single_job")
        self.assertEqual(len(result.tags), 2)
        self.assertEqual(len(fake.submissions), 2)
        self.assertEqual([params["mem_gb"] for params in fake.parameters], [20, 20])
        self.assertEqual(
            [submission[1][3].mem_gb for submission in fake.submissions],
            [16.0, 16.0],
        )

    def test_collect_expected_outputs_writes_report_and_skips_failed_outputs(self):
        df = pd.DataFrame(
            {
                "compound_name": ["ok", "bad", "missing"],
                "smiles": ["CCO", "CCN", "CCC"],
            }
        )
        wf = ft.workflows.raw_mols(dataframe=df)
        targets = wf.targets()
        expected = {target.tag: "final.parquet" for target in targets}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_dir = root / "ok"
            bad_dir = root / "bad"
            missing_dir = root / "missing"
            ok_dir.mkdir()
            bad_dir.mkdir()
            missing_dir.mkdir()
            pd.DataFrame({"value": [1], "calc-NT": [True]}).to_parquet(ok_dir / "final.parquet")
            pd.DataFrame({"value": [1], "calc-NT": [True]}).to_parquet(ok_dir / "init.parquet")
            (ok_dir / "timing.json").write_text(json.dumps({"schema_version": 2, "groups": [], "stages": []}))
            pd.DataFrame(
                {
                    "value": [2],
                    "ts_type": ["TS2"],
                    "substrate_name": ["C6_wb97"],
                    "catalyst_name": ["NEt"],
                    "rpos": [2],
                    "cid": [42],
                    "OptTS-NT": [False],
                    "OptTS-error": ["RuntimeError: Orca calculation did not terminate normally"],
                    "OptTS-orca.out": ["ORCA finished by error termination in Startup"],
                }
            ).to_parquet(bad_dir / "final.parquet")
            pd.DataFrame({"value": [2], "calc-NT": [False]}).to_parquet(bad_dir / "init.parquet")
            pd.DataFrame({"value": [3], "calc-NT": [True]}).to_parquet(missing_dir / "init.parquet")

            merged = workflow_core._collect_expected_outputs(
                wf,
                targets,
                root,
                expected,
                root / "merged.parquet",
                root / "collection_report.json",
                True,
            )

            report = json.loads((root / "collection_report.json").read_text())

            self.assertEqual(len(merged), 1)
            self.assertEqual(report["n_targets"], 3)
            self.assertEqual(report["n_collected"], 1)
            self.assertEqual(report["n_skipped"], 1)
            self.assertEqual(report["n_missing"], 1)
            self.assertEqual(report["n_errored"], 0)
            self.assertEqual(report["n_failures"], 2)
            failure = report["failure_summary"][0]
            self.assertEqual(failure["target"], "bad")
            self.assertEqual(failure["ts_type"], "TS2")
            self.assertEqual(failure["substrate_name"], "C6_wb97")
            self.assertEqual(failure["catalyst_name"], "NEt")
            self.assertEqual(failure["rpos"], 2)
            self.assertEqual(failure["cid"], 42)
            self.assertEqual(failure["failed_stage"], "OptTS")
            self.assertEqual(failure["failed_nt_cols"], ["OptTS-NT"])
            self.assertEqual(failure["problem"], "failed_stage")
            self.assertIn("Orca calculation", failure["error"])
            self.assertIn("Startup", failure["backend_hint"])
            self.assertEqual(report["failure_summary"][1]["problem"], "missing_output")
            self.assertTrue(report["missing_files"][0].endswith("missing/final.parquet"))
            self.assertEqual(report["compaction"]["n_targets"], 1)
            self.assertEqual(report["compaction"]["n_removed_files"], 1)
            self.assertFalse((ok_dir / "init.parquet").exists())
            self.assertTrue((ok_dir / "final.parquet").exists())
            self.assertTrue((bad_dir / "init.parquet").exists())

    def test_collect_expected_outputs_writes_report_before_raising_when_empty(self):
        df = pd.DataFrame({"compound_name": ["bad"], "smiles": ["CCN"]})
        wf = ft.workflows.raw_mols(dataframe=df)
        targets = wf.targets()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_dir = root / "bad"
            bad_dir.mkdir()
            pd.DataFrame({"value": [2], "calc-NT": [False]}).to_parquet(bad_dir / "final.parquet")

            with self.assertRaisesRegex(FileNotFoundError, "No usable workflow outputs"):
                workflow_core._collect_expected_outputs(
                    wf,
                    targets,
                    root,
                    {"bad": "final.parquet"},
                    root / "merged.parquet",
                    root / "collection_report.json",
                    True,
                )

            report = json.loads((root / "collection_report.json").read_text())

        self.assertEqual(report["n_collected"], 0)
        self.assertEqual(report["n_skipped"], 1)
        self.assertEqual(report["n_failures"], 1)
        self.assertEqual(report["failure_summary"][0]["failed_stage"], "calc")

    def test_raw_mols_validates_input_table(self):
        with self.assertRaisesRegex(ValueError, "smiles"):
            ft.workflows.raw_mols(dataframe=pd.DataFrame({"compound_name": ["raw"]})).targets()

        with self.assertRaisesRegex(ValueError, "missing SMILES"):
            ft.workflows.raw_mols(
                dataframe=pd.DataFrame({"compound_name": ["raw"], "smiles": [pd.NA]})
            ).targets()

    def test_raw_mols_invalid_smiles_fails_during_prepare(self):
        wf = ft.workflows.raw_mols(
            dataframe=pd.DataFrame({"compound_name": ["bad"], "smiles": ["not_a_smiles"]})
        )

        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            wf.run(targets=[0], execution="single_job", n_cores=1, mem_gb=1)


if __name__ == "__main__":
    unittest.main()
