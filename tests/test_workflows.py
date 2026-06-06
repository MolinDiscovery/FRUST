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
    def test_mols_per_rpos_targets_use_prepared_molecule_payloads(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2,3"]})
        with patch("frust.workflows.factories.create_mol_per_rpos", return_value=_mol_jobs()) as create:
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", select_mols="int2")
            targets = wf.targets()

        create.assert_called_once()
        self.assertEqual([target.tag for target in targets], ["int2_rpos_2", "int2_rpos_3"])
        self.assertEqual(targets[0].payload, _mol_jobs()[0])

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
                "dft_pre_sp",
                "dft_opt",
                "freq",
                "solv",
            ],
        )
        self.assertEqual(
            list(stages["group"]),
            ["init", "init", "init", "init", "init", "dft_opt", "freq", "solv"],
        )
        self.assertNotIn("hess", stages["stage"].tolist())
        self.assertNotIn("optts", stages["stage"].tolist())
        self.assertEqual(stages.loc[stages["stage"].eq("xtb_sp"), "options"].item(), "gfn=2")

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
                "xtb_preopt",
                "xtb_sp",
                "xtb_opt",
                "dft_pre_sp",
                "dft_pre_opt",
                "hess",
                "optts",
                "freq",
                "solv",
            ],
        )
        self.assertEqual(
            list(stages["group"]),
            ["init", "init", "init", "init", "init", "init", "hess", "optts", "freq", "solv"],
        )
        optts = stages.loc[stages["stage"].eq("optts")].iloc[0]
        self.assertEqual(optts["calculation"], "OptTS")
        self.assertEqual(optts["method_key"], "optts")

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
            ["init", "xtb_preopt", "xtb_sp", "xtb_opt", "dft_pre_sp", "dft_opt", "freq", "solv"],
        )
        self.assertEqual(non_dft_default["group"].unique().tolist(), ["single_job"])
        self.assertIn("filter", non_dft_default["stage"].tolist())

    def test_show_stages_reflects_replaced_method_stage(self):
        df = pd.DataFrame({"compound_name": ["raw"], "smiles": ["CCO"]})
        method = methods.preset("r2scan-3c").replace(xtb_sp=methods.gxtb(job="sp"))
        wf = ft.workflows.raw_mols(dataframe=df, method=method, dft=False)

        stages = wf.show_stages()

        self.assertEqual(stages.loc[stages["stage"].eq("xtb_sp"), "engine"].item(), "gxtb")

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

    def test_local_run_dispatches_gxtb_stage_and_compacts_successful_target(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2"]})
        method = methods.preset("r2scan-3c").replace(
            xtb_sp=methods.gxtb(job="sp"),
            xtb_opt=methods.gxtb(job="opt"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("frust.workflows.factories.create_mol_per_rpos", return_value=[_mol_jobs()[0]]),
                patch("frust.workflows.factories.Stepper", FakeStepper),
                patch("frust.workflows.core.Stepper", FakeStepper),
            ):
                wf = ft.workflows.mols(dataframe=df, split="per_rpos", method=method, dft=True)
                out = wf.run(
                    targets=[0],
                    out_dir=tmp,
                    execution="dft_staged",
                    n_cores=3,
                    mem_gb=9,
                )

            target_dir = Path(tmp) / "int2_rpos_2"
            self.assertFalse((target_dir / "init.parquet").exists())
            self.assertFalse((target_dir / "init.dft_opt.parquet").exists())
            self.assertTrue((target_dir / "init.dft_opt.freq.solv.parquet").exists())
            self.assertTrue((target_dir / "timing.json").exists())
            self.assertFalse((target_dir / "init.timing.json").exists())
            timing_payload = json.loads((target_dir / "timing.json").read_text())
            self.assertEqual(timing_payload["target"], "int2_rpos_2")
            self.assertEqual(
                [record["group"] for record in timing_payload["groups"]],
                ["init", "dft_opt", "freq", "solv"],
            )
            self.assertIn("prepare", [record["stage"] for record in timing_payload["stages"]])
            collected = wf.collect(tmp)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(collected), 1)
        workflow_timing = ft.show_timing(out, detail="workflow")
        self.assertIn("group", workflow_timing["kind"].tolist())
        self.assertIn("prepare", workflow_timing["kind"].tolist())
        engines = [call[0] for call in FakeStepper.calls]
        self.assertIn("gxtb", engines)
        gxtb_calls = [call for call in FakeStepper.calls if call[0] == "gxtb"]
        self.assertEqual(gxtb_calls[0][2], {})

    def test_raw_mols_local_run_can_keep_all_staged_parquets(self):
        df = pd.DataFrame({"compound_name": ["raw dimer"], "smiles": ["CCO"]})
        method = methods.preset("r2scan-3c").replace(
            xtb_sp=methods.gxtb(job="sp"),
            xtb_opt=methods.gxtb(job="opt"),
        )
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
            self.assertTrue((target_dir / "init.dft_opt.freq.solv.parquet").exists())
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
                if name == "DFT-Opt":
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
            self.assertFalse((target_dir / "init.dft_opt.freq.solv.parquet").exists())

    def test_submit_dft_staged_submits_dependent_groups(self):
        df = pd.DataFrame({"smiles": ["CN1C=CC=C1"], "rpos": ["2"]})
        fake = FakeExecutor()
        cluster = ClusterConfig(backend="slurm", partition="kemi1", log_dir="logs/workflow-test")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("frust.workflows.factories.create_mol_per_rpos", return_value=[_mol_jobs()[0]]),
            patch("frust.workflows.core.create_executor", return_value=fake),
        ):
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", dft=True)
            result = wf.submit(
                out_dir=tmp,
                cluster=cluster,
                execution="dft_staged",
                collect=False,
                stage_resources={
                    "init": Resources(cpus=5, mem_gb=11, timeout_min=120),
                    "dft_opt": Resources(cpus=7, mem_gb=13, timeout_min=240),
                    "freq": Resources(cpus=3, mem_gb=8, timeout_min=180),
                    "solv": Resources(cpus=3, mem_gb=6, timeout_min=60),
                },
            )

        self.assertEqual(result.mode, "mols:dft_staged")
        self.assertEqual(result.tags, ["int2_rpos_2"])
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
                    "freq": Resources(cpus=3, mem_gb=8, timeout_min=180),
                    "solv": Resources(cpus=3, mem_gb=6, timeout_min=60),
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
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", dft=True)
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
                "int2_rpos_2": "init.dft_opt.freq.solv.parquet",
                "int2_rpos_3": "init.dft_opt.freq.solv.parquet",
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
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", dft=True)
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
            wf = ft.workflows.mols(dataframe=df, split="per_rpos", dft=True)
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="single_job", collect=False)

        self.assertEqual(result.mode, "mols:single_job")
        self.assertEqual(len(result.tags), 2)
        self.assertEqual(len(fake.submissions), 2)

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
            pd.DataFrame({"value": [2], "calc-NT": [False]}).to_parquet(bad_dir / "final.parquet")
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
