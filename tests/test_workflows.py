from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import frust as ft
from frust.cluster import ClusterConfig, Resources
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

    def test_local_run_dispatches_gxtb_stage_and_writes_staged_parquets(self):
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
            self.assertTrue((target_dir / "init.parquet").exists())
            self.assertTrue((target_dir / "init.dft_opt.parquet").exists())
            self.assertTrue((target_dir / "init.dft_opt.solv.parquet").exists())
            collected = wf.collect(tmp)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(collected), 1)
        engines = [call[0] for call in FakeStepper.calls]
        self.assertIn("gxtb", engines)
        gxtb_calls = [call for call in FakeStepper.calls if call[0] == "gxtb"]
        self.assertEqual(gxtb_calls[0][2], {})

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
                stage_resources={
                    "init": Resources(cpus=5, mem_gb=11, timeout_min=120),
                    "dft_opt": Resources(cpus=7, mem_gb=13, timeout_min=240),
                    "solv": Resources(cpus=3, mem_gb=6, timeout_min=60),
                },
            )

        self.assertEqual(result.mode, "mols:dft_staged")
        self.assertEqual(result.tags, ["int2_rpos_2"])
        self.assertEqual(len(fake.submissions), 3)
        dependencies = [
            params.get("slurm_additional_parameters", {}).get("dependency")
            for params in fake.parameters
        ]
        self.assertEqual(dependencies[0], None)
        self.assertEqual(dependencies[1], "afterok:job-1")
        self.assertEqual(dependencies[2], "afterok:job-2")

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
            result = wf.submit(out_dir=tmp, cluster=cluster, execution="single_job")

        self.assertEqual(result.mode, "mols:single_job")
        self.assertEqual(len(result.tags), 2)
        self.assertEqual(len(fake.submissions), 2)


if __name__ == "__main__":
    unittest.main()
